"""Trace event records and per-iteration time accounting."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, replace
from math import isfinite
import math
import sys
from typing import Mapping
from TokenSim.trace.occupancy import occupied_until, unique_executions

from TokenSim.trace.events import TraceEvent, register_records
from TokenSim.af.providers.queries import (
    AFEndpoint,
    AFLayerType,
    AFRoutingEdge,
    AFTensorLayout,
    AFTensorSlice,
)


class _ResourceBusyTime:
    """Incremental float sum, matching Python's sum used by the original summary.

    Python 3.12+ uses compensated summation. Plain += would change exported
    utilization in the last bits even when events and their order are identical.
    """

    __slots__ = ("total", "correction")

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        combined = self.total + value
        if sys.version_info >= (3, 12):
            if abs(self.total) >= abs(value):
                self.correction += (self.total - combined) + value
            else:
                self.correction += (value - combined) + self.total
        self.total = combined

    def value(self) -> float:
        if self.correction and isfinite(self.correction):
            return self.total + self.correction
        return self.total


@dataclass(frozen=True)
class AFTraceEvent(TraceEvent):
    dataset_version: str = "test-stub"
    source_domain: str | None = None
    target_domain: str | None = None
    logical_participants: tuple[str, ...] = ()
    participant_ranks: tuple[int, ...] = ()
    physical_endpoints: tuple[str, ...] = ()
    tp_group: tuple[int, ...] = ()
    ep_group: tuple[int, ...] = ()
    local_work: Mapping[str, object] = field(default_factory=dict)
    layer_type: AFLayerType | None = None
    original_token_count: int | None = None
    assignment_count: int | None = None
    input_edge_count: int | None = None
    output_token_count: int | None = None
    logical_stage_id: str | None = None
    leg_id: str | None = None
    source_endpoint: AFEndpoint | None = None
    target_endpoint: AFEndpoint | None = None
    participant_endpoints: tuple[AFEndpoint, ...] = ()
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    source_layout: AFTensorLayout | None = None
    target_layout: AFTensorLayout | None = None
    effective_dtypes: Mapping[str, str] = field(default_factory=dict)
    dtype_role: str | None = None
    dtype: str | None = None
    bytes_per_element: int | None = None
    network: str | None = None
    resource_group: str | None = None
    resource_lane: str | None = None
    payload_breakdown: tuple[Mapping[str, object], ...] = ()
    routing_edges: tuple[AFRoutingEdge, ...] = ()
    source_dtype: str | None = None
    target_dtype: str | None = None


def _endpoint_identity(endpoint: AFEndpoint | None) -> tuple[object, ...] | None:
    if endpoint is None:
        return None
    return (endpoint.domain, endpoint.domain_rank, endpoint.physical_worker)


def _is_f_domain(domain: str | None) -> bool:
    if domain is None:
        return False
    value = domain.strip().lower()
    return value.startswith("f") or "ffn" in value or value.endswith("-f")


def communication_window(start, end, *, fixed_intervals=(), transport_intervals=()):
    """Partition one observer's window. Blockers are not additive costs.

    Fixed intervals must be the observer's own service, not the remote CPU's
    overlapping work. Transport may include an inseparable device startup cost.
    """
    groups = [tuple(fixed_intervals), tuple(transport_intervals)]
    if not all(math.isfinite(t) for t in (start, end)) or end < start:
        raise ValueError("invalid communication observation window")
    for group in groups:
        for left, right in group:
            if (
                not all(math.isfinite(t) for t in (left, right))
                or left < start
                or right > end
                or right < left
            ):
                raise ValueError("service interval outside communication observation")
    fixed, transport = (_union(g) for g in groups)
    combined = _union(groups[0] + groups[1])
    if fixed + transport - combined > 1e-12:
        raise ValueError("fixed and transport service overlap in one observation")
    return dict(
        start_at=start,
        end_at=end,
        total_duration=end - start,
        waiting_duration=max(0.0, end - start - combined),
        fixed_service_duration=fixed,
        transport_service_duration=transport,
    )


def summarize_message_waits(events):
    """Join this iteration's point-to-point events once, independently of history.

    These are modeled invocation windows, not proof of native recv preposting.
    Sender/receiver observations are alternatives; never sum them as wall time.
    """
    events = tuple(events)
    by_id = {e.node_id: e for e in events}
    rows = []
    for wire in events:
        if wire.event_type not in {
            "transfer",
            "control_transfer",
        } or not wire.node_id.endswith(":transport"):
            continue
        message_id = wire.node_id.removesuffix(":transport")
        completion = by_id.get(message_id)
        finish = completion.end_at if completion is not None else wire.end_at
        for side, suffix in (
            ("sender", ":host_submit"),
            ("receiver", ":receive_submit"),
        ):
            submit = by_id.get(message_id + suffix)
            if submit is None:
                continue  # Missing receiver invocation is unknown, not zero wait.
            begin = submit.ready_at
            fixed = [(submit.start_at, submit.end_at)]
            if completion is not None and set(completion.physical_endpoints) & set(
                submit.physical_endpoints
            ):
                fixed.append((completion.start_at, completion.end_at))
            row = communication_window(
                begin,
                finish,
                fixed_intervals=fixed,
                transport_intervals=[(wire.start_at, wire.end_at)],
            )
            rows.append(
                dict(
                    message_id=message_id,
                    observation_side=side,
                    observation_endpoints=list(submit.physical_endpoints),
                    observation_scope="modeled local submit ready to modeled data completion",
                    message_class=wire.details.get("message_class", "data"),
                    transport_scope="provider service including inseparable device startup",
                    full_fixed_transfer_split_available=False,
                    **row,
                )
            )
    return rows


def _union(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(stop, end) if end is not None else stop
    return total


def _category(event):
    if event.event_type == "operation_service":
        return "operation_service"
    if event.event_type == "control_transfer":
        return "control_transport"
    if event.event_type in {"transfer", "collective"}:
        return "data_transport"
    if event.event_type == "control_device":
        return "control_device"
    if event.event_type == "compute":
        return "compute"
    return "host_service"


def summarize_timeline(events, start, end, capacities=None):
    """A timestamp-supported critical chain, with gaps explicitly retained.

    Only this iteration is indexed. Resource delays caused by other iterations
    are retained as queue intervals rather than scanning historical traces.
    A critical chain may be non-unique; ties choose one valid blocker.
    """
    events = tuple(events)
    capacities = capacities or {}
    by_id = {event.node_id: event for event in events}
    indices = {event.node_id: i for i, event in enumerate(events)}
    intervals = defaultdict(list)
    sums = defaultdict(float)
    resources = defaultdict(list)
    counted = {id(event) for event in unique_executions(events)}
    for index, event in enumerate(events):
        left, right = max(start, event.start_at), min(end, event.end_at)
        if right > left:
            intervals[_category(event)].append((left, right))
            if id(event) in counted:
                sums[_category(event)] += right - left
        for resource in event.resource_ids or (event.resource_id,):
            if capacities.get(resource) == 1:
                resources[resource].append(
                    (occupied_until(event), index, event.node_id)
                )
    for values in resources.values():
        values.sort()
    segments = []

    def segment(left, right, category, node_id):
        left, right = max(start, left), min(end, right)
        if right > left:
            segments.append(
                dict(
                    start_at=left,
                    end_at=right,
                    duration=right - left,
                    category=category,
                    node_id=node_id,
                )
            )

    if not events:
        segment(start, end, "unexplained", None)
    else:
        current = max(events, key=lambda e: (e.end_at, indices[e.node_id]))
        segment(current.end_at, end, "unexplained", None)
        seen = set()
        while current is not None:
            if current.node_id in seen:
                raise ValueError("cycle in trace critical chain")
            seen.add(current.node_id)
            segment(
                current.start_at, current.end_at, _category(current), current.node_id
            )
            blockers = [by_id[p] for p in current.predecessor_ids if p in by_id]
            for resource in current.resource_ids or (current.resource_id,):
                values = resources.get(resource, ())
                position = (
                    bisect_right(
                        values,
                        (current.start_at, indices[current.node_id] - 1, chr(0x10FFFF)),
                    )
                    - 1
                )
                if position >= 0:
                    blocking = by_id[values[position][2]]
                    released = occupied_until(blocking)
                    blockers.append(
                        replace(
                            blocking,
                            end_at=released,
                            service_duration=released - blocking.start_at,
                        )
                    )
            blockers = [
                e
                for e in blockers
                if e.node_id not in seen and e.end_at <= current.start_at + 1e-12
            ]
            previous = max(
                blockers, key=lambda e: (e.end_at, indices[e.node_id]), default=None
            )
            previous_end = (
                max(start, previous.end_at) if previous is not None else start
            )
            dependency_end = min(current.ready_at, current.start_at)
            if dependency_end > previous_end:
                wait_start = current.details.get(
                    "readiness_wait_started_at", dependency_end
                )
                segment(
                    previous_end,
                    min(wait_start, dependency_end),
                    "unexplained",
                    current.node_id,
                )
                segment(
                    max(previous_end, wait_start),
                    dependency_end,
                    "peer_wait",
                    current.node_id,
                )
            segment(
                max(previous_end, current.ready_at),
                current.start_at,
                "resource_queue",
                current.node_id,
            )
            current = previous
    segments.sort(key=lambda s: (s["start_at"], s["end_at"]))
    totals = defaultdict(float)
    for part in segments:
        totals[part["category"]] += part["duration"]
    reconstructed = sum(part["duration"] for part in segments)
    if abs(reconstructed - (end - start)) > 1e-10:
        raise ValueError("critical chain does not reconstruct iteration makespan")
    return dict(
        makespan=end - start,
        reconstructed_makespan=reconstructed,
        service_sum_by_category=dict(sums),
        busy_union_by_category={
            name: _union(values) for name, values in intervals.items()
        },
        critical_chain=segments,
        critical_duration_by_category=dict(totals),
        unexplained_duration=totals.get("unexplained", 0.0),
        interpretation="one timestamp-supported chain; busy unions across categories may overlap; external blockers remain queue/gaps",
    )


from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.providers import queries as _queries

register_records(
    AFTraceEvent,
    FrozenDict,
    *[
        getattr(_queries, name)
        for name in (
            "AFEndpoint",
            "AFTensorSlice",
            "AFRoutingEdge",
            "AFComputeQuery",
            "AFTransferQuery",
            "AFServiceEstimate",
        )
    ],
)


def summarize_iteration(recorder, iteration):
    """Return aggregate metadata without materializing full event dictionaries."""
    events = recorder.events_for(iteration.iteration_id)
    utilization = {}
    makespan = iteration.makespan or 0.0
    busy_by_resource = {}
    for event in unique_executions(events):
        for resource_id in set(event.resource_ids or (event.resource_id,)):
            if resource_id not in busy_by_resource:
                busy_by_resource[resource_id] = _ResourceBusyTime()
            busy_by_resource[resource_id].add(occupied_until(event) - event.start_at)
    for resource_id, busy in sorted(busy_by_resource.items()):
        capacity = recorder.resource_capacities.get(resource_id, 1)
        utilization[resource_id] = (
            busy.value() / (makespan * capacity) if makespan > 0 else 0.0
        )
    return {
        "iteration_id": iteration.iteration_id,
        "phase": iteration.phase,
        "start_at": iteration.start_at,
        "end_at": iteration.end_at,
        "makespan": iteration.makespan,
        "block_completed_at": dict(iteration.block_completed_at),
        "barrier_wait": dict(iteration.barrier_wait),
        "resource_utilization": utilization,
    }
