from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from TokenSim.af.models import AFIteration, AFPhase
from TokenSim.errors import SimulationStateError


@dataclass(frozen=True)
class AFTraceEvent:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: AFPhase
    event_type: Literal["compute", "transfer", "collective"]
    resource_id: str
    ready_at: float
    start_at: float
    end_at: float
    queue_duration: float
    service_duration: float
    request_count: int
    token_count: int
    source_id: str
    dataset_version: str = "test-stub"
    match_type: Literal["exact", "interpolated"] = "exact"
    stage: Literal["a", "router", "expert"] | None = None
    direction: str | None = None
    worker_id: int | None = None
    payload_bytes: int | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    operation: str | None = None
    source_domain: str | None = None
    target_domain: str | None = None
    logical_participants: tuple[str, ...] = ()
    participant_ranks: tuple[int, ...] = ()
    physical_endpoints: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()
    node_id: str | None = None
    predecessor_ids: tuple[str, ...] = ()
    dp_rank: int | None = None
    tp_rank: int | None = None
    ep_rank: int | None = None
    tp_group: tuple[int, ...] = ()
    ep_group: tuple[int, ...] = ()
    local_work: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AFTraceRecorder:
    def __init__(
        self,
        enabled: bool = True,
        resource_capacities: Mapping[str, int] | None = None,
    ) -> None:
        self.enabled = enabled
        self.events: list[AFTraceEvent] = []
        self.resource_capacities = dict(resource_capacities or {})

    def record(self, event: AFTraceEvent) -> None:
        if self.enabled:
            self.events.append(event)

    def events_for(self, iteration_id: int) -> tuple[AFTraceEvent, ...]:
        return tuple(
            event for event in self.events if event.iteration_id == iteration_id
        )

    def validate_invariants(self, iteration_id: int) -> None:
        events = self.events_for(iteration_id)
        by_node: dict[str, AFTraceEvent] = {}
        tolerance = 1e-12
        for event in events:
            if not event.node_id:
                raise SimulationStateError(
                    f"AF trace invariant failed: iteration={iteration_id} event has no node_id"
                )
            if event.node_id in by_node:
                raise SimulationStateError(
                    f"AF trace invariant failed: duplicate node_id={event.node_id!r}"
                )
            by_node[event.node_id] = event
            if not (
                event.ready_at <= event.start_at + tolerance
                and event.start_at <= event.end_at + tolerance
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: invalid timestamps for node={event.node_id}"
                )
            if (
                abs(event.queue_duration - (event.start_at - event.ready_at))
                > tolerance
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: queue duration mismatch for node={event.node_id}"
                )
            if (
                abs(event.service_duration - (event.end_at - event.start_at))
                > tolerance
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: service duration mismatch for node={event.node_id}"
                )
            returned_payload = event.details.get("payload_bytes")
            if (
                returned_payload is not None
                and event.payload_bytes is not None
                and returned_payload != event.payload_bytes
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: payload mismatch for node={event.node_id}: "
                    f"planned={event.payload_bytes}, provider={returned_payload}"
                )

        for event in events:
            missing = set(event.predecessor_ids) - set(by_node)
            if missing:
                raise SimulationStateError(
                    f"AF trace invariant failed: node={event.node_id} has missing "
                    f"predecessors={sorted(missing)}"
                )
            if event.predecessor_ids:
                latest = max(
                    by_node[node_id].end_at for node_id in event.predecessor_ids
                )
                if event.ready_at + tolerance < latest:
                    raise SimulationStateError(
                        f"AF trace invariant failed: dependency time for node={event.node_id}; "
                        f"ready_at={event.ready_at}, predecessor_end={latest}"
                    )
                if (
                    len(event.predecessor_ids) > 1
                    and abs(event.ready_at - latest) > tolerance
                ):
                    raise SimulationStateError(
                        f"AF trace invariant failed: join ready time for node={event.node_id}; "
                        f"ready_at={event.ready_at}, latest_predecessor={latest}"
                    )

        layers = {(event.block_id, event.layer_id) for event in events}
        for block_id, layer_id in layers:
            layer_operations = [
                event.operation
                for event in events
                if event.block_id == block_id and event.layer_id == layer_id
            ]
            for operation in ("af_a_to_f", "af_f_to_a"):
                count = layer_operations.count(operation)
                if count != 1:
                    raise SimulationStateError(
                        f"AF trace invariant failed: block={block_id}, layer={layer_id} "
                        f"requires exactly one {operation}; got {count}"
                    )

        intervals: dict[str, list[tuple[float, int, str]]] = {}
        for event in events:
            if event.service_duration <= 0:
                continue
            for resource_id in event.resource_ids or (event.resource_id,):
                intervals.setdefault(resource_id, []).extend(
                    (
                        (event.start_at, 1, event.node_id),
                        (event.end_at, -1, event.node_id),
                    )
                )
        for resource_id, points in intervals.items():
            active = 0
            capacity = self.resource_capacities.get(resource_id, 1)
            for time, delta, node_id in sorted(
                points, key=lambda item: (item[0], item[1])
            ):
                active += delta
                if active > capacity:
                    raise SimulationStateError(
                        f"AF trace invariant failed: resource={resource_id!r} capacity="
                        f"{capacity} exceeded at time={time} by node={node_id}"
                    )

    def summarize(self, iteration: AFIteration) -> dict[str, Any]:
        events = self.events_for(iteration.iteration_id)
        utilization: dict[str, float] = {}
        makespan = iteration.makespan or 0.0
        event_resources = {
            resource_id
            for event in events
            for resource_id in (event.resource_ids or (event.resource_id,))
        }
        for resource_id in sorted(event_resources):
            busy = sum(
                event.service_duration
                for event in events
                if resource_id in (event.resource_ids or (event.resource_id,))
            )
            capacity = self.resource_capacities.get(resource_id, 1)
            utilization[resource_id] = (
                busy / (makespan * capacity) if makespan > 0 else 0.0
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
            "events": [event.as_dict() for event in events],
        }
