"""Trace."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from TokenSim.af.planning.batches import AFIteration
from TokenSim.af.trace.events import AFTraceEvent, _ResourceBusyTime
from TokenSim.af.trace.validation import TraceValidationMixin
from TokenSim.errors import SimulationStateError


class AFTraceRecorder(TraceValidationMixin):
    """Own the full event history and index it by iteration.

    Append through record(), and reset through clear(). In-place replacement
    in events is supported only when position and iteration identity are kept.
    """

    def __init__(
        self,
        enabled: bool = True,
        resource_capacities: Mapping[str, int] | None = None,
    ) -> None:
        self.enabled = enabled
        self.events: list[AFTraceEvent] = []
        self.iteration_event_indices: dict[int, list[int]] = {}
        self.resource_capacities = dict(resource_capacities or {})

    def record(self, event: AFTraceEvent) -> None:
        if self.enabled:
            index = len(self.events)
            self.events.append(event)
            self.iteration_event_indices.setdefault(event.iteration_id, []).append(
                index
            )

    def clear(self) -> None:
        """Explicitly discard a drained batch, including its position indexes."""
        self.events.clear()
        self.iteration_event_indices.clear()

    def events_for(self, iteration_id: int) -> tuple[AFTraceEvent, ...]:
        return tuple(
            self.events[index]
            for index in self.iteration_event_indices.get(iteration_id, ())
        )

    def summarize(self, iteration: AFIteration) -> dict[str, Any]:
        """Return aggregate metadata without materializing full event dictionaries."""
        events = self.events_for(iteration.iteration_id)
        utilization: dict[str, float] = {}
        makespan = iteration.makespan or 0.0
        busy_by_resource: dict[str, _ResourceBusyTime] = {}
        for event in events:
            for resource_id in set(event.resource_ids or (event.resource_id,)):
                if resource_id not in busy_by_resource:
                    busy_by_resource[resource_id] = _ResourceBusyTime()
                busy_by_resource[resource_id].add(event.service_duration)
        for resource_id, busy in sorted(busy_by_resource.items()):
            capacity = self.resource_capacities.get(resource_id, 1)
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

    def summarize_timeline(self, iteration: AFIteration) -> dict[str, Any]:
        """Optional detailed accounting, restricted to this iteration's index."""
        from TokenSim.af.trace.metrics import summarize_timeline

        if iteration.start_at is None or iteration.end_at is None:
            raise SimulationStateError(
                "timeline summary requires a completed iteration"
            )
        return summarize_timeline(
            self.events_for(iteration.iteration_id),
            iteration.start_at,
            iteration.end_at,
            self.resource_capacities,
        )

    def export_summary(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        """Materialize a detached public result at the export boundary only."""
        result = deepcopy(dict(summary))
        result["events"] = (
            [event.as_dict() for event in self.events_for(summary["iteration_id"])]
            if self.enabled
            else []
        )
        return result
