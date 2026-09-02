from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from TokenSim.af.models import AFIteration, AFPhase


@dataclass(frozen=True)
class AFTraceEvent:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: AFPhase
    event_type: Literal["compute", "transfer"]
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
    stage: Literal["a", "f"] | None = None
    direction: Literal["a_to_f", "f_to_a"] | None = None
    worker_id: int | None = None
    payload_bytes: int | None = None
    details: Mapping[str, object] = field(default_factory=dict)

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

    def summarize(self, iteration: AFIteration) -> dict[str, Any]:
        events = self.events_for(iteration.iteration_id)
        utilization: dict[str, float] = {}
        makespan = iteration.makespan or 0.0
        for resource_id in sorted({event.resource_id for event in events}):
            busy = sum(
                event.service_duration
                for event in events
                if event.resource_id == resource_id
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
