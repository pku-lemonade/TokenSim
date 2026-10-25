"""Versioned records with simulation-second timestamps and stable identities."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

RECORD_TYPES = {}


def register_records(*types):
    for kind in types:
        RECORD_TYPES[kind.__module__, kind.__name__] = kind


@dataclass(frozen=True)
class TraceEvent:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: str
    event_type: Literal[
        "compute",
        "transfer",
        "control_transfer",
        "control_device",
        "operation_service",
        "collective",
        "host_submit",
        "host_prepare",
        "synchronize",
    ]
    resource_id: str
    ready_at: float
    start_at: float
    end_at: float
    queue_duration: float
    service_duration: float
    request_count: int
    token_count: int
    source_id: str
    dataset_version: str = "unspecified"
    match_type: str = "exact"
    stage: str | None = None
    direction: str | None = None
    worker_id: int | None = None
    payload_bytes: int | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    operation: str | None = None
    resource_ids: tuple[str, ...] = ()
    node_id: str | None = None
    predecessor_ids: tuple[str, ...] = ()
    dp_rank: int | None = None
    tp_rank: int | None = None
    ep_rank: int | None = None
    pp_rank: int | None = None
    run_phase: str = "profiling"
    incomplete: bool = False
    execution_group_id: str | None = None
    input_ready_at: float | None = None
    resource_queue_intervals: tuple[tuple[float, float], ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


register_records(TraceEvent)
