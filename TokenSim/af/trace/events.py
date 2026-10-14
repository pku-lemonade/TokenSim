"""Trace event."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
import sys
from typing import Any, Literal, Mapping

from TokenSim.af.planning.batches import AFPhase
from TokenSim.af.providers.queries import (
    AFComputeStage,
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
class AFTraceEvent:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: AFPhase
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
    dataset_version: str = "test-stub"
    match_type: Literal["exact", "interpolated"] = "exact"
    stage: AFComputeStage | Literal["cast"] | None = None
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _endpoint_identity(endpoint: AFEndpoint | None) -> tuple[object, ...] | None:
    if endpoint is None:
        return None
    return (endpoint.domain, endpoint.domain_rank, endpoint.physical_worker)


def _is_f_domain(domain: str | None) -> bool:
    if domain is None:
        return False
    value = domain.strip().lower()
    return value.startswith("f") or "ffn" in value or value.endswith("-f")
