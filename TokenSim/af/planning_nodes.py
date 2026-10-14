"""Planning nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from TokenSim.af.latency import (
    AFComputeStage,
    AFEndpoint,
    AFLayerType,
    AFRoutingEdge,
    AFTensorLayout,
    AFTensorSlice,
)
from TokenSim.errors import ConfigurationError


NodeKind = Literal["compute", "communication"]


ExecutionStage = AFComputeStage | Literal["cast"]


_EXECUTION_STAGES = frozenset(
    {
        "a",
        "router",
        "expert",
        "attention",
        "dense_ffn",
        "routed_expert",
        "shared_expert",
        "weighted_combine",
        "residual",
        "cast",
        "encode",
        "decode",
        "layout",
    }
)


@dataclass(frozen=True)
class _TransferCopy:
    source_binding: Any
    target_binding: Any
    tensor_slice: AFTensorSlice
    predecessor_id: str
    consumer_id: str
    source_layout: AFTensorLayout
    target_layout: AFTensorLayout
    routing_edges: tuple[AFRoutingEdge, ...] = ()


@dataclass(frozen=True)
class RankLocalWork:
    token_count: int
    tensor_shape: tuple[tuple[str, int], ...] = ()
    expert_ids: tuple[int, ...] = ()
    expert_workload: tuple[tuple[int, int], ...] = ()
    original_token_count: int | None = None
    assignment_count: int | None = None
    input_edge_count: int | None = None
    output_token_count: int | None = None
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    owner_endpoint: AFEndpoint | None = None
    source_layout: AFTensorLayout | None = None
    target_layout: AFTensorLayout | None = None
    routing_edges: tuple[AFRoutingEdge, ...] = ()
    source_dtype: str | None = None
    target_dtype: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "token_count",
            "original_token_count",
            "assignment_count",
            "input_edge_count",
            "output_token_count",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError(
                    f"rank-local work {field_name} must be a non-negative integer"
                )
        for field_name in ("source_layout", "target_layout"):
            value = getattr(self, field_name)
            if value is not None and value not in {
                "sharded",
                "replicated",
                "partial",
            }:
                raise ConfigurationError(
                    f"rank-local work {field_name} has unsupported layout {value!r}"
                )
        object.__setattr__(self, "tensor_shape", tuple(self.tensor_shape))
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))
        object.__setattr__(self, "expert_workload", tuple(self.expert_workload))
        object.__setattr__(self, "tensor_slices", tuple(self.tensor_slices))
        object.__setattr__(self, "routing_edges", tuple(self.routing_edges))

    def as_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "tensor_shape": dict(self.tensor_shape),
            "expert_ids": self.expert_ids,
            "expert_workload": dict(self.expert_workload),
            "original_token_count": self.original_token_count,
            "assignment_count": self.assignment_count,
            "input_edge_count": self.input_edge_count,
            "output_token_count": self.output_token_count,
            "tensor_slices": tuple(asdict(item) for item in self.tensor_slices),
            "owner_endpoint": (
                asdict(self.owner_endpoint) if self.owner_endpoint is not None else None
            ),
            "source_layout": self.source_layout,
            "target_layout": self.target_layout,
            "routing_edges": tuple(asdict(item) for item in self.routing_edges),
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
        }


@dataclass(frozen=True)
class ExecutionNode:
    node_id: str
    operation: str
    kind: NodeKind
    predecessor_ids: tuple[str, ...]
    stage: ExecutionStage | None
    direction: str | None
    source_domain: str
    target_domain: str
    logical_participants: tuple[str, ...]
    domain_ranks: tuple[int, ...]
    physical_workers: tuple[str, ...]
    dp_rank: int
    tp_rank: int | None = None
    ep_rank: int | None = None
    tp_group: tuple[int, ...] = ()
    ep_group: tuple[int, ...] = ()
    local_work: RankLocalWork | None = None
    network: str | None = None
    resource_group: str | None = None
    resource_lane: str | None = None
    payload_bytes: int | None = None
    logical_input_bytes_per_rank: tuple[tuple[int, int], ...] = ()
    logical_output_bytes_per_rank: tuple[tuple[int, int], ...] = ()
    layer_type: AFLayerType | None = None
    logical_stage_id: str | None = None
    leg_id: str | None = None
    source_endpoint: AFEndpoint | None = None
    target_endpoint: AFEndpoint | None = None
    participant_endpoints: tuple[AFEndpoint, ...] = ()
    input_producer_ids: tuple[tuple[str, str], ...] = ()
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    source_layout: AFTensorLayout | None = None
    target_layout: AFTensorLayout | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    routing_edges: tuple[AFRoutingEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ConfigurationError("execution node id must be non-empty")
        if not self.operation:
            raise ConfigurationError(
                f"execution node {self.node_id!r} operation must be non-empty"
            )
        if self.kind not in {"compute", "communication"}:
            raise ConfigurationError(
                f"execution node {self.node_id!r} has unsupported kind {self.kind!r}"
            )
        if self.kind == "compute" and len(self.physical_workers) != 1:
            raise ConfigurationError(
                f"compute node {self.node_id!r} must bind exactly one physical worker"
            )
        if self.kind == "compute" and self.stage is None:
            raise ConfigurationError(
                f"compute node {self.node_id!r} must declare a compute stage"
            )
        if self.stage is not None and self.stage not in _EXECUTION_STAGES:
            raise ConfigurationError(
                f"execution node {self.node_id!r} has unsupported stage {self.stage!r}"
            )
        if self.layer_type is not None and self.layer_type not in {"dense", "moe"}:
            raise ConfigurationError(
                f"execution node {self.node_id!r} has unsupported layer type "
                f"{self.layer_type!r}"
            )
        if self.logical_stage_id is not None and not self.logical_stage_id:
            raise ConfigurationError(
                f"execution node {self.node_id!r} logical_stage_id must be non-empty"
            )
        if self.payload_bytes is not None and (
            type(self.payload_bytes) is not int or self.payload_bytes < 0
        ):
            raise ConfigurationError(
                f"execution node {self.node_id!r} payload_bytes must be non-negative"
            )
        for field_name in ("source_layout", "target_layout"):
            value = getattr(self, field_name)
            if value is not None and value not in {
                "sharded",
                "replicated",
                "partial",
            }:
                raise ConfigurationError(
                    f"execution node {self.node_id!r} {field_name} has unsupported "
                    f"layout {value!r}"
                )
        for field_name in (
            "predecessor_ids",
            "logical_participants",
            "domain_ranks",
            "physical_workers",
            "tp_group",
            "ep_group",
            "participant_endpoints",
            "input_producer_ids",
            "tensor_slices",
            "logical_input_bytes_per_rank",
            "logical_output_bytes_per_rank",
            "routing_edges",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if self.input_producer_ids:
            endpoints = tuple(endpoint for endpoint, _ in self.input_producer_ids)
            if (
                self.kind != "communication"
                or endpoints != self.physical_workers
                or len(set(endpoints)) != len(endpoints)
                or any(
                    producer not in self.predecessor_ids
                    for _, producer in self.input_producer_ids
                )
            ):
                raise ConfigurationError(
                    "collective input bindings must uniquely cover endpoints in participant order and reference predecessors"
                )
        elif self.metadata.get("collective_input_binding_required"):
            raise ConfigurationError("collective is missing required input bindings")
