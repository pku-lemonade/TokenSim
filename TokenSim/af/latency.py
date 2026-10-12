from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol

from TokenSim.af.models import AFPhase

# How a provider answered: measured row, interpolated between rows, extrapolated
# beyond the measured range, or the device family's analytical formula.
AF_MATCH_TYPES = ("exact", "interpolated", "extrapolated", "analytical")


AFLayerType = Literal["dense", "moe"]
AFComputeStage = Literal[
    "a",
    "router",
    "expert",
    "attention",
    "dense_ffn",
    "routed_expert",
    "shared_expert",
    "weighted_combine",
    "residual",
    "encode",
    "decode",
    "layout",
]
AFTensorLayout = Literal["sharded", "replicated", "partial"]
AFCommunicationKind = Literal["endpoint_leg", "collective"]


@dataclass(frozen=True)
class AFRoutingEdge:
    iteration_id: int
    phase: AFPhase
    decode_step: int
    layer_id: int
    request_id: int
    token_id: int
    top_k_index: int
    expert_id: int
    routing_weight: float
    a_dp_rank: int
    a_tp_rank: int

    def __post_init__(self) -> None:
        for name in (
            "iteration_id",
            "decode_step",
            "layer_id",
            "request_id",
            "token_id",
            "top_k_index",
            "expert_id",
            "a_dp_rank",
            "a_tp_rank",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"AF routing edge {name} must be a non-negative integer"
                )
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(
                f"AF routing edge phase must be 'prefill' or 'decode', got "
                f"{self.phase!r}"
            )
        if (
            isinstance(self.routing_weight, bool)
            or not isinstance(self.routing_weight, (int, float))
            or not math.isfinite(self.routing_weight)
            or self.routing_weight < 0
        ):
            raise ValueError(
                "AF routing edge routing_weight must be finite and non-negative"
            )
        object.__setattr__(self, "routing_weight", float(self.routing_weight))


@dataclass(frozen=True)
class AFEndpoint:
    domain: str
    domain_rank: int
    physical_worker: str
    device_type: str
    dp_rank: int = 0
    tp_rank: int = 0
    ep_rank: int = 0

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("AF endpoint domain must be non-empty")
        if not self.physical_worker:
            raise ValueError("AF endpoint physical_worker must be non-empty")
        if not self.device_type:
            raise ValueError("AF endpoint device_type must be non-empty")
        for name in ("domain_rank", "dp_rank", "tp_rank", "ep_rank"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"AF endpoint {name} must be a non-negative integer")


@dataclass(frozen=True)
class AFTensorSlice:
    tensor_item_id: str
    interval: tuple[int, int]
    layout: AFTensorLayout
    owner_domain: str | None = None
    owner_domain_rank: int | None = None
    owner_dp_rank: int | None = None
    owner_tp_rank: int | None = None
    owner_physical_worker: str | None = None
    partial_id: str | None = None
    canonical_producer: str | None = None
    routing_weight: float | None = None
    element_count: int | None = None
    payload_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.tensor_item_id:
            raise ValueError("AF tensor slice tensor_item_id must be non-empty")
        if (
            not isinstance(self.interval, tuple)
            or len(self.interval) != 2
            or any(type(value) is not int for value in self.interval)
            or self.interval[0] < 0
            or self.interval[1] <= self.interval[0]
        ):
            raise ValueError(
                "AF tensor slice interval must be a non-empty half-open integer range"
            )
        if self.layout not in {"sharded", "replicated", "partial"}:
            raise ValueError(f"unsupported AF tensor layout {self.layout!r}")
        for name in ("owner_domain_rank", "owner_dp_rank", "owner_tp_rank"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    f"AF tensor slice {name} must be a non-negative integer"
                )
        for name in ("element_count", "payload_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    f"AF tensor slice {name} must be a non-negative integer"
                )
        if self.routing_weight is not None and (
            isinstance(self.routing_weight, bool)
            or not isinstance(self.routing_weight, (int, float))
            or not math.isfinite(self.routing_weight)
            or self.routing_weight < 0
        ):
            raise ValueError(
                "AF tensor slice routing_weight must be finite and non-negative"
            )


@dataclass(frozen=True)
class AFComputeQuery:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: AFPhase
    stage: AFComputeStage
    token_count: int
    context_lengths: tuple[int, ...]
    model_id: str
    dtype: str
    device_type: str
    worker_id: int
    operation: str = ""
    source_domain: str | None = None
    dp_rank: int = 0
    tp_rank: int = 0
    ep_rank: int = 0
    logical_participant: str | None = None
    physical_worker: str | None = None
    local_shape: Mapping[str, int] = field(default_factory=dict)
    local_token_count: int | None = None
    expert_ids: tuple[int, ...] = ()
    expert_workload: tuple[tuple[int, int], ...] = ()
    tp_group: tuple[int, ...] = ()
    ep_group: tuple[int, ...] = ()
    layer_type: AFLayerType | None = None
    original_token_count: int | None = None
    assignment_count: int | None = None
    input_edge_count: int | None = None
    output_token_count: int | None = None
    weight_dtype: str | None = None
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    a_to_f_dtype: str | None = None
    f_to_a_dtype: str | None = None
    logical_stage_id: str | None = None
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    owner_endpoint: AFEndpoint | None = None
    routing_edges: tuple[AFRoutingEdge, ...] = ()
    source_dtype: str | None = None
    target_dtype: str | None = None


@dataclass(frozen=True)
class AFTransferQuery:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: AFPhase
    direction: str
    token_count: int
    hidden_size: int
    dtype: str
    source_device: str
    target_device: str
    network: str
    operation: str = "af_a_to_f"
    link_group: str | None = None
    group_size: int | None = None
    payload_bytes: int | None = None
    top_k: int = 1
    source_domain: str | None = None
    target_domain: str | None = None
    logical_participants: tuple[str, ...] = ()
    participant_ranks: tuple[int, ...] = ()
    physical_endpoints: tuple[str, ...] = ()
    resource_identities: tuple[str, ...] = ()
    topology: Mapping[str, object] = field(default_factory=dict)
    node_id: str | None = None
    communication_kind: AFCommunicationKind | None = None
    layer_type: AFLayerType | None = None
    original_token_count: int | None = None
    assignment_count: int | None = None
    weight_dtype: str | None = None
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    a_to_f_dtype: str | None = None
    f_to_a_dtype: str | None = None
    logical_stage_id: str | None = None
    leg_id: str | None = None
    source_endpoint: AFEndpoint | None = None
    target_endpoint: AFEndpoint | None = None
    participant_endpoints: tuple[AFEndpoint, ...] = ()
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    source_layout: AFTensorLayout | None = None
    target_layout: AFTensorLayout | None = None
    fabric: str | None = None
    resource_group: str | None = None
    resource_lane: str | None = None
    logical_input_bytes_per_rank: tuple[tuple[int, int], ...] = ()
    logical_output_bytes_per_rank: tuple[tuple[int, int], ...] = ()
    routing_edges: tuple[AFRoutingEdge, ...] = ()
    local_fraction: float | None = None
    remote_fraction: float | None = None
    load_imbalance: float = 1.0
    destination_redistribution: bool = False


@dataclass(frozen=True)
class AFServiceEstimate:
    service_duration: float
    source_id: str
    payload_bytes: int | None = None
    dataset_version: str = "test-stub"
    match_type: Literal["exact", "interpolated", "extrapolated", "analytical"] = "exact"
    details: Mapping[str, object] = field(default_factory=dict)


class AFComputeLatencyProvider(Protocol):
    def estimate(self, query: AFComputeQuery) -> AFServiceEstimate: ...


class AFTransferLatencyProvider(Protocol):
    def estimate(self, query: AFTransferQuery) -> AFServiceEstimate: ...


def resolve_af_latency_providers(
    af_config,
    cluster_path: str | Path | None = None,
    compute_provider: AFComputeLatencyProvider | None = None,
    transfer_provider: AFTransferLatencyProvider | None = None,
) -> tuple[AFComputeLatencyProvider | None, AFTransferLatencyProvider | None]:
    """Validate the provider selection made by ``af.latency_provider``.

    ``injected`` returns the caller's providers unchanged. ``operator_table``
    rejects injected providers and returns ``(None, None)``: the providers need
    the hardware catalog, the resolved model and the worker placement, so
    :class:`TokenSim.llm.llm_engine.LLMEngine` builds them via
    :func:`TokenSim.af.operator_provider.build_operator_table_af_providers`.
    """
    del cluster_path  # kept for call-site compatibility; paths are no longer needed
    if af_config.latency_provider == "injected":
        return compute_provider, transfer_provider
    if af_config.latency_provider != "operator_table":
        raise ValueError(
            f"unsupported af.latency_provider={af_config.latency_provider!r}"
        )
    if compute_provider is not None or transfer_provider is not None:
        raise ValueError(
            "AF operator_table configuration cannot be combined with injected "
            "latency providers"
        )
    return None, None
