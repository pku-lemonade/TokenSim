from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol

from TokenSim.af.models import AFPhase

# How a provider answered: measured row, interpolated between rows, extrapolated
# beyond the measured range, or the device family's analytical formula.
AF_MATCH_TYPES = ("exact", "interpolated", "extrapolated", "analytical")


@dataclass(frozen=True)
class AFComputeQuery:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: AFPhase
    stage: Literal["a", "router", "expert"]
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
