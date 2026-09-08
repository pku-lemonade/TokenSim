from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol

from TokenSim.af.models import AFPhase


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
    match_type: Literal["exact", "interpolated"] = "exact"
    details: Mapping[str, object] = field(default_factory=dict)


class AFComputeLatencyProvider(Protocol):
    def estimate(self, query: AFComputeQuery) -> AFServiceEstimate: ...


class AFTransferLatencyProvider(Protocol):
    def estimate(self, query: AFTransferQuery) -> AFServiceEstimate: ...


def resolve_af_latency_providers(
    af_config,
    cluster_path: str | Path,
    compute_provider: AFComputeLatencyProvider | None = None,
    transfer_provider: AFTransferLatencyProvider | None = None,
) -> tuple[AFComputeLatencyProvider | None, AFTransferLatencyProvider | None]:
    if af_config.latency_provider == "injected":
        return compute_provider, transfer_provider
    if compute_provider is not None or transfer_provider is not None:
        raise ValueError(
            "AF operator_data configuration cannot be combined with injected "
            "latency providers"
        )
    from TokenSim.operator_data.providers import build_operator_data_providers

    config_path = Path(cluster_path).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else config_path.parent / path

    return build_operator_data_providers(
        resolve(af_config.operator_data_path),
        resolve(af_config.operator_manifest_path),
        af_config.operator_data_backend,
        af_config.operator_link_group,
        af_config.operator_group_size,
    )
