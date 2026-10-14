"""Runtime records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from TokenSim.config.af.domains import PhysicalWorkerRef
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.mla import MLADimensions


RankKey = tuple[str, int]


ReplicaKey = tuple[str, int]


LayerType = Literal["dense", "moe"]


TensorLayout = Literal["sharded", "replicated", "partial"]


@dataclass(frozen=True, order=True)
class AFTensorInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ConfigurationError("AF tensor interval bounds must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ConfigurationError(
                f"AF tensor interval must be a non-empty half-open range; "
                f"got [{self.start}, {self.end})"
            )

    @property
    def size(self) -> int:
        return self.end - self.start

    def intersection(self, other: "AFTensorInterval") -> "AFTensorInterval | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return AFTensorInterval(start, end) if start < end else None

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "size": self.size}


@dataclass(frozen=True)
class AFTensorOwnership:
    domain: str
    domain_rank: int
    dp_rank: int
    tp_rank: int
    physical_worker: str
    interval: AFTensorInterval
    layout: TensorLayout = "sharded"
    canonical_producer: bool = False
    retains_residual: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "domain_rank": self.domain_rank,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "physical_worker": self.physical_worker,
            "interval": self.interval.as_dict(),
            "layout": self.layout,
            "canonical_producer": self.canonical_producer,
            "retains_residual": self.retains_residual,
        }


@dataclass(frozen=True)
class AFRequestOwner:
    phase: str
    domain: str
    instance: str
    dp_rank: int
    leader_rank: int
    physical_worker: str

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "domain": self.domain,
            "instance": self.instance,
            "dp_rank": self.dp_rank,
            "leader_rank": self.leader_rank,
            "physical_worker": self.physical_worker,
        }


@dataclass(frozen=True)
class AFPartialDestination:
    phase: str
    source_domain: str
    source_rank: int
    source_dp_rank: int
    target_domain: str
    target_rank: int
    target_dp_rank: int
    interval: AFTensorInterval
    source_layout: TensorLayout = "sharded"
    target_layout: TensorLayout = "sharded"

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "source_domain": self.source_domain,
            "source_rank": self.source_rank,
            "source_dp_rank": self.source_dp_rank,
            "target_domain": self.target_domain,
            "target_rank": self.target_rank,
            "target_dp_rank": self.target_dp_rank,
            "interval": self.interval.as_dict(),
            "source_layout": self.source_layout,
            "target_layout": self.target_layout,
        }


@dataclass(frozen=True)
class AFLayerRuntime:
    layer_id: int
    layer_type: LayerType
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dense_intermediate_size: int
    routed_intermediate_size: int
    shared_intermediate_size: int
    num_experts: int
    top_k: int
    num_shared_experts: int
    mla: MLADimensions | None = None
    tie_word_embeddings: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "mla": vars(self.mla) if self.mla else None,
            "layer_type": self.layer_type,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dense_intermediate_size": self.dense_intermediate_size,
            "routed_intermediate_size": self.routed_intermediate_size,
            "shared_intermediate_size": self.shared_intermediate_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_shared_experts": self.num_shared_experts,
        }


@dataclass(frozen=True)
class AFResolvedCollectivePlan:
    domain: str
    dp_rank: int
    after_stage: str
    operation: str
    participant_ranks: tuple[int, ...]
    physical_workers: tuple[str, ...]
    network: str
    resource_group: str
    resource_lane: str
    payload_bytes: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "dp_rank": self.dp_rank,
            "after_stage": self.after_stage,
            "operation": self.operation,
            "participant_ranks": list(self.participant_ranks),
            "physical_workers": list(self.physical_workers),
            "network": self.network,
            "resource_group": self.resource_group,
            "resource_lane": self.resource_lane,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class AFModelComponent:
    component_id: str
    category: str
    bytes: int
    formula: str
    layer_id: int | None = None
    formula_source: str = "exact_model_shape"
    shape: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "category": self.category,
            "bytes": self.bytes,
            "formula": self.formula,
            "layer_id": self.layer_id,
            "formula_source": self.formula_source,
            "shape": dict(self.shape),
        }


@dataclass(frozen=True)
class AFKVHeadPartition:
    tp_rank: int
    global_head_ids: tuple[int, ...]
    replica_factor: int
    bytes_per_token: int
    bytes_per_block: int

    @property
    def local_head_count(self) -> int:
        return len(self.global_head_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "tp_rank": self.tp_rank,
            "global_head_ids": list(self.global_head_ids),
            "replica_factor": self.replica_factor,
            "local_head_count": self.local_head_count,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_block": self.bytes_per_block,
        }


@dataclass(frozen=True)
class AFRankBinding:
    domain: str
    capability: str
    instance: str
    domain_rank: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    ep_rank: int
    physical_worker: PhysicalWorkerRef
    hardware: str
    tp_group: tuple[int, ...]
    ep_group: tuple[int, ...]
    is_scheduler_leader: bool
    expert_ids: tuple[int, ...]
    component_ids: tuple[str, ...]
    kv_partition: AFKVHeadPartition | None = None

    @property
    def rank_key(self) -> RankKey:
        return (self.domain, self.domain_rank)

    @property
    def physical_identity(self) -> str:
        return self.physical_worker.identity

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "capability": self.capability,
            "instance": self.instance,
            "domain_rank": self.domain_rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "dp_rank": self.dp_rank,
            "ep_rank": self.ep_rank,
            "physical_worker": self.physical_identity,
            "hardware": self.hardware,
            "tp_group": list(self.tp_group),
            "ep_group": list(self.ep_group),
            "is_scheduler_leader": self.is_scheduler_leader,
            "expert_ids": list(self.expert_ids),
            "component_ids": list(self.component_ids),
            "kv_partition": (
                self.kv_partition.as_dict() if self.kv_partition is not None else None
            ),
        }


@dataclass(frozen=True)
class AFPhysicalRuntimeState:
    physical_identity: str
    hardware: str
    capacity_bytes: int
    bindings: tuple[RankKey, ...]
    components: tuple[AFModelComponent, ...]
    model_bytes: int
    kv_bytes_per_block: int
    protocol_buffers: tuple[tuple[str, int], ...] = ()

    @property
    def protocol_buffer_bytes(self) -> int:
        return sum(size for _, size in self.protocol_buffers)

    def component_bytes(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for component in self.components:
            totals[component.category] = (
                totals.get(component.category, 0) + component.bytes
            )
        return totals

    def as_dict(self) -> dict[str, object]:
        return {
            "physical_worker": self.physical_identity,
            "hardware": self.hardware,
            "capacity_bytes": self.capacity_bytes,
            "bindings": [f"{domain}:{rank}" for domain, rank in self.bindings],
            "component_bytes": self.component_bytes(),
            "components": [component.as_dict() for component in self.components],
            "model_bytes": self.model_bytes,
            "kv_bytes_per_block": self.kv_bytes_per_block,
            "protocol_buffers": dict(self.protocol_buffers),
            "protocol_buffer_bytes": self.protocol_buffer_bytes,
        }


@dataclass(frozen=True)
class AFReplicaRuntime:
    domain: str
    capability: str
    instance: str
    dp_rank: int
    rank_keys: tuple[RankKey, ...]
    leader_rank_key: RankKey | None
    local_gpu_blocks: tuple[tuple[RankKey, int], ...]
    local_cpu_blocks: tuple[tuple[RankKey, int], ...]
    logical_gpu_blocks: int
    logical_cpu_blocks: int

    @property
    def replica_key(self) -> ReplicaKey:
        return (self.domain, self.dp_rank)

    @property
    def is_attention(self) -> bool:
        return self.capability == "af_attention"

    def gpu_capacity_by_rank(self) -> dict[RankKey, int]:
        return dict(self.local_gpu_blocks)

    def cpu_capacity_by_rank(self) -> dict[RankKey, int]:
        return dict(self.local_cpu_blocks)

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "capability": self.capability,
            "instance": self.instance,
            "dp_rank": self.dp_rank,
            "rank_keys": [f"{domain}:{rank}" for domain, rank in self.rank_keys],
            "leader_rank_key": (
                f"{self.leader_rank_key[0]}:{self.leader_rank_key[1]}"
                if self.leader_rank_key is not None
                else None
            ),
            "local_gpu_blocks": {
                f"{domain}:{rank}": value
                for (domain, rank), value in self.local_gpu_blocks
            },
            "local_cpu_blocks": {
                f"{domain}:{rank}": value
                for (domain, rank), value in self.local_cpu_blocks
            },
            "logical_gpu_blocks": self.logical_gpu_blocks,
            "logical_cpu_blocks": self.logical_cpu_blocks,
        }


@dataclass(frozen=True)
class AFDomainRuntime:
    name: str
    capability: str
    tensor_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int
    expert_placement_strategy: str
    expert_parallel_mode: Literal["expert_tp", "rank_reuse", "expert_sharded"] = (
        "expert_tp"
    )
    expert_tensor_parallel_size: int = 1

    def parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=1,
            data_parallel_size=self.data_parallel_size,
            enable_expert_parallel=self.expert_parallel_size > 1,
            expert_placement_strategy=self.expert_placement_strategy,
        )
