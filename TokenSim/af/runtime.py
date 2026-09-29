from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Mapping

from TokenSim.config.cluster_config import (
    AFEffectiveDTypes,
    ClusterConfig,
    DomainCollectiveConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    PhysicalWorkerRef,
    WorkerConfig,
)
from TokenSim.config.constants import _GB
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.dtypes import canonicalize_dtype, dtype_size_bytes
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.moe.config import MoEModelConfig
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
    expert_parallel_mode: Literal["expert_tp", "rank_reuse"] = "expert_tp"
    expert_tensor_parallel_size: int = 1

    def parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=1,
            data_parallel_size=self.data_parallel_size,
            enable_expert_parallel=self.expert_parallel_size > 1,
            expert_placement_strategy=self.expert_placement_strategy,
        )


@dataclass(frozen=True)
class AFRuntimePlan:
    schema_version: str
    model: str
    block_size: int
    effective_dtypes: AFEffectiveDTypes
    total_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    domains: tuple[AFDomainRuntime, ...]
    rank_bindings: tuple[AFRankBinding, ...]
    replicas: tuple[AFReplicaRuntime, ...]
    physical_states: tuple[AFPhysicalRuntimeState, ...]
    layers: tuple[AFLayerRuntime, ...] = ()
    request_owners: tuple[AFRequestOwner, ...] = ()
    tensor_ownerships: tuple[AFTensorOwnership, ...] = ()
    partial_destinations: tuple[AFPartialDestination, ...] = ()
    collective_plans: tuple[AFResolvedCollectivePlan, ...] = ()
    link_effective_dtypes: tuple[tuple[str, AFEffectiveDTypes], ...] = ()
    mla: MLADimensions | None = None

    @property
    def dtype(self) -> str:
        return self.effective_dtypes.activation_dtype

    @property
    def dtype_size_bytes(self) -> int:
        return dtype_size_bytes(self.dtype, role="activation_dtype")

    @property
    def weight_dtype_size_bytes(self) -> int:
        return dtype_size_bytes(
            self.effective_dtypes.weight_dtype, role="weight_dtype"
        )

    @property
    def kv_dtype_size_bytes(self) -> int:
        return dtype_size_bytes(
            self.effective_dtypes.kv_cache_dtype, role="kv_cache_dtype"
        )

    def layer_plan(self, layer_id: int) -> AFLayerRuntime:
        try:
            return self.layers[layer_id]
        except (IndexError, TypeError) as exc:
            raise ConfigurationError(
                f"AF runtime plan has no layer metadata for layer_id={layer_id}"
            ) from exc

    def dtypes_for_link(self, link_name: str) -> AFEffectiveDTypes:
        for name, effective in self.link_effective_dtypes:
            if name == link_name:
                return effective
        if not self.link_effective_dtypes:
            return self.effective_dtypes
        raise ConfigurationError(
            f"AF runtime plan has no dtype contract for link {link_name!r}"
        )

    def ownerships_for_replica(
        self, domain: str, dp_rank: int
    ) -> tuple[AFTensorOwnership, ...]:
        domain_plan = self.domain_runtime(domain)
        candidates = tuple(
            item
            for item in self.tensor_ownerships
            if item.domain == domain and item.dp_rank == dp_rank
        )
        if domain_plan.capability == "af_ffn" and domain_plan.expert_parallel_size > 1:
            # Residual activations have one base TP owner.  EP ranks in a
            # rank-reuse F pool replicate that owner and must not create
            # overlapping tensor intervals.
            by_tp: dict[int, AFTensorOwnership] = {}
            for item in sorted(candidates, key=lambda value: value.domain_rank):
                by_tp.setdefault(item.tp_rank, item)
            candidates = tuple(by_tp.values())
        ownerships = tuple(
            sorted(
                candidates,
                key=lambda item: item.tp_rank,
            )
        )
        if not ownerships:
            raise ConfigurationError(
                f"AF runtime plan has no tensor ownership for domain={domain!r}, "
                f"dp_rank={dp_rank}"
            )
        return ownerships

    def collective_plan(
        self, domain: str, dp_rank: int, after_stage: str
    ) -> AFResolvedCollectivePlan | None:
        matches = tuple(
            item
            for item in self.collective_plans
            if item.domain == domain
            and item.dp_rank == dp_rank
            and item.after_stage == after_stage
        )
        if len(matches) > 1:
            raise ConfigurationError(
                f"AF runtime plan has ambiguous collective for domain={domain!r}, "
                f"dp_rank={dp_rank}, after_stage={after_stage!r}"
            )
        return matches[0] if matches else None

    def local_shape(
        self,
        domain: str,
        domain_rank: int,
        layer_id: int,
        operation: str,
    ) -> dict[str, int]:
        binding = self.binding(domain, domain_rank)
        domain_plan = self.domain_runtime(domain)
        layer = self.layer_plan(layer_id)
        shape = {
            "global_hidden_size": self.hidden_size,
            "hidden_size": self.hidden_size,
            "tp_size": domain_plan.tensor_parallel_size,
        }
        if operation in {"attention", "a_rank_compute"}:
            if binding.kv_partition is None:
                raise ConfigurationError(
                    f"attention rank {binding.rank_key!r} has no KV partition"
                )
            shape.update(
                {
                    "num_heads": self.num_attention_heads
                    // domain_plan.tensor_parallel_size,
                    "num_kv_heads": binding.kv_partition.local_head_count,
                    "head_dim": self.head_dim,
                    "query_width": (
                        self.num_attention_heads
                        // domain_plan.tensor_parallel_size
                        * self.head_dim
                    ),
                    "kv_width": (
                        binding.kv_partition.local_head_count * self.head_dim
                    ),
                }
            )
            if self.mla is not None:
                shape.update(self.mla.local_shape(self.num_attention_heads, domain_plan.tensor_parallel_size))
        elif operation == "dense_ffn":
            if layer.layer_type != "dense":
                raise ConfigurationError(
                    f"layer={layer_id} is {layer.layer_type}, not Dense FFN"
                )
            shape["intermediate_size"] = (
                layer.dense_intermediate_size
                // domain_plan.tensor_parallel_size
            )
        elif operation == "shared_expert":
            if layer.layer_type != "moe" or not layer.shared_intermediate_size:
                raise ConfigurationError(
                    f"layer={layer_id} has no Shared Expert component"
                )
            shape["intermediate_size"] = (
                layer.shared_intermediate_size
                // domain_plan.tensor_parallel_size
            )
        elif operation in {"routed_expert", "expert_rank_compute"}:
            if layer.layer_type != "moe" or not layer.num_experts:
                raise ConfigurationError(
                    f"layer={layer_id} has no Routed Expert component"
                )
            expert_tp_size = domain_plan.expert_tensor_parallel_size
            shape.update(
                {
                    "tp_size": expert_tp_size,
                    "expert_tp_size": expert_tp_size,
                    "expert_tp_rank": (
                        binding.tp_rank if expert_tp_size > 1 else 0
                    ),
                    "intermediate_size": (
                        layer.routed_intermediate_size // expert_tp_size
                    ),
                }
            )
        return shape

    def expert_execution(
        self, domain: str, expert_id: int, f_service_dp_rank: int
    ) -> tuple[AFRankBinding, ...]:
        domain_plan = self.domain_runtime(domain)
        if domain_plan.capability != "af_ffn":
            raise ConfigurationError(
                f"expert execution requires an F domain, got {domain!r}"
            )
        if not 0 <= f_service_dp_rank < domain_plan.data_parallel_size:
            raise ConfigurationError(
                f"F domain={domain!r} has no service replica "
                f"dp_rank={f_service_dp_rank}"
            )
        candidates = tuple(
            binding
            for binding in self.rank_bindings
            if binding.domain == domain and expert_id in binding.expert_ids
        )
        if domain_plan.expert_parallel_mode == "expert_tp":
            candidates = tuple(
                binding
                for binding in candidates
                if binding.dp_rank == f_service_dp_rank
            )
        resolved = tuple(
            sorted(candidates, key=lambda item: (item.dp_rank, item.tp_rank))
        )
        if len(resolved) != domain_plan.expert_tensor_parallel_size:
            raise ConfigurationError(
                f"expert={expert_id} in F domain={domain!r} resolves to "
                f"{len(resolved)} ranks; expected Expert TP="
                f"{domain_plan.expert_tensor_parallel_size}"
            )
        if len({binding.physical_identity for binding in resolved}) != len(resolved):
            raise ConfigurationError(
                f"expert={expert_id} in F domain={domain!r} resolves multiple "
                "Expert TP ranks to the same physical worker"
            )
        return resolved

    def component_execution(
        self,
        domain: str,
        layer_id: int,
        component_category: Literal["dense_ffn", "shared_expert"],
        f_service_dp_rank: int,
    ) -> tuple[AFRankBinding, ...]:
        domain_plan = self.domain_runtime(domain)
        if domain_plan.capability != "af_ffn":
            raise ConfigurationError(
                f"F component execution requires an F domain, got {domain!r}"
            )
        if component_category not in {"dense_ffn", "shared_expert"}:
            raise ConfigurationError(
                f"unsupported F TP component {component_category!r}; expected "
                "'dense_ffn' or 'shared_expert'"
            )
        if not 0 <= f_service_dp_rank < domain_plan.data_parallel_size:
            raise ConfigurationError(
                f"F domain={domain!r} has no service replica "
                f"dp_rank={f_service_dp_rank}"
            )

        layer = self.layer_plan(layer_id)
        if component_category == "dense_ffn" and layer.layer_type != "dense":
            raise ConfigurationError(
                f"layer={layer_id} is {layer.layer_type}, not Dense FFN"
            )
        if component_category == "shared_expert" and (
            layer.layer_type != "moe" or not layer.shared_intermediate_size
        ):
            raise ConfigurationError(
                f"layer={layer_id} has no Shared Expert component"
            )

        candidates = tuple(
            binding
            for binding in self.rank_bindings
            if binding.domain == domain and binding.dp_rank == f_service_dp_rank
        )
        if domain_plan.capability == "af_ffn" and domain_plan.expert_parallel_size > 1:
            # Dense/shared weights are replicated on every EP rank in a
            # rank-reuse F pool.  Execute one representative per base TP
            # coordinate; routed experts are resolved separately by
            # ``expert_execution``.
            by_tp: dict[int, AFRankBinding] = {}
            for binding in sorted(candidates, key=lambda item: item.domain_rank):
                by_tp.setdefault(binding.tp_rank, binding)
            resolved = tuple(sorted(by_tp.values(), key=lambda item: item.tp_rank))
        else:
            resolved = tuple(sorted(candidates, key=lambda item: item.tp_rank))
        if len(resolved) != domain_plan.tensor_parallel_size:
            raise ConfigurationError(
                f"F component={component_category!r}, layer={layer_id}, "
                f"domain={domain!r}, dp_rank={f_service_dp_rank} resolves to "
                f"{len(resolved)} ranks; expected F TP="
                f"{domain_plan.tensor_parallel_size}"
            )
        expected_group = tuple(binding.domain_rank for binding in resolved)
        for binding in resolved:
            if binding.tp_group != expected_group:
                raise ConfigurationError(
                    f"F component={component_category!r}, layer={layer_id} rank "
                    f"{binding.rank_key!r} has TP group {binding.tp_group}; "
                    f"expected {expected_group}"
                )
            state = self.physical_state(binding.physical_identity)
            matches = tuple(
                component
                for component in state.components
                if component.category == component_category
                and component.layer_id == layer_id
                and component.component_id in binding.component_ids
            )
            if len(matches) != 1:
                raise ConfigurationError(
                    f"F component={component_category!r}, layer={layer_id} does "
                    f"not resolve uniquely on domain={domain!r}, "
                    f"rank={binding.domain_rank}; found {len(matches)}"
                )
        if len({binding.physical_identity for binding in resolved}) != len(resolved):
            raise ConfigurationError(
                f"F component={component_category!r}, layer={layer_id}, "
                f"domain={domain!r}, dp_rank={f_service_dp_rank} reuses a "
                "physical worker inside its TP group"
            )
        return resolved

    def binding(self, domain: str, domain_rank: int) -> AFRankBinding:
        for binding in self.rank_bindings:
            if binding.domain == domain and binding.domain_rank == domain_rank:
                return binding
        raise ConfigurationError(
            f"AF runtime plan has no rank binding for domain={domain!r}, "
            f"rank={domain_rank}"
        )

    def bindings_for_instance(
        self, domain: str, instance: str
    ) -> tuple[AFRankBinding, ...]:
        bindings = tuple(
            sorted(
                (
                    binding
                    for binding in self.rank_bindings
                    if binding.domain == domain and binding.instance == instance
                ),
                key=lambda binding: (binding.ep_rank, binding.tp_rank),
            )
        )
        if not bindings:
            raise ConfigurationError(
                f"AF runtime plan has no instance {instance!r} in domain {domain!r}"
            )
        return bindings

    def instance_for_physical(self, domain: str, physical_identity: str) -> str:
        instances = {
            binding.instance
            for binding in self.rank_bindings
            if binding.domain == domain
            and binding.physical_identity == physical_identity
        }
        if len(instances) != 1:
            raise ConfigurationError(
                f"AF runtime plan expected one instance in domain {domain!r} for "
                f"physical worker {physical_identity!r}; found {sorted(instances)}"
            )
        return next(iter(instances))

    def replica(self, domain: str, dp_rank: int) -> AFReplicaRuntime:
        for replica in self.replicas:
            if replica.domain == domain and replica.dp_rank == dp_rank:
                return replica
        raise ConfigurationError(
            f"AF runtime plan has no replica for domain={domain!r}, dp_rank={dp_rank}"
        )

    def replica_for_instance(self, domain: str, instance: str) -> AFReplicaRuntime:
        for replica in self.replicas:
            if replica.domain == domain and replica.instance == instance:
                return replica
        raise ConfigurationError(
            f"AF runtime plan has no replica {instance!r} in domain {domain!r}"
        )

    def scheduler_replica_for_physical(
        self, physical_identity: str
    ) -> AFReplicaRuntime:
        return self.scheduler_replicas_for_physical(physical_identity)[0]

    def scheduler_replicas_for_physical(
        self, physical_identity: str
    ) -> tuple[AFReplicaRuntime, ...]:
        matches = []
        for replica in self.replicas:
            if replica.leader_rank_key is None:
                continue
            binding = self.binding(*replica.leader_rank_key)
            if binding.physical_identity == physical_identity:
                matches.append(replica)
        if not matches:
            raise ConfigurationError(
                f"physical worker {physical_identity!r} is not an A scheduler leader"
            )
        signatures = {self._scheduler_replica_signature(replica) for replica in matches}
        if len(signatures) != 1:
            raise ConfigurationError(
                f"physical worker {physical_identity!r} leads incompatible A replicas "
                f"{[(item.domain, item.dp_rank) for item in matches]}"
            )
        return tuple(matches)

    def _scheduler_replica_signature(
        self, replica: AFReplicaRuntime
    ) -> tuple[object, ...]:
        domain = self.domain_runtime(replica.domain)
        bindings = [self.binding(*rank_key) for rank_key in replica.rank_keys]
        return (
            domain.tensor_parallel_size,
            domain.data_parallel_size,
            replica.dp_rank,
            tuple(
                (binding.tp_rank, binding.physical_identity)
                for binding in sorted(bindings, key=lambda item: item.tp_rank)
            ),
        )

    def physical_state(self, physical_identity: str) -> AFPhysicalRuntimeState:
        for state in self.physical_states:
            if state.physical_identity == physical_identity:
                return state
        raise ConfigurationError(
            f"AF runtime plan has no physical state for {physical_identity!r}"
        )

    def bindings_for_physical(
        self, physical_identity: str
    ) -> tuple[AFRankBinding, ...]:
        return tuple(
            binding
            for binding in self.rank_bindings
            if binding.physical_identity == physical_identity
        )

    def domain_runtime(self, domain: str) -> AFDomainRuntime:
        for item in self.domains:
            if item.name == domain:
                return item
        raise ConfigurationError(f"AF runtime plan has no domain {domain!r}")

    def expert_placement(
        self, domain: str, instance: str
    ) -> tuple[tuple[int, int, tuple[int, ...], tuple[str, ...]], ...]:
        by_expert: dict[int, list[AFRankBinding]] = {}
        domain_plan = self.domain_runtime(domain)
        bindings = (
            tuple(
                binding for binding in self.rank_bindings if binding.domain == domain
            )
            if domain_plan.expert_parallel_mode == "rank_reuse"
            else self.bindings_for_instance(domain, instance)
        )
        for binding in bindings:
            for expert_id in binding.expert_ids:
                by_expert.setdefault(expert_id, []).append(binding)
        return tuple(
            (
                expert_id,
                bindings[0].ep_rank,
                tuple(binding.domain_rank for binding in bindings),
                tuple(binding.physical_identity for binding in bindings),
            )
            for expert_id, bindings in sorted(by_expert.items())
        )

    def validate_execution_node(self, node: Any) -> None:
        if node.kind != "compute":
            return
        if len(node.domain_ranks) != 1 or len(node.physical_workers) != 1:
            raise SimulationStateError(
                f"runtime node {node.node_id!r} must bind one rank and one worker"
            )
        binding = self.binding(node.source_domain, node.domain_ranks[0])
        if binding.physical_identity != node.physical_workers[0]:
            raise SimulationStateError(
                f"runtime node {node.node_id!r} physical mismatch for "
                f"domain={binding.domain}, rank={binding.domain_rank}: "
                f"expected={binding.physical_identity}, actual={node.physical_workers[0]}"
            )
        expected_tp_group = binding.tp_group
        if node.stage in {"expert", "routed_expert"}:
            domain_plan = self.domain_runtime(binding.domain)
            if domain_plan.expert_parallel_mode == "rank_reuse":
                expected_tp_group = (binding.domain_rank,)
        if node.tp_group and tuple(node.tp_group) != expected_tp_group:
            raise SimulationStateError(
                f"runtime node {node.node_id!r} TP group mismatch for "
                f"domain={binding.domain}, rank={binding.domain_rank}: "
                f"expected={expected_tp_group}, actual={tuple(node.tp_group)}"
            )
        expert_ids = tuple(getattr(node.local_work, "expert_ids", ()) or ())
        if node.stage in {"expert", "routed_expert"} and not set(
            expert_ids
        ).issubset(binding.expert_ids):
            raise SimulationStateError(
                f"runtime node {node.node_id!r} requests experts {expert_ids} on "
                f"domain={binding.domain}, rank={binding.domain_rank}, "
                f"resident={binding.expert_ids}"
            )

    def validate(self) -> None:
        if tuple(layer.layer_id for layer in self.layers) != tuple(
            range(self.total_layers)
        ):
            raise ConfigurationError("AF runtime layers must cover every model layer once")
        rank_keys = [binding.rank_key for binding in self.rank_bindings]
        if len(rank_keys) != len(set(rank_keys)):
            raise ConfigurationError("AF runtime plan contains duplicate rank bindings")
        physical_ids = [state.physical_identity for state in self.physical_states]
        if len(physical_ids) != len(set(physical_ids)):
            raise ConfigurationError("AF runtime plan contains duplicate physical states")
        state_by_physical = {
            state.physical_identity: state for state in self.physical_states
        }
        layer_by_id = {layer.layer_id: layer for layer in self.layers}
        for binding in self.rank_bindings:
            state = state_by_physical.get(binding.physical_identity)
            if state is None or binding.rank_key not in state.bindings:
                raise ConfigurationError(
                    f"AF runtime binding domain={binding.domain}, "
                    f"rank={binding.domain_rank} has no matching physical state"
                )
            components = {
                item.component_id: item for item in state.components
            }
            if not set(binding.component_ids).issubset(components):
                raise ConfigurationError(
                    f"AF runtime domain={binding.domain}, rank={binding.domain_rank} "
                    "component ownership differs from physical ledger"
                )
            allowed = (
                {"attention", "embedding_lm_head", "router"}
                if binding.capability == "af_attention"
                else {"dense_ffn", "routed_expert", "shared_expert", "router"}
            )
            for component_id in binding.component_ids:
                component = components[component_id]
                if component.category not in allowed:
                    raise ConfigurationError(
                        f"AF runtime rank {binding.rank_key!r} has invalid component "
                        f"category {component.category!r}"
                    )
                if component.layer_id is None:
                    continue
                layer = layer_by_id[component.layer_id]
                if component.category == "router" and layer.layer_type != "moe":
                    raise ConfigurationError("Dense layer cannot own a Router component")
                if component.category == "dense_ffn" and layer.layer_type != "dense":
                    raise ConfigurationError("MoE layer cannot own a Dense FFN component")
                if component.category in {"routed_expert", "shared_expert"} and (
                    layer.layer_type != "moe"
                ):
                    raise ConfigurationError(
                        "Dense layer cannot own Routed/Shared Expert components"
                    )
        for state in self.physical_states:
            expected_bindings = tuple(
                sorted(
                    binding.rank_key
                    for binding in self.rank_bindings
                    if binding.physical_identity == state.physical_identity
                )
            )
            if tuple(sorted(state.bindings)) != expected_bindings:
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} binding "
                    "mismatch"
                )
            component_ids = [item.component_id for item in state.components]
            if len(component_ids) != len(set(component_ids)):
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} counts a "
                    "component more than once"
                )
            if state.model_bytes != sum(item.bytes for item in state.components):
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} model byte "
                    "total does not match its components"
                )
            if state.model_bytes >= state.capacity_bytes:
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} exceeds "
                    f"capacity: model_bytes={state.model_bytes}, "
                    f"capacity_bytes={state.capacity_bytes}"
                )

        domain_by_name = {domain.name: domain for domain in self.domains}
        for domain in self.domains:
            domain_bindings = tuple(
                item for item in self.rank_bindings if item.domain == domain.name
            )
            expected_count = domain.tensor_parallel_size * domain.data_parallel_size
            if len(domain_bindings) != expected_count:
                raise ConfigurationError(
                    f"AF runtime domain={domain.name} rank count mismatch: "
                    f"expected base TP*DP={expected_count}, "
                    f"actual={len(domain_bindings)}"
                )
            if domain.capability == "af_ffn" and domain.expert_parallel_mode not in {
                "expert_tp",
                "rank_reuse",
            }:
                raise ConfigurationError(
                    f"AF runtime domain={domain.name} has unsupported EP mode"
                )
            if domain.capability == "af_attention":
                if any(binding.ep_group for binding in domain_bindings):
                    raise ConfigurationError(
                        f"A domain={domain.name!r} must not have an Expert group"
                    )
                continue
            if domain.expert_parallel_mode == "expert_tp":
                expected_ep_size = 1
                expected_expert_tp = domain.tensor_parallel_size
            else:
                expected_ep_size = expected_count
                expected_expert_tp = 1
            if domain.expert_parallel_size != expected_ep_size or (
                domain.expert_tensor_parallel_size != expected_expert_tp
            ):
                raise ConfigurationError(
                    f"F domain={domain.name!r} EP contract mismatch: mode="
                    f"{domain.expert_parallel_mode}, EP={domain.expert_parallel_size}, "
                    f"Expert TP={domain.expert_tensor_parallel_size}; expected "
                    f"EP={expected_ep_size}, Expert TP={expected_expert_tp}"
                )
            world_group = tuple(
                binding.domain_rank
                for binding in sorted(
                    domain_bindings, key=lambda binding: binding.domain_rank
                )
            )
            for binding in domain_bindings:
                expected_ep_group = (
                    binding.tp_group
                    if domain.expert_parallel_mode == "expert_tp"
                    else world_group
                )
                if binding.ep_group != expected_ep_group:
                    raise ConfigurationError(
                        f"F rank {binding.rank_key!r} EP group mismatch: "
                        f"expected={expected_ep_group}, actual={binding.ep_group}"
                    )

        for replica in self.replicas:
            domain = domain_by_name.get(replica.domain)
            if domain is None:
                raise ConfigurationError(
                    f"AF runtime replica references missing domain {replica.domain!r}"
                )
            bindings = tuple(
                sorted(
                    (self.binding(*rank_key) for rank_key in replica.rank_keys),
                    key=lambda item: item.tp_rank,
                )
            )
            if any(item.dp_rank != replica.dp_rank for item in bindings):
                raise ConfigurationError(
                    f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                    "contains a rank from another DP replica"
                )
            if {item.tp_rank for item in bindings} != set(
                range(domain.tensor_parallel_size)
            ):
                raise ConfigurationError(
                    f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                    "does not contain exactly one base rank per TP coordinate"
                )
            expected_tp_group = tuple(item.domain_rank for item in bindings)
            for binding in bindings:
                expected_ep_rank = (
                    binding.dp_rank * domain.tensor_parallel_size + binding.tp_rank
                    if domain.capability == "af_ffn"
                    and domain.expert_parallel_mode == "rank_reuse"
                    else 0
                )
                if binding.ep_rank != expected_ep_rank:
                    raise ConfigurationError(
                        f"AF runtime rank {binding.rank_key!r} EP identity mismatch: "
                        f"expected={expected_ep_rank}, actual={binding.ep_rank}"
                    )
                if binding.tp_group != expected_tp_group:
                    raise ConfigurationError(
                        f"AF runtime rank {binding.rank_key!r} TP group mismatch"
                    )
            if replica.is_attention:
                leaders = [item for item in bindings if item.is_scheduler_leader]
                if len(leaders) != 1 or leaders[0].rank_key != replica.leader_rank_key:
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "must have exactly one scheduler leader"
                    )
                expected_partitions = kv_head_partitions(
                    num_kv_heads=self.num_kv_heads,
                    tensor_parallel_size=domain.tensor_parallel_size,
                    head_dim=self.head_dim,
                    num_layers=self.total_layers,
                    block_size=self.block_size,
                    dtype_size_bytes=self.kv_dtype_size_bytes,
                    mla_cache_width=self.mla.cache_width if self.mla else 0,
                )
                for binding in bindings:
                    if binding.kv_partition != expected_partitions[binding.tp_rank]:
                        raise ConfigurationError(
                            f"AF runtime rank {binding.rank_key!r} KV partition mismatch"
                        )
                gpu_capacities = replica.gpu_capacity_by_rank()
                cpu_capacities = replica.cpu_capacity_by_rank()
                if replica.logical_gpu_blocks != min(gpu_capacities.values()):
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "logical GPU capacity is not the smallest rank capacity"
                    )
                if replica.logical_cpu_blocks != min(cpu_capacities.values()):
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "logical CPU capacity is not the smallest rank capacity"
                    )
            else:
                if replica.leader_rank_key is not None:
                    raise ConfigurationError("F replica must not own a scheduler leader")
                if any(binding.kv_partition is not None for binding in bindings):
                    raise ConfigurationError("F ranks must not own KV state")
                if domain.expert_parallel_mode == "expert_tp" and self.num_experts:
                    for expert_id in range(self.num_experts):
                        owners = [
                            item for item in bindings if expert_id in item.expert_ids
                        ]
                        if {item.tp_rank for item in owners} != set(
                            range(domain.tensor_parallel_size)
                        ):
                            raise ConfigurationError(
                                f"F domain={domain.name}, replica={replica.dp_rank}, "
                                f"expert={expert_id} lacks exact Expert TP shards"
                            )

        for domain in self.domains:
            if (
                domain.capability != "af_ffn"
                or domain.expert_parallel_mode != "rank_reuse"
                or not self.num_experts
            ):
                continue
            domain_bindings = tuple(
                item for item in self.rank_bindings if item.domain == domain.name
            )
            expected_ep_group = tuple(
                item.domain_rank
                for item in sorted(
                    domain_bindings, key=lambda item: item.domain_rank
                )
            )
            for binding in domain_bindings:
                if binding.ep_group != expected_ep_group:
                    raise ConfigurationError(
                        f"rank-reuse F rank {binding.rank_key!r} EP group mismatch"
                    )
            for expert_id in range(self.num_experts):
                owners = [
                    item for item in domain_bindings if expert_id in item.expert_ids
                ]
                if len(owners) != 1:
                    raise ConfigurationError(
                        f"rank-reuse F domain={domain.name}, expert={expert_id} must "
                        f"resolve to one physical rank; got {len(owners)}"
                    )

        for domain in self.domains:
            if domain.capability != "af_ffn":
                continue
            for dp_rank in range(domain.data_parallel_size):
                for layer in self.layers:
                    if layer.layer_type == "dense":
                        self.component_execution(
                            domain.name, layer.layer_id, "dense_ffn", dp_rank
                        )
                    elif layer.shared_intermediate_size:
                        self.component_execution(
                            domain.name, layer.layer_id, "shared_expert", dp_rank
                        )
                if domain.expert_parallel_mode == "expert_tp":
                    for expert_id in range(self.num_experts):
                        self.expert_execution(domain.name, expert_id, dp_rank)
            if domain.expert_parallel_mode == "rank_reuse":
                for expert_id in range(self.num_experts):
                    self.expert_execution(domain.name, expert_id, 0)

        for domain in self.domains:
            for dp_rank in range(domain.data_parallel_size):
                ownerships = self.ownerships_for_replica(domain.name, dp_rank)
                cursor = 0
                for ownership in ownerships:
                    binding = self.binding(
                        ownership.domain, ownership.domain_rank
                    )
                    if ownership.physical_worker != binding.physical_identity:
                        raise ConfigurationError(
                            f"AF tensor ownership for rank {binding.rank_key!r} "
                            "references a different physical worker"
                        )
                    expected_residual = binding.capability == "af_attention"
                    if ownership.retains_residual != expected_residual:
                        raise ConfigurationError(
                            f"AF tensor ownership for rank {binding.rank_key!r} "
                            f"retains_residual={ownership.retains_residual}; "
                            f"expected={expected_residual}"
                        )
                    if ownership.interval.start != cursor:
                        raise ConfigurationError(
                            f"AF tensor ownership for domain={domain.name}, "
                            f"dp_rank={dp_rank} has a gap or overlap"
                        )
                    cursor = ownership.interval.end
                if cursor != self.hidden_size:
                    raise ConfigurationError(
                        f"AF tensor ownership for domain={domain.name}, "
                        f"dp_rank={dp_rank} does not cover hidden_size={self.hidden_size}"
                    )

        ownership_by_rank = {
            (item.domain, item.domain_rank): item for item in self.tensor_ownerships
        }
        destination_keys = [
            (
                item.phase,
                item.source_domain,
                item.source_rank,
                item.target_domain,
                item.target_rank,
                item.interval.start,
                item.interval.end,
                item.source_layout,
                item.target_layout,
            )
            for item in self.partial_destinations
        ]
        if len(destination_keys) != len(set(destination_keys)):
            raise ConfigurationError("AF runtime has duplicate partial destinations")
        for destination in self.partial_destinations:
            source = ownership_by_rank.get(
                (destination.source_domain, destination.source_rank)
            )
            target = ownership_by_rank.get(
                (destination.target_domain, destination.target_rank)
            )
            if source is None or target is None:
                raise ConfigurationError(
                    "AF partial destination references a missing tensor owner"
                )
            if source.dp_rank != destination.source_dp_rank or (
                target.dp_rank != destination.target_dp_rank
            ):
                raise ConfigurationError(
                    "AF partial destination DP owner does not match its rank"
                )
            expected_interval = source.interval.intersection(target.interval)
            if expected_interval != destination.interval:
                raise ConfigurationError(
                    f"AF partial destination {destination.source_domain}:"
                    f"{destination.source_rank} -> {destination.target_domain}:"
                    f"{destination.target_rank} has interval "
                    f"{destination.interval}; expected {expected_interval}"
                )

        collective_keys = [
            (item.domain, item.dp_rank, item.after_stage)
            for item in self.collective_plans
        ]
        if len(collective_keys) != len(set(collective_keys)):
            raise ConfigurationError("AF runtime has ambiguous collective plans")
        for collective in self.collective_plans:
            if not collective.participant_ranks:
                raise ConfigurationError("AF collective must have participants")
            for rank, physical in zip(
                collective.participant_ranks, collective.physical_workers
            ):
                binding = self.binding(collective.domain, rank)
                if binding.dp_rank != collective.dp_rank or (
                    binding.physical_identity != physical
                ):
                    raise ConfigurationError(
                        f"AF collective {collective.operation!r} endpoint mismatch"
                    )

    def summary(self, replica_managers: Mapping[ReplicaKey, Any] | None = None) -> dict:
        managers = replica_managers or {}
        physical = []
        for state in self.physical_states:
            data = state.as_dict()
            rank_usage = []
            for binding in self.bindings_for_physical(state.physical_identity):
                replica = self.replica(binding.domain, binding.dp_rank)
                manager = managers.get(replica.replica_key)
                usage = (
                    manager.rank_status(binding.rank_key)
                    if manager is not None and replica.is_attention
                    else {
                        "gpu_capacity": replica.gpu_capacity_by_rank().get(
                            binding.rank_key, 0
                        ),
                        "gpu_used": 0,
                        "cpu_capacity": replica.cpu_capacity_by_rank().get(
                            binding.rank_key, 0
                        ),
                        "cpu_used": 0,
                    }
                )
                rank_usage.append(
                    {
                        "domain": binding.domain,
                        "domain_rank": binding.domain_rank,
                        **usage,
                    }
                )
            data["rank_usage"] = rank_usage
            physical.append(data)
        replicas = []
        for replica in self.replicas:
            data = replica.as_dict()
            manager = managers.get(replica.replica_key)
            if manager is not None:
                free_gpu, used_gpu, total_gpu = manager.get_gpu_status()
                free_cpu, used_cpu, total_cpu = manager.get_cpu_status()
                data["logical_gpu_usage"] = {
                    "free": free_gpu,
                    "used": used_gpu,
                    "capacity": total_gpu,
                }
                data["logical_cpu_usage"] = {
                    "free": free_cpu,
                    "used": used_cpu,
                    "capacity": total_cpu,
                }
            replicas.append(data)
        return {
            "schema": "tokensim.af_runtime_state",
            "version": self.schema_version,
            "model": self.model,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "effective_dtypes": self.effective_dtypes.as_dict(),
            "link_effective_dtypes": {
                name: effective.as_dict()
                for name, effective in self.link_effective_dtypes
            },
            "model_shape": {
                "total_layers": self.total_layers,
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "mla": vars(self.mla) if self.mla else None,
            },
            "domains": [
                {
                    "name": domain.name,
                    "capability": domain.capability,
                    "tensor_parallel_size": domain.tensor_parallel_size,
                    "data_parallel_size": domain.data_parallel_size,
                    "expert_parallel_size": domain.expert_parallel_size,
                    "expert_parallel_mode": domain.expert_parallel_mode,
                    "expert_placement_strategy": (
                        domain.expert_placement_strategy
                    ),
                    "base_world_size": (
                        domain.tensor_parallel_size * domain.data_parallel_size
                    ),
                    "expert_tensor_parallel_size": (
                        domain.expert_tensor_parallel_size
                    ),
                }
                for domain in self.domains
            ],
            "rank_bindings": [
                binding.as_dict() for binding in self.rank_bindings
            ],
            "layers": [layer.as_dict() for layer in self.layers],
            "request_owners": [owner.as_dict() for owner in self.request_owners],
            "tensor_ownerships": [
                ownership.as_dict() for ownership in self.tensor_ownerships
            ],
            "partial_destinations": [
                destination.as_dict() for destination in self.partial_destinations
            ],
            "collective_plans": [
                collective.as_dict() for collective in self.collective_plans
            ],
            "physical_workers": physical,
            "replicas": replicas,
        }


@dataclass
class AFCacheConfig:
    block_size: int
    model: str
    num_gpu_blocks: int
    num_cpu_blocks: int
    size_per_token: int
    size_per_token_unsharded: int
    model_param_size: int
    model_param_size_unsharded: int
    num_layers_per_rank: int
    total_num_layers: int
    num_kv_heads: int
    local_kv_heads: int
    head_dim: int
    rank_info: ParallelRankInfo

    @classmethod
    def from_plan(
        cls, plan: AFRuntimePlan, replica: AFReplicaRuntime
    ) -> "AFCacheConfig":
        if not replica.is_attention or replica.leader_rank_key is None:
            raise ConfigurationError(
                f"AF cache config requires an A replica, got {replica.domain!r}"
            )
        bindings = [plan.binding(*rank_key) for rank_key in replica.rank_keys]
        leader = plan.binding(*replica.leader_rank_key)
        partitions = [item.kv_partition for item in bindings]
        if any(partition is None for partition in partitions):
            raise ConfigurationError(
                f"AF A replica {replica.domain}:{replica.dp_rank} lacks KV partitions"
            )
        block_bytes = sum(
            partition.bytes_per_block for partition in partitions if partition
        )
        local_partition = leader.kv_partition
        model_bytes = plan.physical_state(leader.physical_identity).model_bytes
        return cls(
            block_size=plan.block_size,
            model=plan.model,
            num_gpu_blocks=replica.logical_gpu_blocks,
            num_cpu_blocks=replica.logical_cpu_blocks,
            size_per_token=block_bytes // plan.block_size,
            size_per_token_unsharded=block_bytes // plan.block_size,
            model_param_size=model_bytes,
            model_param_size_unsharded=sum(
                state.model_bytes for state in plan.physical_states
            ),
            num_layers_per_rank=plan.total_layers,
            total_num_layers=plan.total_layers,
            num_kv_heads=plan.num_kv_heads,
            local_kv_heads=local_partition.local_head_count,
            head_dim=plan.head_dim,
            rank_info=ParallelRankInfo(
                global_rank=leader.domain_rank,
                rank_in_dp_group=leader.tp_rank,
                tp_rank=leader.tp_rank,
                pp_rank=leader.pp_rank,
                dp_rank=leader.dp_rank,
                kv_cache_group_id=f"{leader.domain}:dp{leader.dp_rank}",
            ),
        )


def kv_head_partitions(
    *,
    num_kv_heads: int,
    tensor_parallel_size: int,
    head_dim: int,
    num_layers: int,
    block_size: int,
    dtype_size_bytes: int,
    mla_cache_width: int = 0,
) -> tuple[AFKVHeadPartition, ...]:
    for name, value in (
        ("num_kv_heads", num_kv_heads),
        ("tensor_parallel_size", tensor_parallel_size),
        ("head_dim", head_dim),
        ("num_layers", num_layers),
        ("block_size", block_size),
        ("dtype_size_bytes", dtype_size_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"AF KV partition {name} must be positive")
    if mla_cache_width:
        if type(mla_cache_width) is not int or mla_cache_width < 1 or num_kv_heads != 1:
            raise ConfigurationError("MLA cache requires positive latent+RoPE width and one shared cache")
        token_bytes = mla_cache_width * dtype_size_bytes * num_layers
        return tuple(AFKVHeadPartition(tp_rank=rank, global_head_ids=(0,),
            replica_factor=tensor_parallel_size, bytes_per_token=token_bytes,
            bytes_per_block=token_bytes*block_size) for rank in range(tensor_parallel_size))
    if num_kv_heads >= tensor_parallel_size:
        if num_kv_heads % tensor_parallel_size:
            raise ConfigurationError(
                f"KV heads {num_kv_heads} cannot be evenly sharded across A-domain "
                f"TP={tensor_parallel_size}"
            )
        heads_per_rank = num_kv_heads // tensor_parallel_size
        assignments = tuple(
            tuple(range(rank * heads_per_rank, (rank + 1) * heads_per_rank))
            for rank in range(tensor_parallel_size)
        )
        replica_factor = 1
    else:
        if tensor_parallel_size % num_kv_heads:
            raise ConfigurationError(
                f"A-domain TP={tensor_parallel_size} cannot form unambiguous "
                f"replication groups for {num_kv_heads} KV heads"
            )
        replica_factor = tensor_parallel_size // num_kv_heads
        assignments = tuple(
            (rank // replica_factor,) for rank in range(tensor_parallel_size)
        )
    bytes_per_head_token = head_dim * 2 * dtype_size_bytes * num_layers
    return tuple(
        AFKVHeadPartition(
            tp_rank=rank,
            global_head_ids=head_ids,
            replica_factor=replica_factor,
            bytes_per_token=len(head_ids) * bytes_per_head_token,
            bytes_per_block=len(head_ids) * bytes_per_head_token * block_size,
        )
        for rank, head_ids in enumerate(assignments)
    )


def compile_af_runtime_plan(
    *,
    cluster_config: ClusterConfig,
    worker_configs: list[WorkerConfig],
    model: Any,
    hardware: Any,
    block_size: int,
    dtype: str = "bf16",
    watermark: float = 0.01,
) -> AFRuntimePlan:
    if not cluster_config.af.enabled:
        raise ConfigurationError("AF runtime plan requires af.enabled=true")
    if not 0 <= watermark < 1:
        raise ConfigurationError("AF runtime watermark must be in [0, 1)")
    from TokenSim.config.model_config import ModelSpec
    if not isinstance(model, ModelSpec):
        raise ConfigurationError(f"AF runtime plan requires a ModelSpec, got {type(model).__name__}")
    model_id = model.model_id
    moe_config = model.moe
    try:
        model_dtype = canonicalize_dtype(dtype, role="activation_dtype")
    except ValueError as exc:
        raise ConfigurationError(f"invalid AF runtime dtype: {exc}") from exc
    weight_dtype = canonicalize_dtype(cluster_config.af.weight_dtype or model.dtype, role="weight_dtype")
    activation_dtype = canonicalize_dtype(cluster_config.af.activation_dtype or model_dtype, role="activation_dtype")
    kv_dtype = canonicalize_dtype(cluster_config.af.kv_cache_dtype or model.kv_cache_dtype, role="kv_cache_dtype")
    link_dtypes = tuple((
        link.name,
        AFEffectiveDTypes(weight_dtype=weight_dtype, activation_dtype=activation_dtype,
                          kv_cache_dtype=kv_dtype,
                          a_to_f_dtype=canonicalize_dtype(link.a_to_f_dtype or activation_dtype, role="a_to_f_dtype"),
                          f_to_a_dtype=canonicalize_dtype(link.f_to_a_dtype or activation_dtype, role="f_to_a_dtype")),
    ) for link in cluster_config.af.links)
    if not link_dtypes:
        raise ConfigurationError("AF runtime plan requires at least one AF link")
    decode_link = cluster_config.phase_to_af_link.get("decode")
    primary_link_name = decode_link.name if decode_link is not None else link_dtypes[0][0]
    effective_dtypes = next(
        effective for name, effective in link_dtypes if name == primary_link_name
    )
    weight_bytes = dtype_size_bytes(
        effective_dtypes.weight_dtype, role="weight_dtype"
    )
    kv_bytes = dtype_size_bytes(
        effective_dtypes.kv_cache_dtype, role="kv_cache_dtype"
    )

    hidden = model.hidden_size
    total_layers = model.num_layers
    attention_heads = model.num_attention_heads
    mla = model.mla_dimensions if model.attention_type == "mla" else None
    head_dim = model.head_dim
    kv_heads = 1 if mla else model.num_key_value_heads
    intermediate, intermediate_source = model.intermediate_size, "ModelSpec.intermediate_size"
    moe_intermediate = moe_config.moe_intermediate_size or intermediate
    shared_intermediate = moe_intermediate * (
        moe_config.num_shared_experts if moe_config.enabled else 0
    )
    vocab_size = model.vocab_size
    if vocab_size < 1:
        raise ConfigurationError("AF runtime vocabulary size must be positive")

    moe_layer_ids = set(moe_config.moe_layer_indices(total_layers))
    if moe_config.enabled and len(moe_layer_ids) != moe_config.num_moe_layers:
        raise ConfigurationError(
            "AF runtime MoE layer metadata exceeds the model layer range: "
            f"declared={moe_config.num_moe_layers}, resolved={len(moe_layer_ids)}, "
            f"total_layers={total_layers}"
        )
    layers = tuple(
        AFLayerRuntime(
            layer_id=layer_id,
            mla=mla,
            tie_word_embeddings=model.tie_word_embeddings,
            layer_type="moe" if layer_id in moe_layer_ids else "dense",
            hidden_size=hidden,
            num_attention_heads=attention_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            dense_intermediate_size=intermediate,
            routed_intermediate_size=moe_intermediate,
            shared_intermediate_size=(
                shared_intermediate if layer_id in moe_layer_ids else 0
            ),
            num_experts=(moe_config.num_experts if layer_id in moe_layer_ids else 0),
            top_k=(
                moe_config.num_experts_per_tok if layer_id in moe_layer_ids else 1
            ),
            num_shared_experts=(
                moe_config.num_shared_experts if layer_id in moe_layer_ids else 0
            ),
        )
        for layer_id in range(total_layers)
    )
    _validate_parallel_shapes(
        cluster_config=cluster_config,
        hidden=hidden,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        mla_cache_width=mla.cache_width if mla else 0,
        intermediate=intermediate,
        moe_intermediate=moe_intermediate,
        shared_intermediate=shared_intermediate,
        has_dense=any(layer.layer_type == "dense" for layer in layers),
        has_moe=bool(moe_layer_ids),
    )

    worker_by_physical = {
        worker.physical_worker_ref.identity: worker
        for worker in worker_configs
        if worker.physical_worker_ref is not None
    }
    domain_configs = tuple(
        domain
        for domain in cluster_config.af.execution_domains
        if domain.capability in {"af_attention", "af_ffn"}
    )
    domains = tuple(
        AFDomainRuntime(
            name=domain.name,
            capability=domain.capability,
            tensor_parallel_size=domain.parallel.tensor_parallel_size,
            data_parallel_size=domain.parallel.data_parallel_size,
            expert_parallel_size=(
                domain.expert_parallel.expert_parallel_size
                if domain.capability == "af_ffn"
                else 1
            ),
            expert_placement_strategy=domain.parallel.expert_placement_strategy,
            expert_parallel_mode=domain.expert_parallel.mode,
            expert_tensor_parallel_size=(
                domain.expert_parallel.expert_tensor_parallel_size
            ),
        )
        for domain in domain_configs
    )

    preliminary: list[dict[str, Any]] = []
    kv_by_rank: dict[RankKey, AFKVHeadPartition] = {}
    component_by_id: dict[str, AFModelComponent] = {}
    for domain in domain_configs:
        tp_size = domain.parallel.tensor_parallel_size
        domain_mappings = tuple(
            sorted(
                (
                    mapping
                    for instance in domain.logical_instances
                    for mapping in instance.rank_mapping
                ),
                key=lambda item: item.domain_rank,
            )
        )
        partitions = (
            kv_head_partitions(
                num_kv_heads=kv_heads,
                tensor_parallel_size=tp_size,
                head_dim=head_dim,
                num_layers=total_layers,
                block_size=block_size,
                dtype_size_bytes=kv_bytes,
                mla_cache_width=mla.cache_width if mla else 0,
            )
            if domain.capability == "af_attention"
            else ()
        )
        expert_view = domain.expert_parallel
        experts_by_rank = _experts_by_ep(
            moe_config.num_experts if moe_config.enabled else 0,
            expert_view.expert_parallel_size,
            domain.parallel.expert_placement_strategy,
        )
        for instance in domain.logical_instances:
            mappings = tuple(
                sorted(instance.rank_mapping, key=lambda item: item.tp_rank)
            )
            tp_group = tuple(item.domain_rank for item in mappings)
            for mapping in mappings:
                physical = mapping.physical_worker.identity
                worker = worker_by_physical.get(physical)
                if worker is None:
                    raise ConfigurationError(
                        f"AF runtime domain={domain.name}, rank={mapping.domain_rank} "
                        f"references missing physical worker {physical!r}"
                    )
                if domain.capability == "af_ffn" and expert_view.mode == "rank_reuse":
                    expert_ids = tuple(experts_by_rank.get(mapping.ep_rank, ()))
                    ep_group = tuple(item.domain_rank for item in domain_mappings)
                elif domain.capability == "af_ffn":
                    expert_ids = (
                        tuple(range(moe_config.num_experts))
                        if moe_config.enabled
                        else ()
                    )
                    ep_group = tp_group
                else:
                    expert_ids = ()
                    ep_group = ()
                rank_key = (domain.name, mapping.domain_rank)
                if partitions:
                    kv_by_rank[rank_key] = partitions[mapping.tp_rank]
                components = _components_for_binding(
                    capability=domain.capability,
                    router_location=cluster_config.af.router_location,
                    tp_rank=mapping.tp_rank,
                    tp_size=tp_size,
                    dp_rank=mapping.dp_rank,
                    is_leader=(mapping.tp_rank == 0),
                    expert_ids=expert_ids,
                    layers=layers,
                    hidden=hidden,
                    intermediate_source=intermediate_source,
                    vocab_size=vocab_size,
                    weight_dtype_size=weight_bytes,
                    kv_partition=kv_by_rank.get(rank_key),
                    expert_tp_size=expert_view.expert_tensor_parallel_size,
                    expert_tp_rank=(
                        mapping.tp_rank if expert_view.mode == "expert_tp" else 0
                    ),
                )
                for component in components:
                    previous = component_by_id.get(component.component_id)
                    if previous is not None and previous != component:
                        raise ConfigurationError(
                            f"AF component {component.component_id!r} has inconsistent "
                            f"metadata: {previous} != {component}"
                        )
                    component_by_id[component.component_id] = component
                preliminary.append(
                    {
                        "domain": domain,
                        "instance": instance.name,
                        "mapping": mapping,
                        "worker": worker,
                        "tp_group": tp_group,
                        "ep_group": ep_group,
                        "expert_ids": expert_ids,
                        "components": components,
                    }
                )

    physical_components: dict[str, dict[str, AFModelComponent]] = {}
    physical_bindings: dict[str, list[RankKey]] = {}
    physical_kv_bytes: dict[str, list[int]] = {}
    for item in preliminary:
        domain = item["domain"]
        mapping = item["mapping"]
        rank_key = (domain.name, mapping.domain_rank)
        physical = mapping.physical_worker.identity
        physical_bindings.setdefault(physical, []).append(rank_key)
        ledger = physical_components.setdefault(physical, {})
        for component in item["components"]:
            ledger.setdefault(component.component_id, component)
        partition = kv_by_rank.get(rank_key)
        if partition is not None:
            physical_kv_bytes.setdefault(physical, []).append(partition.bytes_per_block)

    physical_states: list[AFPhysicalRuntimeState] = []
    for physical, rank_keys in sorted(physical_bindings.items()):
        worker = worker_by_physical[physical]
        try:
            hardware_conf = hardware.device(worker.hardware)
            capacity_bytes = int(hardware_conf.usable_memory_bytes)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"AF runtime hardware capacity metadata missing for "
                f"worker={physical!r}, hardware={worker.hardware!r}"
            ) from exc
        components = tuple(
            physical_components[physical][component_id]
            for component_id in sorted(physical_components[physical])
        )
        model_bytes = sum(component.bytes for component in components)
        if model_bytes >= capacity_bytes:
            domains_and_ranks = ",".join(
                f"{domain}:{rank}" for domain, rank in rank_keys
            )
            component_bytes: dict[str, int] = {}
            for component in components:
                component_bytes[component.category] = (
                    component_bytes.get(component.category, 0) + component.bytes
                )
            raise ConfigurationError(
                f"AF model does not fit before execution: domain/rank={domains_and_ranks}, "
                f"physical_worker={physical}, hardware={worker.hardware}, "
                f"component_bytes={component_bytes}, model_bytes={model_bytes}, "
                f"capacity_bytes={capacity_bytes}, required_bytes={model_bytes}, "
                f"available_bytes={capacity_bytes}"
            )
        physical_states.append(
            AFPhysicalRuntimeState(
                physical_identity=physical,
                hardware=worker.hardware,
                capacity_bytes=capacity_bytes,
                bindings=tuple(sorted(rank_keys)),
                components=components,
                model_bytes=model_bytes,
                kv_bytes_per_block=max(physical_kv_bytes.get(physical, [0])),
            )
        )
    state_by_physical = {state.physical_identity: state for state in physical_states}

    bindings = tuple(
        AFRankBinding(
            domain=item["domain"].name,
            capability=item["domain"].capability,
            instance=item["instance"],
            domain_rank=item["mapping"].domain_rank,
            tp_rank=item["mapping"].tp_rank,
            pp_rank=item["mapping"].pp_rank,
            dp_rank=item["mapping"].dp_rank,
            ep_rank=item["mapping"].ep_rank,
            physical_worker=item["mapping"].physical_worker,
            hardware=item["worker"].hardware,
            tp_group=item["tp_group"],
            ep_group=item["ep_group"],
            is_scheduler_leader=(
                item["domain"].capability == "af_attention"
                and item["mapping"].tp_rank == 0
            ),
            expert_ids=item["expert_ids"],
            component_ids=tuple(
                component.component_id for component in item["components"]
            ),
            kv_partition=kv_by_rank.get(
                (item["domain"].name, item["mapping"].domain_rank)
            ),
        )
        for item in preliminary
    )
    binding_by_key = {binding.rank_key: binding for binding in bindings}

    replicas: list[AFReplicaRuntime] = []
    for domain in domain_configs:
        for instance in domain.logical_instances:
            instance_bindings = tuple(
                sorted(
                    (
                        binding_by_key[(domain.name, mapping.domain_rank)]
                        for mapping in instance.rank_mapping
                    ),
                    key=lambda binding: binding.tp_rank,
                )
            )
            local_gpu: list[tuple[RankKey, int]] = []
            local_cpu: list[tuple[RankKey, int]] = []
            for binding in instance_bindings:
                if binding.kv_partition is None:
                    gpu_blocks = cpu_blocks = 0
                else:
                    state = state_by_physical[binding.physical_identity]
                    raw_gpu_blocks = (
                        state.capacity_bytes - state.model_bytes
                    ) // binding.kv_partition.bytes_per_block
                    gpu_blocks = max(
                        0, raw_gpu_blocks - int(watermark * raw_gpu_blocks)
                    )
                    cpu_blocks = int(
                        min(
                            32768,
                            32768 * _GB // binding.kv_partition.bytes_per_block,
                        )
                    )
                local_gpu.append((binding.rank_key, gpu_blocks))
                local_cpu.append((binding.rank_key, cpu_blocks))
            leader = next(
                (
                    binding.rank_key
                    for binding in instance_bindings
                    if binding.is_scheduler_leader
                ),
                None,
            )
            replicas.append(
                AFReplicaRuntime(
                    domain=domain.name,
                    capability=domain.capability,
                    instance=instance.name,
                    dp_rank=instance_bindings[0].dp_rank,
                    rank_keys=tuple(binding.rank_key for binding in instance_bindings),
                    leader_rank_key=leader,
                    local_gpu_blocks=tuple(local_gpu),
                    local_cpu_blocks=tuple(local_cpu),
                    logical_gpu_blocks=(
                        min(value for _, value in local_gpu)
                        if domain.capability == "af_attention"
                        else 0
                    ),
                    logical_cpu_blocks=(
                        min(value for _, value in local_cpu)
                        if domain.capability == "af_attention"
                        else 0
                    ),
                )
            )

    tensor_ownerships = tuple(
        AFTensorOwnership(
            domain=binding.domain,
            domain_rank=binding.domain_rank,
            dp_rank=binding.dp_rank,
            tp_rank=binding.tp_rank,
            physical_worker=binding.physical_identity,
            interval=_tensor_interval(
                hidden,
                next(
                    domain.tensor_parallel_size
                    for domain in domains
                    if domain.name == binding.domain
                ),
                binding.tp_rank,
            ),
            layout="sharded",
            canonical_producer=(binding.tp_rank == 0),
            retains_residual=(binding.capability == "af_attention"),
        )
        for binding in bindings
    )
    request_owners = _compile_request_owners(cluster_config, bindings)
    partial_destinations = _compile_partial_destinations(
        cluster_config, tensor_ownerships
    )
    collective_plans = _compile_collective_plans(
        cluster_config, binding_by_key
    )

    plan = AFRuntimePlan(
        mla=mla,
        schema_version="2.0",
        model=model_id,
        block_size=block_size,
        effective_dtypes=effective_dtypes,
        total_layers=total_layers,
        hidden_size=hidden,
        num_attention_heads=attention_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        num_experts=(moe_config.num_experts if moe_config.enabled else 0),
        domains=domains,
        rank_bindings=tuple(
            sorted(bindings, key=lambda binding: (binding.domain, binding.domain_rank))
        ),
        replicas=tuple(
            sorted(replicas, key=lambda replica: (replica.domain, replica.dp_rank))
        ),
        physical_states=tuple(physical_states),
        layers=layers,
        request_owners=request_owners,
        tensor_ownerships=tensor_ownerships,
        partial_destinations=partial_destinations,
        collective_plans=collective_plans,
        link_effective_dtypes=link_dtypes,
    )
    plan.validate()
    return plan


def _validate_parallel_shapes(
    *,
    cluster_config: ClusterConfig,
    hidden: int,
    attention_heads: int,
    kv_heads: int,
    head_dim: int,
    mla_cache_width: int,
    intermediate: int,
    moe_intermediate: int,
    shared_intermediate: int,
    has_dense: bool,
    has_moe: bool,
) -> None:
    for domain in cluster_config.af.execution_domains:
        if domain.capability not in {"af_attention", "af_ffn"}:
            continue
        tp_size = domain.parallel.tensor_parallel_size
        if hidden % tp_size:
            raise ConfigurationError(
                f"AF runtime hidden_size={hidden} is not divisible by "
                f"domain={domain.name!r} TP={tp_size}; uneven hidden shards are "
                "not implemented"
            )
        if domain.capability == "af_attention":
            if attention_heads % tp_size:
                raise ConfigurationError(
                    f"AF runtime query heads={attention_heads} are not divisible by "
                    f"A domain={domain.name!r} TP={tp_size}"
                )
            kv_head_partitions(
                num_kv_heads=kv_heads,
                tensor_parallel_size=tp_size,
                head_dim=head_dim,
                mla_cache_width=mla_cache_width,
                num_layers=1,
                block_size=1,
                dtype_size_bytes=1,
            )
            continue
        if has_dense and intermediate % tp_size:
            raise ConfigurationError(
                f"AF runtime Dense FFN intermediate={intermediate} is not divisible "
                f"by F domain={domain.name!r} TP={tp_size}"
            )
        if shared_intermediate and shared_intermediate % tp_size:
            raise ConfigurationError(
                f"AF runtime Shared FFN intermediate={shared_intermediate} is not "
                f"divisible by F domain={domain.name!r} TP={tp_size}"
            )
        if (
            has_moe
            and domain.expert_parallel.expert_tensor_parallel_size > 1
            and moe_intermediate
            % domain.expert_parallel.expert_tensor_parallel_size
        ):
            raise ConfigurationError(
                f"AF runtime Routed Expert intermediate={moe_intermediate} is not "
                f"divisible by Expert TP="
                f"{domain.expert_parallel.expert_tensor_parallel_size} in "
                f"domain={domain.name!r}"
            )


def _tensor_interval(hidden: int, tp_size: int, tp_rank: int) -> AFTensorInterval:
    if hidden % tp_size:
        raise ConfigurationError(
            f"AF hidden ownership requires hidden={hidden} divisible by TP={tp_size}"
        )
    width = hidden // tp_size
    return AFTensorInterval(tp_rank * width, (tp_rank + 1) * width)


def _compile_request_owners(
    cluster_config: ClusterConfig,
    bindings: tuple[AFRankBinding, ...],
) -> tuple[AFRequestOwner, ...]:
    by_key = {binding.rank_key: binding for binding in bindings}
    owners: list[AFRequestOwner] = []
    for phase, (a_domain, _) in sorted(cluster_config.phase_to_af_domains.items()):
        for instance in sorted(
            a_domain.logical_instances,
            key=lambda item: min(mapping.dp_rank for mapping in item.rank_mapping),
        ):
            leader_mapping = min(instance.rank_mapping, key=lambda item: item.tp_rank)
            leader = by_key[(a_domain.name, leader_mapping.domain_rank)]
            owners.append(
                AFRequestOwner(
                    phase=phase,
                    domain=a_domain.name,
                    instance=instance.name,
                    dp_rank=leader.dp_rank,
                    leader_rank=leader.domain_rank,
                    physical_worker=leader.physical_identity,
                )
            )
    return tuple(owners)


def _compile_partial_destinations(
    cluster_config: ClusterConfig,
    ownerships: tuple[AFTensorOwnership, ...],
) -> tuple[AFPartialDestination, ...]:
    destinations: list[AFPartialDestination] = []
    for phase, (a_domain, f_domain) in sorted(
        cluster_config.phase_to_af_domains.items()
    ):
        a_owners = tuple(item for item in ownerships if item.domain == a_domain.name)
        f_owners = tuple(item for item in ownerships if item.domain == f_domain.name)
        for source in f_owners:
            for target in a_owners:
                interval = source.interval.intersection(target.interval)
                if interval is None:
                    continue
                destinations.append(
                    AFPartialDestination(
                        phase=phase,
                        source_domain=source.domain,
                        source_rank=source.domain_rank,
                        source_dp_rank=source.dp_rank,
                        target_domain=target.domain,
                        target_rank=target.domain_rank,
                        target_dp_rank=target.dp_rank,
                        interval=interval,
                    )
                )
    return tuple(
        sorted(
            destinations,
            key=lambda item: (
                item.phase,
                item.source_domain,
                item.source_rank,
                item.target_domain,
                item.target_rank,
                item.interval.start,
            ),
        )
    )


def _compile_collective_plans(
    cluster_config: ClusterConfig,
    binding_by_key: Mapping[RankKey, AFRankBinding],
) -> tuple[AFResolvedCollectivePlan, ...]:
    resources = cluster_config.af_communication_resources
    plans: list[AFResolvedCollectivePlan] = []
    for domain in cluster_config.af.execution_domains:
        if domain.capability not in {"af_attention", "af_ffn"}:
            continue
        for replica in domain.logical_instances:
            dp_ranks = {mapping.dp_rank for mapping in replica.rank_mapping}
            if len(dp_ranks) != 1:
                raise ConfigurationError(
                    f"AF runtime logical instance {replica.name!r} spans DP replicas"
                )
            dp_rank = next(iter(dp_ranks))
            for collective in domain.communication_plan:
                resource_group = collective.resource_group
                resource_lane = collective.resource_lane
                network = collective.network
                # Older cluster files omitted collective resource metadata.
                # Resolve it from the AF link while retaining strict checks
                # for references that name a non-existent lane.
                if resource_group is None or resource_lane is None or network is None:
                    candidates = [
                        link for link in cluster_config.af.links
                        if (link.attention_domain == domain.name
                            or link.ffn_domain == domain.name
                            or (link.attention_domain is None and link.ffn_domain is None
                                and link.attention_group == domain.worker_group
                                and ((domain.capability == "af_attention")
                                     or link.ffn_group == domain.worker_group)))
                        and bool(set(link.phases) & set(domain.phases))
                    ]
                    if candidates:
                        link = candidates[0]
                        resource_group = resource_group or link.resource_group
                        resource_lane = resource_lane or (
                            "shared" if link.duplex == "half" else
                            "a_to_f" if domain.capability == "af_attention" else "f_to_a"
                        )
                        network = network or link.network
                if resource_group is None or resource_lane is None:
                    raise ConfigurationError(
                        f"AF collective {collective.operation!r} in domain "
                        f"{domain.name!r} requires resource_group and resource_lane"
                    )
                key = (resource_group, resource_lane)
                resource = resources.get(key)
                if resource is None:
                    raise ConfigurationError(
                        f"AF collective {collective.operation!r} in domain "
                        f"{domain.name!r} references missing communication lane {key!r}"
                    )
                participant_ranks = domain.resolve_collective_participants(
                    collective, dp_rank
                )
                participant_bindings = tuple(
                    binding_by_key[(domain.name, rank)] for rank in participant_ranks
                )
                plans.append(
                    AFResolvedCollectivePlan(
                        domain=domain.name,
                        dp_rank=dp_rank,
                        after_stage=collective.after_stage,
                        operation=collective.operation,
                        participant_ranks=participant_ranks,
                        physical_workers=tuple(
                            binding.physical_identity
                            for binding in participant_bindings
                        ),
                        network=network or resource.network,
                        resource_group=resource_group,
                        resource_lane=resource_lane,
                        payload_bytes=collective.payload_bytes,
                    )
                )
    return tuple(
        sorted(
            plans,
            key=lambda item: (item.domain, item.dp_rank, item.after_stage),
        )
    )


def _components_for_binding(
    *,
    capability: str,
    router_location: str,
    tp_rank: int,
    tp_size: int,
    dp_rank: int,
    is_leader: bool,
    expert_ids: tuple[int, ...],
    layers: tuple[AFLayerRuntime, ...],
    hidden: int,
    intermediate_source: str,
    vocab_size: int,
    weight_dtype_size: int,
    kv_partition: AFKVHeadPartition | None,
    expert_tp_size: int,
    expert_tp_rank: int,
) -> tuple[AFModelComponent, ...]:
    _ = dp_rank
    components: list[AFModelComponent] = []
    if is_leader and capability == ("af_attention" if router_location == "a" else "af_ffn"):
        for layer in layers:
            if layer.layer_type != "moe":
                continue
            components.append(
                AFModelComponent(
                    component_id=f"layer.{layer.layer_id}.router.leader",
                    category="router",
                    bytes=hidden * layer.num_experts * weight_dtype_size,
                    formula=f"hidden * experts * weight_dtype ({router_location.upper()} leader only)",
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("num_experts", layer.num_experts),
                    ),
                )
            )
    if capability == "af_attention":
        if kv_partition is None:
            raise ConfigurationError("A binding requires a KV-head partition")
        q_heads = layers[0].num_attention_heads // tp_size
        q_width = q_heads * layers[0].head_dim
        kv_width = kv_partition.local_head_count * layers[0].head_dim
        attention_elements = 2 * hidden * q_width + 2 * hidden * kv_width
        mla = layers[0].mla
        if mla:
            attention_elements = mla.weight_elements(q_heads, hidden)
        for layer in layers:
            components.append(
                AFModelComponent(
                    component_id=f"layer.{layer.layer_id}.attention.tp{tp_rank}-of-{tp_size}",
                    category="attention",
                    bytes=attention_elements * weight_dtype_size,
                    formula=("MLA replicated Q/KV down + head-sharded Q/KV up and O + latent norms" if mla else
                        "H*(Q_local+2*KV_local)+Q_local*H, using exact Q/K/V/O "
                        "dimensions"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("q_heads_local", q_heads),
                        ("kv_heads_local", kv_partition.local_head_count),
                        ("head_dim", layer.head_dim),
                    ),
                )
            )
        components.append(
            AFModelComponent(
                component_id=f"embedding_lm_head.{'tied' if layers[0].tie_word_embeddings else 'untied'}.tp{tp_rank}-of-{tp_size}",
                category="embedding_lm_head",
                bytes=(vocab_size * hidden // tp_size) * weight_dtype_size * (1 if layers[0].tie_word_embeddings else 2),
                formula="vocab * hidden * weight_dtype / A_TP * (1 if tied else 2)",
                shape=(
                    ("vocab_size", vocab_size),
                    ("hidden_size", hidden),
                    ("tp_size", tp_size),
                ),
            )
        )
        return tuple(components)

    if capability != "af_ffn":
        raise ConfigurationError(f"unsupported AF component capability {capability!r}")
    for layer in layers:
        if layer.layer_type == "dense":
            local_intermediate = layer.dense_intermediate_size // tp_size
            components.append(
                AFModelComponent(
                    component_id=(
                        f"layer.{layer.layer_id}.dense_ffn.tp{tp_rank}-of-{tp_size}"
                    ),
                    category="dense_ffn",
                    bytes=3 * hidden * local_intermediate * weight_dtype_size,
                    formula=(
                        "3 * hidden * local_intermediate * weight_dtype (SwiGLU "
                        f"gate/up/down); intermediate_source={intermediate_source}"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("intermediate_size", layer.dense_intermediate_size),
                        ("local_intermediate_size", local_intermediate),
                    ),
                )
            )
            continue
        local_expert_intermediate = (
            layer.routed_intermediate_size // expert_tp_size
        )
        for expert_id in expert_ids:
            components.append(
                AFModelComponent(
                    component_id=(
                        f"layer.{layer.layer_id}.routed_expert.{expert_id}."
                        f"tp{expert_tp_rank}-of-{expert_tp_size}"
                    ),
                    category="routed_expert",
                    bytes=(
                        3
                        * hidden
                        * local_expert_intermediate
                        * weight_dtype_size
                    ),
                    formula=(
                        "3 * hidden * local_expert_intermediate * weight_dtype "
                        "(SwiGLU gate/up/down)"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("intermediate_size", layer.routed_intermediate_size),
                        ("local_intermediate_size", local_expert_intermediate),
                        ("expert_id", expert_id),
                    ),
                )
            )
        if layer.shared_intermediate_size:
            local_shared_intermediate = layer.shared_intermediate_size // tp_size
            components.append(
                AFModelComponent(
                    component_id=(
                        f"layer.{layer.layer_id}.shared_expert.tp{tp_rank}-of-{tp_size}"
                    ),
                    category="shared_expert",
                    bytes=(
                        3
                        * hidden
                        * local_shared_intermediate
                        * weight_dtype_size
                    ),
                    formula=(
                        "3 * hidden * local_shared_intermediate * weight_dtype "
                        "for fused shared experts"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("intermediate_size", layer.shared_intermediate_size),
                        ("local_intermediate_size", local_shared_intermediate),
                        ("num_shared_experts", layer.num_shared_experts),
                    ),
                )
            )
    return tuple(components)


def _experts_by_ep(
    num_experts: int, ep_size: int, strategy: str
) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {rank: [] for rank in range(ep_size)}
    if num_experts <= 0:
        return {rank: () for rank in range(ep_size)}
    if strategy == "round_robin":
        for expert_id in range(num_experts):
            result[expert_id % ep_size].append(expert_id)
    elif strategy == "linear":
        base, remainder = divmod(num_experts, ep_size)
        cursor = 0
        for rank in range(ep_size):
            count = base + (1 if rank < remainder else 0)
            result[rank].extend(range(cursor, cursor + count))
            cursor += count
    else:
        raise ConfigurationError(f"unsupported expert placement strategy {strategy!r}")
    return {rank: tuple(experts) for rank, experts in result.items()}


def _positive_model_int(model_conf: Any, name: str) -> int:
    value = getattr(model_conf, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"AF runtime model metadata {name} is missing")
    value = int(value)
    if value < 1:
        raise ConfigurationError(f"AF runtime model metadata {name} must be positive")
    return value


def _num_kv_heads(model_conf: Any) -> int:
    if _as_bool(getattr(model_conf, "Multi_Query", False)):
        return 1
    if _as_bool(getattr(model_conf, "Grouped_Query", False)):
        grouped = int(getattr(model_conf, "Grouped_Num", 0))
        attention_heads = _positive_model_int(model_conf, "Nhead")
        if grouped < 1 or attention_heads % grouped:
            raise ConfigurationError(
                f"AF runtime invalid grouped-query metadata: Nhead={attention_heads}, "
                f"Grouped_Num={grouped}"
            )
        return attention_heads // grouped
    return _positive_model_int(model_conf, "Nhead")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _intermediate_size(
    model_conf: Any, moe_config: MoEModelConfig, hidden: int
) -> tuple[int, str]:
    if moe_config.intermediate_size is not None:
        return int(moe_config.intermediate_size), "moe_config.intermediate_size"
    model_value = getattr(model_conf, "FFN_Hidden", None)
    if model_value is not None:
        value = int(model_value)
        if value < 1:
            raise ConfigurationError("AF runtime FFN_Hidden must be positive")
        return value, "model.FFN_Hidden"
    return 4 * hidden, "analytical_default_4x_hidden"
