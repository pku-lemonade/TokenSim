from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from TokenSim.config.cluster_config import (
    ClusterConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    PhysicalWorkerRef,
    WorkerConfig,
)
from TokenSim.config.constants import _GB
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware.device import dtype_bytes


RankKey = tuple[str, int]
ReplicaKey = tuple[str, int]


@dataclass(frozen=True)
class AFModelComponent:
    component_id: str
    category: str
    bytes: int
    formula: str

    def as_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "category": self.category,
            "bytes": self.bytes,
            "formula": self.formula,
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
    dtype: str
    dtype_size_bytes: int
    total_layers: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    domains: tuple[AFDomainRuntime, ...]
    rank_bindings: tuple[AFRankBinding, ...]
    replicas: tuple[AFReplicaRuntime, ...]
    physical_states: tuple[AFPhysicalRuntimeState, ...]

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
        for binding in self.bindings_for_instance(domain, instance):
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
        if node.tp_group and tuple(node.tp_group) != binding.tp_group:
            raise SimulationStateError(
                f"runtime node {node.node_id!r} TP group mismatch for "
                f"domain={binding.domain}, rank={binding.domain_rank}: "
                f"expected={binding.tp_group}, actual={tuple(node.tp_group)}"
            )
        expert_ids = tuple(getattr(node.local_work, "expert_ids", ()) or ())
        if node.stage == "expert" and not set(expert_ids).issubset(binding.expert_ids):
            raise SimulationStateError(
                f"runtime node {node.node_id!r} requests experts {expert_ids} on "
                f"domain={binding.domain}, rank={binding.domain_rank}, "
                f"resident={binding.expert_ids}"
            )

    def validate(self) -> None:
        rank_keys = [binding.rank_key for binding in self.rank_bindings]
        if len(rank_keys) != len(set(rank_keys)):
            raise ConfigurationError("AF runtime plan contains duplicate rank bindings")
        physical_ids = {state.physical_identity for state in self.physical_states}
        if len(physical_ids) != len(self.physical_states):
            raise ConfigurationError(
                "AF runtime plan contains duplicate physical states"
            )
        state_by_physical = {
            state.physical_identity: state for state in self.physical_states
        }
        for binding in self.rank_bindings:
            if binding.physical_identity not in physical_ids:
                raise ConfigurationError(
                    f"AF runtime binding domain={binding.domain}, "
                    f"rank={binding.domain_rank} has no physical state for "
                    f"{binding.physical_identity!r}"
                )
            state = state_by_physical[binding.physical_identity]
            if binding.rank_key not in state.bindings:
                raise ConfigurationError(
                    f"AF runtime domain={binding.domain}, rank={binding.domain_rank} "
                    f"is missing from physical state {binding.physical_identity!r}"
                )
            component_ids = {item.component_id for item in state.components}
            if not set(binding.component_ids).issubset(component_ids):
                raise ConfigurationError(
                    f"AF runtime domain={binding.domain}, rank={binding.domain_rank} "
                    "component ownership differs from physical ledger"
                )
            categories = {
                item.category
                for item in state.components
                if item.component_id in binding.component_ids
            }
            allowed = (
                {"attention", "embedding_lm_head", "router"}
                if binding.capability == "af_attention"
                else {"dense_ffn", "routed_expert", "shared_expert"}
            )
            if not categories.issubset(allowed):
                raise ConfigurationError(
                    f"AF runtime domain={binding.domain}, rank={binding.domain_rank} "
                    f"has invalid component categories {sorted(categories - allowed)}"
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
                    f"mismatch: expected={expected_bindings}, actual={state.bindings}"
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
            bindings = [
                item for item in self.rank_bindings if item.domain == domain.name
            ]
            ep_size = (
                domain.expert_parallel_size if domain.capability == "af_ffn" else 1
            )
            expected_count = (
                domain.tensor_parallel_size * domain.data_parallel_size * ep_size
            )
            if len(bindings) != expected_count:
                raise ConfigurationError(
                    f"AF runtime domain={domain.name} rank count mismatch: "
                    f"expected={expected_count}, actual={len(bindings)}"
                )
        for replica in self.replicas:
            domain = domain_by_name.get(replica.domain)
            if domain is None:
                raise ConfigurationError(
                    f"AF runtime replica references missing domain {replica.domain!r}"
                )
            bindings = [self.binding(*rank_key) for rank_key in replica.rank_keys]
            if any(item.dp_rank != replica.dp_rank for item in bindings):
                raise ConfigurationError(
                    f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                    "contains a rank from another DP replica"
                )
            expected_coordinates = {
                (ep_rank, tp_rank)
                for ep_rank in range(
                    domain.expert_parallel_size if domain.capability == "af_ffn" else 1
                )
                for tp_rank in range(domain.tensor_parallel_size)
            }
            actual_coordinates = {(item.ep_rank, item.tp_rank) for item in bindings}
            if actual_coordinates != expected_coordinates:
                raise ConfigurationError(
                    f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                    f"rank coordinates mismatch: expected={sorted(expected_coordinates)}, "
                    f"actual={sorted(actual_coordinates)}"
                )
            if replica.is_attention:
                leaders = [item for item in bindings if item.is_scheduler_leader]
                if len(leaders) != 1 or leaders[0].rank_key != replica.leader_rank_key:
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        f"must have one scheduler leader; got "
                        f"{[item.domain_rank for item in leaders]}"
                    )
                expected_tp = tuple(item.domain_rank for item in bindings)
                for binding in bindings:
                    if binding.tp_group != expected_tp:
                        raise ConfigurationError(
                            f"AF runtime domain={binding.domain}, rank={binding.domain_rank} "
                            f"TP group mismatch: expected={expected_tp}, "
                            f"actual={binding.tp_group}"
                        )
                    if binding.kv_partition is None:
                        raise ConfigurationError(
                            f"AF runtime domain={binding.domain}, "
                            f"rank={binding.domain_rank} has no KV-head partition"
                        )
                expected_partitions = kv_head_partitions(
                    num_kv_heads=self.num_kv_heads,
                    tensor_parallel_size=domain.tensor_parallel_size,
                    head_dim=self.head_dim,
                    num_layers=self.total_layers,
                    block_size=self.block_size,
                    dtype_size_bytes=self.dtype_size_bytes,
                )
                for binding in bindings:
                    if binding.kv_partition != expected_partitions[binding.tp_rank]:
                        raise ConfigurationError(
                            f"AF runtime domain={binding.domain}, "
                            f"rank={binding.domain_rank} KV-head partition mismatch"
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
                    raise ConfigurationError(
                        f"F runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "must not own a scheduler leader"
                    )
                expert_to_ep: dict[int, set[int]] = {}
                expert_to_tp: dict[int, set[int]] = {}
                for binding in bindings:
                    if binding.kv_partition is not None:
                        raise ConfigurationError(
                            f"F runtime domain={binding.domain}, "
                            f"rank={binding.domain_rank} must not own KV state"
                        )
                    expected_tp_group = tuple(
                        item.domain_rank
                        for item in bindings
                        if item.ep_rank == binding.ep_rank
                    )
                    expected_ep_group = tuple(
                        item.domain_rank
                        for item in bindings
                        if item.tp_rank == binding.tp_rank
                    )
                    if binding.tp_group != expected_tp_group:
                        raise ConfigurationError(
                            f"F runtime domain={binding.domain}, "
                            f"rank={binding.domain_rank} TP group mismatch"
                        )
                    if binding.ep_group != expected_ep_group:
                        raise ConfigurationError(
                            f"F runtime domain={binding.domain}, "
                            f"rank={binding.domain_rank} EP group mismatch"
                        )
                    for expert_id in binding.expert_ids:
                        expert_to_ep.setdefault(expert_id, set()).add(binding.ep_rank)
                        expert_to_tp.setdefault(expert_id, set()).add(binding.tp_rank)
                if set(expert_to_ep) != set(range(self.num_experts)):
                    raise ConfigurationError(
                        f"F runtime domain={replica.domain}, replica={replica.dp_rank} "
                        f"expert placement mismatch: expected={list(range(self.num_experts))}, "
                        f"actual={sorted(expert_to_ep)}"
                    )
                for expert_id in expert_to_ep:
                    if len(expert_to_ep[expert_id]) != 1 or expert_to_tp[
                        expert_id
                    ] != set(range(domain.tensor_parallel_size)):
                        raise ConfigurationError(
                            f"F runtime domain={replica.domain}, "
                            f"replica={replica.dp_rank}, expert={expert_id} has "
                            "inconsistent EP placement or TP shards"
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
    model: ModelSpec,
    hardware: Any,
    block_size: int,
    dtype: str = "fp16",
    watermark: float = 0.01,
) -> AFRuntimePlan:
    """Bind execution domains to physical workers and size their model/KV state.

    ``model`` is the resolved :class:`ModelSpec` (MoE overrides already
    applied) and ``hardware`` the :class:`HardwareContext` whose device catalog
    provides each worker's usable memory.
    """
    if not cluster_config.af.enabled:
        raise ConfigurationError("AF runtime plan requires af.enabled=true")
    if not 0 <= watermark < 1:
        raise ConfigurationError("AF runtime watermark must be in [0, 1)")
    if not isinstance(model, ModelSpec):
        raise ConfigurationError(
            f"AF runtime plan requires a ModelSpec, got {type(model).__name__}"
        )
    dtype_size = _dtype_size(dtype)
    moe_config = model.moe
    hidden = model.hidden_size
    total_layers = model.num_layers
    head_dim = model.head_dim
    kv_heads = model.num_key_value_heads
    intermediate, intermediate_source = model.intermediate_size, "model.intermediate_size"
    moe_intermediate = model.moe_intermediate_size
    vocab_size = model.vocab_size
    moe_layers = len(model.moe_layer_indices())
    dense_layers = total_layers - moe_layers

    worker_by_physical: dict[str, WorkerConfig] = {}
    for worker in worker_configs:
        if worker.physical_worker_ref is not None:
            worker_by_physical[worker.physical_worker_ref.identity] = worker

    domains = tuple(
        AFDomainRuntime(
            name=domain.name,
            capability=domain.capability,
            tensor_parallel_size=domain.parallel.tensor_parallel_size,
            data_parallel_size=domain.parallel.data_parallel_size,
            expert_parallel_size=(
                domain.parallel.expert_parallel_size
                if domain.capability == "af_ffn"
                else 1
            ),
            expert_placement_strategy=domain.parallel.expert_placement_strategy,
        )
        for domain in cluster_config.af.execution_domains
        if domain.capability in {"af_attention", "af_ffn"}
    )

    component_by_id: dict[str, AFModelComponent] = {}
    components_by_rank: dict[RankKey, tuple[AFModelComponent, ...]] = {}
    preliminary: list[dict[str, Any]] = []
    kv_by_rank: dict[RankKey, AFKVHeadPartition] = {}
    for domain in cluster_config.af.execution_domains:
        if domain.capability not in {"af_attention", "af_ffn"}:
            continue
        tp_size = domain.parallel.tensor_parallel_size
        partitions = (
            kv_head_partitions(
                num_kv_heads=kv_heads,
                tensor_parallel_size=tp_size,
                head_dim=head_dim,
                num_layers=total_layers,
                block_size=block_size,
                dtype_size_bytes=dtype_size,
            )
            if domain.capability == "af_attention"
            else ()
        )
        for instance in domain.logical_instances:
            mappings = sorted(
                instance.rank_mapping,
                key=lambda mapping: (mapping.ep_rank, mapping.tp_rank),
            )
            experts_by_ep = _experts_by_ep(
                moe_config.num_experts if moe_config.enabled else 0,
                domain.parallel.expert_parallel_size,
                domain.parallel.expert_placement_strategy,
            )
            for mapping in mappings:
                physical = mapping.physical_worker.identity
                worker = worker_by_physical.get(physical)
                if worker is None:
                    raise ConfigurationError(
                        f"AF runtime domain={domain.name}, rank={mapping.domain_rank} "
                        f"references missing physical worker {physical!r}"
                    )
                tp_group = tuple(
                    item.domain_rank
                    for item in mappings
                    if item.ep_rank == mapping.ep_rank
                )
                ep_group = tuple(
                    item.domain_rank
                    for item in mappings
                    if item.tp_rank == mapping.tp_rank
                )
                expert_ids = (
                    tuple(experts_by_ep.get(mapping.ep_rank, ()))
                    if domain.capability == "af_ffn"
                    else ()
                )
                components = _components_for_binding(
                    capability=domain.capability,
                    tp_rank=mapping.tp_rank,
                    tp_size=tp_size,
                    is_leader=(mapping.tp_rank == 0 and mapping.ep_rank == 0),
                    expert_ids=expert_ids,
                    hidden=hidden,
                    total_layers=total_layers,
                    dense_layers=dense_layers,
                    moe_layers=moe_layers,
                    intermediate=intermediate,
                    intermediate_source=intermediate_source,
                    moe_intermediate=moe_intermediate,
                    num_experts=(moe_config.num_experts if moe_config.enabled else 0),
                    num_shared_experts=(
                        moe_config.num_shared_experts if moe_config.enabled else 0
                    ),
                    vocab_size=vocab_size,
                    dtype_size=dtype_size,
                )
                rank_key = (domain.name, mapping.domain_rank)
                for component in components:
                    previous = component_by_id.get(component.component_id)
                    if previous is not None and previous != component:
                        raise ConfigurationError(
                            f"AF component {component.component_id!r} has inconsistent "
                            f"metadata: {previous} != {component}"
                        )
                    component_by_id[component.component_id] = component
                components_by_rank[rank_key] = components
                if partitions:
                    kv_by_rank[rank_key] = partitions[mapping.tp_rank]
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
        mapping = item["mapping"]
        rank_key = (item["domain"].name, mapping.domain_rank)
        physical = mapping.physical_worker.identity
        physical_bindings.setdefault(physical, []).append(rank_key)
        ledger = physical_components.setdefault(physical, {})
        for component in item["components"]:
            ledger.setdefault(component.component_id, component)
        partition = kv_by_rank.get(rank_key)
        if partition is not None:
            physical_kv_bytes.setdefault(physical, []).append(partition.bytes_per_block)

    physical_states = []
    for physical, rank_keys in sorted(physical_bindings.items()):
        worker = worker_by_physical[physical]
        try:
            device = hardware.device(worker.hardware)
            capacity_bytes = int(device.usable_memory_bytes)
        except (AttributeError, ConfigurationError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"AF runtime hardware capacity metadata missing for "
                f"worker={physical!r}, hardware={worker.hardware!r}: {exc}"
            ) from exc
        components = tuple(
            physical_components.get(physical, {}).get(component_id)
            for component_id in sorted(physical_components.get(physical, {}))
        )
        components = tuple(
            component for component in components if component is not None
        )
        model_bytes = sum(component.bytes for component in components)
        if model_bytes >= capacity_bytes:
            domains_and_ranks = ",".join(
                f"{domain}:{rank}" for domain, rank in rank_keys
            )
            raise ConfigurationError(
                f"AF model does not fit before execution: domain/rank={domains_and_ranks}, "
                f"physical_worker={physical}, hardware={worker.hardware}, "
                f"model_bytes={model_bytes}, capacity_bytes={capacity_bytes}"
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

    bindings = []
    for item in preliminary:
        domain: ExecutionDomainConfig = item["domain"]
        mapping: DomainRankMapping = item["mapping"]
        rank_key = (domain.name, mapping.domain_rank)
        bindings.append(
            AFRankBinding(
                domain=domain.name,
                capability=domain.capability,
                instance=item["instance"],
                domain_rank=mapping.domain_rank,
                tp_rank=mapping.tp_rank,
                pp_rank=mapping.pp_rank,
                dp_rank=mapping.dp_rank,
                ep_rank=mapping.ep_rank,
                physical_worker=mapping.physical_worker,
                hardware=item["worker"].hardware,
                tp_group=item["tp_group"],
                ep_group=item["ep_group"],
                is_scheduler_leader=(
                    domain.capability == "af_attention"
                    and mapping.tp_rank == 0
                    and mapping.ep_rank == 0
                ),
                expert_ids=item["expert_ids"],
                component_ids=tuple(
                    component.component_id for component in item["components"]
                ),
                kv_partition=kv_by_rank.get(rank_key),
            )
        )
    binding_by_key = {binding.rank_key: binding for binding in bindings}

    replicas = []
    for domain in cluster_config.af.execution_domains:
        if domain.capability not in {"af_attention", "af_ffn"}:
            continue
        for instance in domain.logical_instances:
            instance_bindings = tuple(
                sorted(
                    (
                        binding_by_key[(domain.name, mapping.domain_rank)]
                        for mapping in instance.rank_mapping
                    ),
                    key=lambda binding: (binding.ep_rank, binding.tp_rank),
                )
            )
            local_gpu = []
            local_cpu = []
            for binding in instance_bindings:
                if binding.kv_partition is None:
                    gpu_blocks = 0
                    cpu_blocks = 0
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

    plan = AFRuntimePlan(
        schema_version="1.0",
        model=model.model_id,
        block_size=block_size,
        dtype=dtype,
        dtype_size_bytes=dtype_size,
        total_layers=total_layers,
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
    )
    plan.validate()
    return plan


def _components_for_binding(
    *,
    capability: str,
    tp_rank: int,
    tp_size: int,
    is_leader: bool,
    expert_ids: tuple[int, ...],
    hidden: int,
    total_layers: int,
    dense_layers: int,
    moe_layers: int,
    intermediate: int,
    intermediate_source: str,
    moe_intermediate: int,
    num_experts: int,
    num_shared_experts: int,
    vocab_size: int,
    dtype_size: int,
) -> tuple[AFModelComponent, ...]:
    components: list[AFModelComponent] = []
    if capability == "af_attention":
        components.append(
            _component(
                f"attention.layers.tp{tp_rank}-of-{tp_size}",
                "attention",
                4 * total_layers * hidden * hidden * dtype_size,
                tp_size,
                "4 * layers * hidden^2 * dtype / A_TP",
            )
        )
        components.append(
            _component(
                f"embedding_lm_head.tied.tp{tp_rank}-of-{tp_size}",
                "embedding_lm_head",
                vocab_size * hidden * dtype_size,
                tp_size,
                "vocab * hidden * dtype / A_TP (tied)",
            )
        )
        if is_leader and moe_layers and num_experts:
            components.append(
                AFModelComponent(
                    component_id="router.layers.leader",
                    category="router",
                    bytes=moe_layers * hidden * num_experts * dtype_size,
                    formula="moe_layers * hidden * experts * dtype (A leader only)",
                )
            )
        return tuple(components)

    if capability != "af_ffn":
        raise ConfigurationError(f"unsupported AF component capability {capability!r}")
    if dense_layers:
        components.append(
            _component(
                f"dense_ffn.layers.tp{tp_rank}-of-{tp_size}",
                "dense_ffn",
                2 * dense_layers * hidden * intermediate * dtype_size,
                tp_size,
                f"2 * dense_layers * hidden * intermediate * dtype / F_TP; "
                f"intermediate_source={intermediate_source}",
            )
        )
    for expert_id in expert_ids:
        components.append(
            _component(
                f"routed_expert.{expert_id}.tp{tp_rank}-of-{tp_size}",
                "routed_expert",
                2 * moe_layers * hidden * moe_intermediate * dtype_size,
                tp_size,
                "2 * moe_layers * hidden * moe_intermediate * dtype / F_TP",
            )
        )
    for shared_id in range(num_shared_experts):
        components.append(
            _component(
                f"shared_expert.{shared_id}.tp{tp_rank}-of-{tp_size}",
                "shared_expert",
                2 * moe_layers * hidden * moe_intermediate * dtype_size,
                tp_size,
                "2 * moe_layers * hidden * moe_intermediate * dtype / F_TP",
            )
        )
    return tuple(components)


def _component(
    component_id: str,
    category: str,
    unsharded_bytes: int,
    shard_count: int,
    formula: str,
) -> AFModelComponent:
    return AFModelComponent(
        component_id=component_id,
        category=category,
        bytes=int(math.ceil(unsharded_bytes / shard_count)),
        formula=formula,
    )


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


def _dtype_size(dtype: str) -> int:
    """Whole-byte size of the AF activation/KV dtype (sub-byte formats are rejected)."""
    if not isinstance(dtype, str) or not dtype.strip():
        raise ConfigurationError("AF runtime dtype must be non-empty")
    try:
        size = dtype_bytes(dtype)
    except Exception as exc:  # unknown alias
        raise ConfigurationError(
            f"AF runtime dtype {dtype!r} has no explicit byte size"
        ) from exc
    if size < 1 or int(size) != size:
        raise ConfigurationError(
            f"AF runtime dtype {dtype!r} is a sub-byte format; KV partitions need whole bytes"
        )
    return int(size)
