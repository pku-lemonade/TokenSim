"""Runtime plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from TokenSim.af.runtime.geometry import RuntimeGeometryMixin
from TokenSim.af.runtime.records import (
    AFDomainRuntime,
    AFLayerRuntime,
    AFPartialDestination,
    AFPhysicalRuntimeState,
    AFRankBinding,
    AFReplicaRuntime,
    AFRequestOwner,
    AFResolvedCollectivePlan,
    AFTensorOwnership,
    ReplicaKey,
)
from TokenSim.af.runtime.validation import RuntimeValidationMixin
from TokenSim.config.af.links import AFEffectiveDTypes
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.mla import MLADimensions


@dataclass(frozen=True)
class AFRuntimePlan(RuntimeGeometryMixin, RuntimeValidationMixin):
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
        return dtype_size_bytes(self.effective_dtypes.weight_dtype, role="weight_dtype")

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
            tuple(binding for binding in self.rank_bindings if binding.domain == domain)
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
        if node.stage in {"expert", "routed_expert"} and not set(expert_ids).issubset(
            binding.expert_ids
        ):
            raise SimulationStateError(
                f"runtime node {node.node_id!r} requests experts {expert_ids} on "
                f"domain={binding.domain}, rank={binding.domain_rank}, "
                f"resident={binding.expert_ids}"
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
                    "expert_placement_strategy": (domain.expert_placement_strategy),
                    "base_world_size": (
                        domain.tensor_parallel_size * domain.data_parallel_size
                    ),
                    "expert_tensor_parallel_size": (domain.expert_tensor_parallel_size),
                }
                for domain in self.domains
            ],
            "rank_bindings": [binding.as_dict() for binding in self.rank_bindings],
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
