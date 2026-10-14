"""Planning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from TokenSim.af.caching.snapshots import ConfigurationOwner, snapshot
from TokenSim.af.planning.batches import AFBlock, AFBlockPlacement, AFExpertPlacement
from TokenSim.af.planning.compute import ComputeNodePlannerMixin
from TokenSim.af.planning.copies import TensorCopyPlannerMixin
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.grouping import group_routed_expert_nodes
from TokenSim.af.planning.layers import RuntimeLayerPlannerMixin
from TokenSim.af.planning.legacy import LegacyLayerPlannerMixin
from TokenSim.af.planning.moe import MoELayerPlannerMixin
from TokenSim.af.planning.native import NativeLayerPlannerMixin
from TokenSim.af.planning.nodes import (
    ExecutionNode,
    ExecutionStage,
    NodeKind,
    RankLocalWork,
)
from TokenSim.af.planning.replicas import (
    AFReplicaSelection,
    GlobalRoundRobinReplicaSelector,
    IndependentRoundRobinAFReplicaSelector,
)
from TokenSim.af.planning.router import RouterLogitsMixin
from TokenSim.af.planning.transfer import TransferStagePlannerMixin
from TokenSim.config.af.domains import (
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
)
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


if TYPE_CHECKING:
    from TokenSim.af.runtime.plan import AFRuntimePlan


class ParallelExecutionPlanner(
    LegacyLayerPlannerMixin,
    RuntimeLayerPlannerMixin,
    MoELayerPlannerMixin,
    NativeLayerPlannerMixin,
    ComputeNodePlannerMixin,
    TensorCopyPlannerMixin,
    TransferStagePlannerMixin,
    RouterLogitsMixin,
    ConfigurationOwner,
):
    _configuration_fields = (
        "cluster_config",
        "runtime_plan",
        "hidden_size",
        "dtype_size_bytes",
        "num_experts",
        "top_k",
    )

    def __init__(
        self,
        cluster_config,
        *,
        hidden_size: int,
        dtype_size_bytes: int = 2,
        num_experts: int = 0,
        top_k: int = 1,
        runtime_plan: "AFRuntimePlan | None" = None,
    ) -> None:
        self.cluster_config = snapshot(cluster_config)
        self.hidden_size = hidden_size
        self.dtype_size_bytes = dtype_size_bytes
        self.num_experts = num_experts
        self.top_k = max(1, top_k)
        self.runtime_plan = runtime_plan
        from TokenSim.af.caching.templates import TemplateCache

        self.template_cache = TemplateCache()
        self._runtime_links = {link.name: link for link in self.cluster_config.af.links}

    def placement(
        self,
        phase: str,
        source_worker_identity: str,
        f_instance: LogicalInstanceConfig,
    ) -> AFBlockPlacement:
        a_domain, f_domain = self.cluster_config.phase_to_af_domains[phase]
        if self.runtime_plan is None:
            a_instance = self.instance_for_physical(a_domain, source_worker_identity)
        else:
            a_instance = self._instance(
                a_domain,
                self.runtime_plan.instance_for_physical(
                    a_domain.name, source_worker_identity
                ),
            )
        return AFBlockPlacement(
            a_domain=a_domain.name,
            a_instance=a_instance.name,
            a_dp_rank=self.instance_dp_rank(a_instance),
            f_domain=f_domain.name,
            f_instance=f_instance.name,
            f_dp_rank=self.instance_dp_rank(f_instance),
            expert_placement=self._expert_placement_view(f_domain, f_instance),
        )

    def build_layer(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
    ) -> AFExecutionDAG:
        configured = self._runtime_links[link.name]
        if link is not configured:
            from TokenSim.af.caching.estimates import identity

            if identity(link) != identity(configured):
                raise ConfigurationError(
                    "AF link configuration changed; create a new runtime"
                )
            link = configured
        formal_runtime = self.runtime_plan is not None and block.placement is not None
        if formal_runtime:
            if self.cluster_config.af.dag_template_cache:
                return self.template_cache.build(self, block, layer_id, link)
            return self._build_runtime_layer(block, layer_id, link)
        return self._build_legacy_layer(block, layer_id, link)

    def _ep_workloads(
        self,
        block: AFBlock,
        domain: ExecutionDomainConfig,
        instance: LogicalInstanceConfig,
    ) -> tuple[dict[int, int], dict[int, tuple[tuple[int, int], ...]]]:
        ep_size = domain.parallel.expert_parallel_size
        loads = {rank: 0 for rank in range(ep_size)}
        expert_loads: dict[int, list[tuple[int, int]]] = {
            rank: [] for rank in range(ep_size)
        }
        histogram = dict(block.expert_histogram)
        placement = block.placement.expert_placement if block.placement else ()
        if not placement:
            placement = self._expert_placement_view(domain, instance)
        by_expert = {item.expert_id: item for item in placement}
        if not histogram:
            if self.num_experts <= 0:
                loads[0] = block.token_count
                return loads, {rank: () for rank in range(ep_size)}
            total = block.token_count * self.top_k
            for index in range(total):
                expert_id = index % self.num_experts
                histogram[expert_id] = histogram.get(expert_id, 0) + 1
        for expert_id, count in histogram.items():
            try:
                ep_rank = by_expert[expert_id].ep_rank
            except KeyError as exc:
                raise ConfigurationError(
                    f"expert id {expert_id} has no legal placement in F domain "
                    f"{domain.name!r} DP replica {instance.name!r}"
                ) from exc
            loads[ep_rank] += count
            expert_loads[ep_rank].append((expert_id, count))
        return loads, {rank: tuple(values) for rank, values in expert_loads.items()}

    def _expert_placement_view(
        self, domain: ExecutionDomainConfig, instance: LogicalInstanceConfig
    ) -> tuple[AFExpertPlacement, ...]:
        if self.runtime_plan is not None:
            return tuple(
                AFExpertPlacement(
                    expert_id=expert_id,
                    ep_rank=ep_rank,
                    domain_ranks=domain_ranks,
                    physical_workers=physical_workers,
                )
                for (
                    expert_id,
                    ep_rank,
                    domain_ranks,
                    physical_workers,
                ) in self.runtime_plan.expert_placement(domain.name, instance.name)
            )
        if self.num_experts <= 0:
            return ()
        source_mappings = self._routed_runtime_mappings(domain, instance)
        mappings_by_ep = {
            ep_rank: tuple(
                sorted(
                    (
                        mapping
                        for mapping in source_mappings
                        if mapping.ep_rank == ep_rank
                    ),
                    key=lambda mapping: mapping.tp_rank,
                )
            )
            for ep_rank in range(domain.parallel.expert_parallel_size)
        }
        placements = []
        for expert_id in range(self.num_experts):
            ep_rank = self._ep_rank_for_expert(
                expert_id, domain.parallel.expert_parallel_size, domain
            )
            mappings = mappings_by_ep.get(ep_rank, ())
            if not mappings:
                raise ConfigurationError(
                    f"expert id {expert_id} has no legal rank in F domain "
                    f"{domain.name!r} DP replica {instance.name!r} at ep_rank={ep_rank}"
                )
            placements.append(
                AFExpertPlacement(
                    expert_id=expert_id,
                    ep_rank=ep_rank,
                    domain_ranks=tuple(mapping.domain_rank for mapping in mappings),
                    physical_workers=tuple(
                        mapping.physical_worker.identity for mapping in mappings
                    ),
                )
            )
        return tuple(placements)

    def _routed_runtime_mappings(
        self,
        domain: ExecutionDomainConfig,
        instance: LogicalInstanceConfig,
    ) -> tuple[DomainRankMapping, ...]:
        if domain.expert_parallel.mode != "rank_reuse":
            return self._runtime_mappings(domain, instance)
        return tuple(
            mapping
            for candidate in domain.logical_instances
            for mapping in self._runtime_mappings(domain, candidate)
        )

    @staticmethod
    def _instance_for_mapping(
        domain: ExecutionDomainConfig,
        mapping: DomainRankMapping,
    ) -> LogicalInstanceConfig:
        for instance in domain.logical_instances:
            if any(
                candidate.domain_rank == mapping.domain_rank
                for candidate in instance.rank_mapping
            ):
                return instance
        raise ConfigurationError(
            f"execution domain {domain.name!r} has no DP replica for "
            f"domain rank {mapping.domain_rank}"
        )

    def _ep_rank_for_expert(
        self, expert_id: int, ep_size: int, domain: ExecutionDomainConfig
    ) -> int:
        if expert_id < 0 or (self.num_experts and expert_id >= self.num_experts):
            raise ConfigurationError(
                f"expert id {expert_id} has no legal placement in F domain "
                f"{domain.name!r}"
            )
        if domain.parallel.expert_placement_strategy == "round_robin":
            return expert_id % ep_size
        if self.num_experts <= 0:
            return expert_id % ep_size
        base, remainder = divmod(self.num_experts, ep_size)
        cursor = 0
        for ep_rank in range(ep_size):
            count = base + (1 if ep_rank < remainder else 0)
            if cursor <= expert_id < cursor + count:
                return ep_rank
            cursor += count
        raise ConfigurationError(
            f"expert id {expert_id} has no legal placement in F domain {domain.name!r}"
        )

    @staticmethod
    def _instance(domain: ExecutionDomainConfig, name: str) -> LogicalInstanceConfig:
        for instance in domain.logical_instances:
            if instance.name == name:
                return instance
        raise ConfigurationError(
            f"execution domain {domain.name!r} has no DP replica {name!r}"
        )

    def _runtime_mappings(
        self,
        domain: ExecutionDomainConfig,
        instance: LogicalInstanceConfig,
    ) -> tuple[DomainRankMapping, ...]:
        if self.runtime_plan is None:
            return tuple(instance.rank_mapping)
        return tuple(
            DomainRankMapping(
                domain_rank=binding.domain_rank,
                tp_rank=binding.tp_rank,
                pp_rank=binding.pp_rank,
                dp_rank=binding.dp_rank,
                ep_rank=binding.ep_rank,
                physical_worker=binding.physical_worker,
            )
            for binding in self.runtime_plan.bindings_for_instance(
                domain.name, instance.name
            )
        )

    @staticmethod
    def instance_for_physical(
        domain: ExecutionDomainConfig, physical_identity: str
    ) -> LogicalInstanceConfig:
        for instance in domain.logical_instances:
            if any(
                mapping.physical_worker.identity == physical_identity
                for mapping in instance.rank_mapping
            ):
                return instance
        raise ConfigurationError(
            f"execution domain {domain.name!r} has no logical instance mapped to "
            f"physical worker {physical_identity!r}"
        )

    @staticmethod
    def instance_dp_rank(instance: LogicalInstanceConfig) -> int:
        return GlobalRoundRobinReplicaSelector._instance_dp_rank(instance)
