"""Planning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from TokenSim.af.caching.snapshots import ConfigurationOwner, snapshot
from TokenSim.af.planning.batches import AFBlock, AFBlockPlacement, AFExpertPlacement
from TokenSim.af.planning.compute import ComputeNodePlannerMixin
from TokenSim.af.planning.copies import TensorCopyPlannerMixin
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.layers import RuntimeLayerPlannerMixin
from TokenSim.af.planning.moe import MoELayerPlannerMixin
from TokenSim.af.planning.native import NativeLayerPlannerMixin
from TokenSim.af.planning.replicas import (
    GlobalRoundRobinReplicaSelector,
)
from TokenSim.af.planning.router import RouterLogitsMixin
from TokenSim.af.planning.transfer import TransferStagePlannerMixin
from TokenSim.config.af.domains import (
    ExecutionDomainConfig,
    LogicalInstanceConfig,
)
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError


if TYPE_CHECKING:
    from TokenSim.af.runtime.plan import AFRuntimePlan


class ParallelExecutionPlanner(
    RuntimeLayerPlannerMixin,
    MoELayerPlannerMixin,
    NativeLayerPlannerMixin,
    ComputeNodePlannerMixin,
    TensorCopyPlannerMixin,
    TransferStagePlannerMixin,
    RouterLogitsMixin,
    ConfigurationOwner,
):
    _configuration_fields = ("cluster_config", "runtime_plan", "hidden_size")

    def __init__(self, cluster_config, *, runtime_plan: AFRuntimePlan) -> None:
        self.cluster_config = snapshot(cluster_config)
        self.runtime_plan = runtime_plan
        self.hidden_size = runtime_plan.hidden_size
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
        if block.placement is None:
            raise ConfigurationError(
                "AF blocks require a preselected replica placement"
            )
        if self.cluster_config.af.dag_template_cache:
            return self.template_cache.build(self, block, layer_id, link)
        return self._build_runtime_layer(block, layer_id, link)

    def _expert_placement_view(
        self, domain: ExecutionDomainConfig, instance: LogicalInstanceConfig
    ) -> tuple[AFExpertPlacement, ...]:
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

    @staticmethod
    def _instance(domain: ExecutionDomainConfig, name: str) -> LogicalInstanceConfig:
        for instance in domain.logical_instances:
            if instance.name == name:
                return instance
        raise ConfigurationError(
            f"execution domain {domain.name!r} has no DP replica {name!r}"
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
