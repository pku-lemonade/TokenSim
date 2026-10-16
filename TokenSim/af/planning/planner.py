"""Replica selection and model-layer planning entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from TokenSim.af.caching.estimates import ConfigurationOwner, snapshot
from TokenSim.af.planning.batching import AFBlock, AFBlockPlacement, AFExpertPlacement
from TokenSim.af.planning.compute import ComputeNodePlannerMixin
from TokenSim.af.planning.copies import TensorCopyPlannerMixin
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.layers import RuntimeLayerPlannerMixin
from TokenSim.af.planning.moe import MoELayerPlannerMixin
from TokenSim.af.planning.transfer import TransferStagePlannerMixin
from TokenSim.config.af.domains import ExecutionDomainConfig, LogicalInstanceConfig
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError


class GlobalRoundRobinReplicaSelector:
    def __init__(self) -> None:
        self._cursors: dict[tuple[str, str], int] = {}

    def select(
        self, phase: str, domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        replicas = self.ordered_replicas(phase, domain)
        key = (phase, domain.name)
        cursor = self._cursors.get(key, 0)
        self._cursors[key] = cursor + 1
        return replicas[cursor % len(replicas)]

    @classmethod
    def ordered_replicas(
        cls, phase: str, domain: ExecutionDomainConfig
    ) -> tuple[LogicalInstanceConfig, ...]:
        if phase not in domain.phases:
            raise ConfigurationError(
                f"execution domain {domain.name!r} does not cover phase {phase!r}"
            )
        replicas = tuple(
            sorted(
                domain.logical_instances,
                key=lambda item: (cls._instance_dp_rank(item), item.name),
            )
        )
        if not replicas:
            raise ConfigurationError(
                f"execution domain {domain.name!r} has no DP replicas"
            )
        dp_ranks = [cls._instance_dp_rank(instance) for instance in replicas]
        if len(dp_ranks) != len(set(dp_ranks)):
            raise ConfigurationError(
                f"execution domain {domain.name!r} has duplicate DP replica ranks "
                f"{dp_ranks}"
            )
        return replicas

    @staticmethod
    def _instance_dp_rank(instance: LogicalInstanceConfig) -> int:
        ranks = {mapping.dp_rank for mapping in instance.rank_mapping}
        if len(ranks) != 1:
            raise ConfigurationError(
                f"logical instance {instance.name!r} must represent one DP replica"
            )
        return next(iter(ranks))


@dataclass(frozen=True)
class AFReplicaSelection:
    phase: str
    a_owner: LogicalInstanceConfig
    f_service: LogicalInstanceConfig
    routed_expert_instances: tuple[LogicalInstanceConfig, ...]
    expert_parallel_mode: Literal["expert_tp", "rank_reuse", "expert_sharded"]

    @property
    def dense_ffn_instance(self) -> LogicalInstanceConfig:
        return self.f_service

    @property
    def shared_expert_instance(self) -> LogicalInstanceConfig:
        return self.f_service


class IndependentRoundRobinAFReplicaSelector:
    """Select A ownership and F service replicas from independent global streams."""

    def __init__(self) -> None:
        self._a_owner_selector = GlobalRoundRobinReplicaSelector()
        self._f_service_selector = GlobalRoundRobinReplicaSelector()

    def select_a_owner(
        self, phase: str, a_domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        self._require_capability(a_domain, "af_attention", "A owner")
        return self._a_owner_selector.select(phase, a_domain)

    def select_f_service(
        self, phase: str, f_domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        self._require_capability(f_domain, "af_ffn", "F service")
        return self._f_service_selector.select(phase, f_domain)

    def select_for_block(
        self,
        phase: str,
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
    ) -> AFReplicaSelection:
        a_owner = self.select_a_owner(phase, a_domain)
        return self.select_for_preselected_a(
            phase,
            a_domain,
            f_domain,
            a_owner,
        )

    def select_for_preselected_a(
        self,
        phase: str,
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
        a_owner: LogicalInstanceConfig,
    ) -> AFReplicaSelection:
        """Validate scheduler-owned A placement and independently select F service."""
        self._require_capability(a_domain, "af_attention", "A owner")
        self._require_capability(f_domain, "af_ffn", "F service")
        a_replicas = GlobalRoundRobinReplicaSelector.ordered_replicas(phase, a_domain)
        f_replicas = GlobalRoundRobinReplicaSelector.ordered_replicas(phase, f_domain)
        ep_view = f_domain.expert_parallel

        matching_a_replicas = tuple(
            replica for replica in a_replicas if replica.name == a_owner.name
        )
        if len(matching_a_replicas) != 1 or matching_a_replicas[0] != a_owner:
            raise ConfigurationError(
                f"preselected A owner {a_owner.name!r} is outside execution domain "
                f"{a_domain.name!r} for phase {phase!r}"
            )
        f_service = self._f_service_selector.select(phase, f_domain)
        routed_instances = (f_service,) if ep_view.mode == "expert_tp" else f_replicas
        return AFReplicaSelection(
            phase=phase,
            a_owner=a_owner,
            f_service=f_service,
            routed_expert_instances=routed_instances,
            expert_parallel_mode=ep_view.mode,
        )

    @staticmethod
    def _require_capability(
        domain: ExecutionDomainConfig, expected: str, selection_role: str
    ) -> None:
        if domain.capability != expected:
            raise ConfigurationError(
                f"{selection_role} selector requires capability {expected!r}; "
                f"domain {domain.name!r} has {domain.capability!r}"
            )


if TYPE_CHECKING:
    from TokenSim.af.runtime.plan import AFRuntimePlan


class ParallelExecutionPlanner(
    RuntimeLayerPlannerMixin,
    MoELayerPlannerMixin,
    ComputeNodePlannerMixin,
    TensorCopyPlannerMixin,
    TransferStagePlannerMixin,
    ConfigurationOwner,
):
    _configuration_fields = ("cluster_config", "runtime_plan", "hidden_size")

    def __init__(self, cluster_config, *, runtime_plan: AFRuntimePlan) -> None:
        self.cluster_config = snapshot(cluster_config)
        self.runtime_plan = runtime_plan
        self.hidden_size = runtime_plan.hidden_size
        from TokenSim.af.caching.templates import TemplateCache

        self.template_cache = TemplateCache()
        self._endpoints = {}
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
        cache_limit = self.cluster_config.af.dag_template_cache_max_tokens
        if self.cluster_config.af.dag_template_cache and (
            not cache_limit or block.token_count <= cache_limit
        ):
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

    def _build_native_moe_tail(
        self,
        *,
        layer_id,
        layer,
        snapshot,
        link,
        prefix,
        nodes,
        a_bindings,
        f_bindings,
        attention_ids,
        a_collective,
    ) -> AFExecutionDAG:
        """Single-rank Qwen AFD: send hidden, route/combine on F, return hidden.

        The planner retains mathematical combine as a separate device service.
        A fused measured estimate must explicitly cover it before that service
        can be elided. No communication time is inferred from expert count.
        """
        if len(a_bindings) != 1 or len(f_bindings) != 1 or layer.num_shared_experts:
            raise ConfigurationError(
                "native AFD MoE tail requires 1A1F without shared experts"
            )
        a, f = a_bindings[0], f_bindings[0]
        tokens = tuple(snapshot.original_tokens)
        edges = tuple(self._routing_edge(item) for item in snapshot.assignments)
        router_id, expert_id, combine_id, residual_id = (
            f"{prefix}:{name}"
            for name in ("router", "grouped_experts", "f_combine", "residual")
        )
        copies = self._activation_copies(
            tokens=tokens,
            a_bindings=a_bindings,
            target_specs=((f, router_id, tokens),),
            producer_ids=attention_ids,
            collective=a_collective,
            barrier_predecessor=None,
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype,
        )
        arrivals = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:a_to_f",
            operation="af_a_to_f",
            direction="a_to_f",
            copies=copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=router_id,
                operation="router_topk",
                stage="router",
                predecessors=arrivals[router_id],
                binding=f,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(tokens),
                original_token_count=len(tokens),
                assignment_count=len(edges),
                output_token_count=len(tokens),
                routing_edges=edges,
                expert_ids=tuple(e for e, _ in snapshot.expert_histogram),
                shape={
                    "tokens": len(tokens),
                    "hidden_size": self.hidden_size,
                    "num_experts": layer.num_experts,
                    "top_k": layer.top_k,
                },
            )
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=expert_id,
                operation="routed_expert",
                stage="routed_expert",
                predecessors=(router_id,),
                binding=f,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(edges),
                original_token_count=len(tokens),
                assignment_count=len(edges),
                output_token_count=len(edges),
                routing_edges=edges,
                expert_ids=tuple(e for e, _ in snapshot.expert_histogram),
                expert_workload=tuple(snapshot.expert_histogram),
            )
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=combine_id,
                operation="f_weighted_combine",
                stage="weighted_combine",
                predecessors=(expert_id,),
                binding=f,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(tokens),
                original_token_count=len(tokens),
                assignment_count=len(edges),
                input_edge_count=len(edges),
                output_token_count=len(tokens),
                routing_edges=edges,
                source_layout="partial",
                target_layout="replicated",
                shape={
                    "tokens": len(tokens),
                    "hidden_size": self.hidden_size,
                    "elements": len(tokens) * self.hidden_size,
                },
            )
        )
        copies = self._output_copies(
            items=tuple(
                (token, f"request:{token[0]}:token:{token[1]}:combined", None, None)
                for token in tokens
            ),
            source_bindings=f_bindings,
            producer_ids={f.domain_rank: combine_id},
            collective=None,
            target_bindings=a_bindings,
            consumer_ids={a.domain_rank: residual_id},
            tensor_layout="replicated",
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).f_to_a_dtype,
        )
        returned = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:f_to_a",
            operation="af_f_to_a",
            direction="f_to_a",
            copies=copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=residual_id,
                operation="residual",
                stage="residual",
                predecessors=returned[residual_id],
                binding=a,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(tokens),
                original_token_count=len(tokens),
                output_token_count=len(tokens),
                source_layout="replicated",
                target_layout="replicated",
            )
        )
        return AFExecutionDAG(tuple(nodes))
