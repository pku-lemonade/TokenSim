"""Planner legacy."""

from __future__ import annotations

from typing import Literal

from TokenSim.af.models import AFBlock
from TokenSim.af.planning_dag import AFExecutionDAG
from TokenSim.af.planning_nodes import ExecutionNode, RankLocalWork
from TokenSim.config.cluster_config import (
    AFLinkConfig,
    DomainCollectiveConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
)
from TokenSim.errors import ConfigurationError


class LegacyLayerPlannerMixin:
    def _build_legacy_layer(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
    ) -> AFExecutionDAG:
        if block.placement is None:
            raise ConfigurationError(
                f"AF block {block.block_id} has no preselected DP placement"
            )
        a_domain = self.cluster_config.execution_domains[block.placement.a_domain]
        f_domain = self.cluster_config.execution_domains[block.placement.f_domain]
        a_instance = self._instance(a_domain, block.placement.a_instance)
        f_instance = self._instance(f_domain, block.placement.f_instance)
        prefix = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}"
        nodes: list[ExecutionNode] = []

        a_mappings = sorted(
            self._runtime_mappings(a_domain, a_instance),
            key=lambda item: item.tp_rank,
        )
        a_group = tuple(mapping.domain_rank for mapping in a_mappings)
        a_ids = []
        for mapping in a_mappings:
            node_id = f"{prefix}:a:tp{mapping.tp_rank}"
            a_ids.append(node_id)
            nodes.append(
                self._compute_node(
                    node_id=node_id,
                    operation="a_rank_compute",
                    stage="a",
                    predecessors=(),
                    domain=a_domain,
                    instance=a_instance,
                    mapping=mapping,
                    token_count=block.token_count,
                    expert_ids=(),
                    tp_group=a_group,
                )
            )

        a_ready = tuple(a_ids)
        if len(a_mappings) > 1:
            collective = self._collective_node(
                node_id=f"{prefix}:a:tp_collective",
                predecessors=a_ready,
                domain=a_domain,
                instance=a_instance,
                mappings=a_mappings,
                link=link,
                after_stage="a_compute",
                token_count=block.token_count,
            )
            nodes.append(collective)
            a_ready = (collective.node_id,)

        a_leader = min(a_mappings, key=lambda item: item.tp_rank)
        router = self._compute_node(
            node_id=f"{prefix}:router",
            operation="router_topk",
            stage="router",
            predecessors=a_ready,
            domain=a_domain,
            instance=a_instance,
            mapping=a_leader,
            token_count=block.token_count,
            expert_ids=tuple(expert for expert, _ in block.expert_histogram),
            tp_group=a_group,
        )
        nodes.append(router)

        f_mappings = sorted(
            self._routed_runtime_mappings(f_domain, f_instance),
            key=lambda item: (item.ep_rank, item.tp_rank),
        )
        loads, expert_workloads = self._ep_workloads(block, f_domain, f_instance)
        active_ep_ranks = tuple(
            sorted(rank for rank, count in loads.items() if count > 0)
        )
        routed_count = sum(loads.values())
        routed_payload = routed_count * self.hidden_size * self.dtype_size_bytes
        active_f_mappings = tuple(
            mapping for mapping in f_mappings if mapping.ep_rank in active_ep_ranks
        )
        transfer_mappings = tuple(dict.fromkeys((a_leader,) + active_f_mappings))
        a_to_f = self._communication_node(
            node_id=f"{prefix}:af_a_to_f",
            operation="af_a_to_f",
            direction="a_to_f",
            predecessors=(router.node_id,),
            source_domain=a_domain,
            target_domain=f_domain,
            source_instance=a_instance,
            target_instance=f_instance,
            mappings=transfer_mappings,
            link=link,
            payload_bytes=routed_payload,
            dp_rank=block.placement.f_dp_rank,
        )
        nodes.append(a_to_f)
        expert_predecessors = (a_to_f.node_id,)

        branch_ends: list[str] = []
        for ep_rank in active_ep_ranks:
            shard = [mapping for mapping in f_mappings if mapping.ep_rank == ep_rank]
            tp_group = tuple(mapping.domain_rank for mapping in shard)
            compute_ids = []
            for mapping in shard:
                mapping_instance = self._instance_for_mapping(f_domain, mapping)
                node_id = f"{prefix}:expert:ep{ep_rank}:tp{mapping.tp_rank}"
                compute_ids.append(node_id)
                nodes.append(
                    self._compute_node(
                        node_id=node_id,
                        operation="expert_rank_compute",
                        stage="expert",
                        predecessors=expert_predecessors,
                        domain=f_domain,
                        instance=mapping_instance,
                        mapping=mapping,
                        token_count=loads[ep_rank],
                        expert_ids=tuple(
                            expert_id
                            for expert_id, _ in expert_workloads.get(ep_rank, ())
                        ),
                        expert_workload=expert_workloads.get(ep_rank, ()),
                        tp_group=tp_group,
                        ep_group=tuple(
                            item.domain_rank
                            for item in f_mappings
                            if item.tp_rank == mapping.tp_rank
                        ),
                    )
                )
            if len(shard) > 1:
                collective = self._collective_node(
                    node_id=f"{prefix}:expert:ep{ep_rank}:tp_collective",
                    predecessors=tuple(compute_ids),
                    domain=f_domain,
                    instance=f_instance,
                    mappings=shard,
                    link=link,
                    after_stage="expert_compute",
                    token_count=loads[ep_rank],
                )
                nodes.append(collective)
                branch_ends.append(collective.node_id)
            else:
                branch_ends.extend(compute_ids)

        f_ready = tuple(branch_ends) or expert_predecessors
        nodes.append(
            self._communication_node(
                node_id=f"{prefix}:af_f_to_a",
                operation="af_f_to_a",
                direction="f_to_a",
                predecessors=f_ready,
                source_domain=f_domain,
                target_domain=a_domain,
                source_instance=f_instance,
                target_instance=a_instance,
                mappings=tuple(dict.fromkeys(active_f_mappings + (a_leader,))),
                link=link,
                payload_bytes=routed_payload,
                dp_rank=block.placement.a_dp_rank,
            )
        )
        dag = AFExecutionDAG(tuple(nodes))
        if self.runtime_plan is not None:
            for node in dag.nodes:
                self.runtime_plan.validate_execution_node(node)
        return dag

    def _compute_node(
        self,
        *,
        node_id: str,
        operation: str,
        stage: Literal["a", "router", "expert"],
        predecessors: tuple[str, ...],
        domain: ExecutionDomainConfig,
        instance: LogicalInstanceConfig,
        mapping: DomainRankMapping,
        token_count: int,
        expert_ids: tuple[int, ...],
        expert_workload: tuple[tuple[int, int], ...] = (),
        tp_group: tuple[int, ...],
        ep_group: tuple[int, ...] = (),
    ) -> ExecutionNode:
        tp_size = domain.parallel.tensor_parallel_size
        local_hidden = (self.hidden_size + tp_size - 1) // tp_size
        effective_hidden = self.hidden_size if stage == "router" else local_hidden
        work = RankLocalWork(
            token_count=token_count,
            tensor_shape=(
                ("tokens", token_count),
                ("hidden_size", effective_hidden),
                ("global_hidden_size", self.hidden_size),
                ("tp_size", tp_size),
            ),
            expert_ids=expert_ids,
            expert_workload=expert_workload,
        )
        return ExecutionNode(
            node_id=node_id,
            operation=operation,
            kind="compute",
            predecessor_ids=predecessors,
            stage=stage,
            direction=None,
            source_domain=domain.name,
            target_domain=domain.name,
            logical_participants=(instance.name,),
            domain_ranks=(mapping.domain_rank,),
            physical_workers=(mapping.physical_worker.identity,),
            dp_rank=mapping.dp_rank,
            tp_rank=mapping.tp_rank,
            ep_rank=mapping.ep_rank,
            tp_group=tp_group,
            ep_group=ep_group,
            local_work=work,
        )

    def _collective_node(
        self,
        *,
        node_id: str,
        predecessors: tuple[str, ...],
        domain: ExecutionDomainConfig,
        instance: LogicalInstanceConfig,
        mappings: list[DomainRankMapping],
        link: AFLinkConfig,
        after_stage: str,
        token_count: int,
    ) -> ExecutionNode:
        profile = self._collective_profile(domain, after_stage)
        payload = (
            profile.payload_bytes
            if profile is not None and profile.payload_bytes is not None
            else token_count * self.hidden_size * self.dtype_size_bytes
        )
        return ExecutionNode(
            node_id=node_id,
            operation="tp_collective",
            kind="communication",
            predecessor_ids=predecessors,
            stage=None,
            direction="tp_collective",
            source_domain=domain.name,
            target_domain=domain.name,
            logical_participants=(instance.name,),
            domain_ranks=tuple(mapping.domain_rank for mapping in mappings),
            physical_workers=tuple(
                mapping.physical_worker.identity for mapping in mappings
            ),
            dp_rank=mappings[0].dp_rank,
            tp_group=tuple(mapping.domain_rank for mapping in mappings),
            network=(profile.network if profile else None) or link.network,
            resource_group=(profile.resource_group if profile else None)
            or link.resource_group,
            resource_lane=(
                (profile.resource_lane if profile is not None else None)
                or (
                    "shared"
                    if link.duplex == "half"
                    else "a_to_f" if domain.capability == "af_attention" else "f_to_a"
                )
            ),
            payload_bytes=payload,
            metadata={
                "collective_algorithm": profile.operation if profile else "all_reduce"
            },
        )

    @staticmethod
    def _collective_profile(
        domain: ExecutionDomainConfig, after_stage: str
    ) -> DomainCollectiveConfig | None:
        return next(
            (
                item
                for item in domain.communication_plan
                if item.after_stage == after_stage
            ),
            None,
        )

    @staticmethod
    def _communication_node(
        *,
        node_id: str,
        operation: str,
        direction: str,
        predecessors: tuple[str, ...],
        source_domain: ExecutionDomainConfig,
        target_domain: ExecutionDomainConfig,
        source_instance: LogicalInstanceConfig,
        target_instance: LogicalInstanceConfig,
        mappings: tuple[DomainRankMapping, ...],
        link: AFLinkConfig,
        payload_bytes: int,
        dp_rank: int,
    ) -> ExecutionNode:
        return ExecutionNode(
            node_id=node_id,
            operation=operation,
            kind="communication",
            predecessor_ids=predecessors,
            stage=None,
            direction=direction,
            source_domain=source_domain.name,
            target_domain=target_domain.name,
            logical_participants=(source_instance.name, target_instance.name),
            domain_ranks=tuple(mapping.domain_rank for mapping in mappings),
            physical_workers=tuple(
                dict.fromkeys(mapping.physical_worker.identity for mapping in mappings)
            ),
            dp_rank=dp_rank,
            ep_group=tuple(mapping.domain_rank for mapping in mappings),
            network=link.network,
            resource_group=link.resource_group,
            payload_bytes=payload_bytes,
        )
