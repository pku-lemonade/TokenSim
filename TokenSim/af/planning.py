from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from TokenSim.af.models import AFBlock, AFBlockPlacement, AFExpertPlacement
from TokenSim.config.cluster_config import (
    AFLinkConfig,
    DomainCollectiveConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
)
from TokenSim.errors import ConfigurationError

if False:  # pragma: no cover - type-only import without a runtime cycle
    from TokenSim.af.runtime import AFRuntimePlan


NodeKind = Literal["compute", "communication"]


@dataclass(frozen=True)
class RankLocalWork:
    token_count: int
    tensor_shape: tuple[tuple[str, int], ...] = ()
    expert_ids: tuple[int, ...] = ()
    expert_workload: tuple[tuple[int, int], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "tensor_shape": dict(self.tensor_shape),
            "expert_ids": self.expert_ids,
            "expert_workload": dict(self.expert_workload),
        }


@dataclass(frozen=True)
class ExecutionNode:
    node_id: str
    operation: str
    kind: NodeKind
    predecessor_ids: tuple[str, ...]
    stage: Literal["a", "router", "expert"] | None
    direction: str | None
    source_domain: str
    target_domain: str
    logical_participants: tuple[str, ...]
    domain_ranks: tuple[int, ...]
    physical_workers: tuple[str, ...]
    dp_rank: int
    tp_rank: int | None = None
    ep_rank: int | None = None
    tp_group: tuple[int, ...] = ()
    ep_group: tuple[int, ...] = ()
    local_work: RankLocalWork | None = None
    network: str | None = None
    resource_group: str | None = None
    payload_bytes: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ConfigurationError("execution node id must be non-empty")
        if self.kind == "compute" and len(self.physical_workers) != 1:
            raise ConfigurationError(
                f"compute node {self.node_id!r} must bind exactly one physical worker"
            )


@dataclass(frozen=True)
class AFExecutionDAG:
    nodes: tuple[ExecutionNode, ...]

    def __post_init__(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("AF execution DAG contains duplicate node ids")
        known: set[str] = set()
        for node in self.nodes:
            missing = set(node.predecessor_ids) - known
            if missing:
                raise ConfigurationError(
                    f"execution node {node.node_id!r} has missing or cyclic "
                    f"predecessors {sorted(missing)}"
                )
            known.add(node.node_id)

    @property
    def final_node_id(self) -> str:
        if not self.nodes:
            raise ConfigurationError("AF execution DAG must contain at least one node")
        return self.nodes[-1].node_id


class GlobalRoundRobinReplicaSelector:
    def __init__(self) -> None:
        self._cursors: dict[tuple[str, str], int] = {}

    def select(
        self, phase: str, domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        replicas = sorted(
            domain.logical_instances,
            key=lambda item: (self._instance_dp_rank(item), item.name),
        )
        if not replicas:
            raise ConfigurationError(
                f"execution domain {domain.name!r} has no DP replicas"
            )
        key = (phase, domain.name)
        cursor = self._cursors.get(key, 0)
        self._cursors[key] = cursor + 1
        return replicas[cursor % len(replicas)]

    @staticmethod
    def _instance_dp_rank(instance: LogicalInstanceConfig) -> int:
        ranks = {mapping.dp_rank for mapping in instance.rank_mapping}
        if len(ranks) != 1:
            raise ConfigurationError(
                f"logical instance {instance.name!r} must represent one DP replica"
            )
        return next(iter(ranks))


class ParallelExecutionPlanner:
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
        self.cluster_config = cluster_config
        self.hidden_size = hidden_size
        self.dtype_size_bytes = dtype_size_bytes
        self.num_experts = num_experts
        self.top_k = max(1, top_k)
        self.runtime_plan = runtime_plan

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
            self._runtime_mappings(f_domain, f_instance),
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
                node_id = f"{prefix}:expert:ep{ep_rank}:tp{mapping.tp_rank}"
                compute_ids.append(node_id)
                nodes.append(
                    self._compute_node(
                        node_id=node_id,
                        operation="expert_rank_compute",
                        stage="expert",
                        predecessors=expert_predecessors,
                        domain=f_domain,
                        instance=f_instance,
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
        mappings_by_ep = {
            ep_rank: tuple(
                sorted(
                    (
                        mapping
                        for mapping in instance.rank_mapping
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


# Import compatibility for callers migrating from the previous linear planner.
AFExecutionStep = ExecutionNode
AFExecutionPlanBuilder = ParallelExecutionPlanner
