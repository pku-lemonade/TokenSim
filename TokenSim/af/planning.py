from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Mapping

from TokenSim.af.latency import (
    AFComputeStage,
    AFEndpoint,
    AFLayerType,
    AFRoutingEdge,
    AFTensorLayout,
    AFTensorSlice,
)
from TokenSim.af.models import AFBlock, AFBlockPlacement, AFExpertPlacement
from TokenSim.af.snapshot import ConfigurationOwner, snapshot
from TokenSim.config.cluster_config import (
    AFLinkConfig,
    DomainCollectiveConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
)
from TokenSim.errors import ConfigurationError
from TokenSim.dtypes import dtype_size_bytes

if False:  # pragma: no cover - type-only import without a runtime cycle
    from TokenSim.af.runtime import AFRuntimePlan


NodeKind = Literal["compute", "communication"]
ExecutionStage = AFComputeStage | Literal["cast"]
_EXECUTION_STAGES = frozenset(
    {
        "a",
        "router",
        "expert",
        "attention",
        "dense_ffn",
        "routed_expert",
        "shared_expert",
        "weighted_combine",
        "residual",
        "cast",
        "encode",
        "decode",
        "layout",
    }
)


@dataclass(frozen=True)
class _TransferCopy:
    source_binding: Any
    target_binding: Any
    tensor_slice: AFTensorSlice
    predecessor_id: str
    consumer_id: str
    source_layout: AFTensorLayout
    target_layout: AFTensorLayout
    routing_edges: tuple[AFRoutingEdge, ...] = ()


@dataclass(frozen=True)
class RankLocalWork:
    token_count: int
    tensor_shape: tuple[tuple[str, int], ...] = ()
    expert_ids: tuple[int, ...] = ()
    expert_workload: tuple[tuple[int, int], ...] = ()
    original_token_count: int | None = None
    assignment_count: int | None = None
    input_edge_count: int | None = None
    output_token_count: int | None = None
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    owner_endpoint: AFEndpoint | None = None
    source_layout: AFTensorLayout | None = None
    target_layout: AFTensorLayout | None = None
    routing_edges: tuple[AFRoutingEdge, ...] = ()
    source_dtype: str | None = None
    target_dtype: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "token_count",
            "original_token_count",
            "assignment_count",
            "input_edge_count",
            "output_token_count",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError(
                    f"rank-local work {field_name} must be a non-negative integer"
                )
        for field_name in ("source_layout", "target_layout"):
            value = getattr(self, field_name)
            if value is not None and value not in {
                "sharded",
                "replicated",
                "partial",
            }:
                raise ConfigurationError(
                    f"rank-local work {field_name} has unsupported layout {value!r}"
                )
        object.__setattr__(self, "tensor_shape", tuple(self.tensor_shape))
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))
        object.__setattr__(self, "expert_workload", tuple(self.expert_workload))
        object.__setattr__(self, "tensor_slices", tuple(self.tensor_slices))
        object.__setattr__(self, "routing_edges", tuple(self.routing_edges))

    def as_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "tensor_shape": dict(self.tensor_shape),
            "expert_ids": self.expert_ids,
            "expert_workload": dict(self.expert_workload),
            "original_token_count": self.original_token_count,
            "assignment_count": self.assignment_count,
            "input_edge_count": self.input_edge_count,
            "output_token_count": self.output_token_count,
            "tensor_slices": tuple(asdict(item) for item in self.tensor_slices),
            "owner_endpoint": (
                asdict(self.owner_endpoint) if self.owner_endpoint is not None else None
            ),
            "source_layout": self.source_layout,
            "target_layout": self.target_layout,
            "routing_edges": tuple(asdict(item) for item in self.routing_edges),
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
        }


@dataclass(frozen=True)
class ExecutionNode:
    node_id: str
    operation: str
    kind: NodeKind
    predecessor_ids: tuple[str, ...]
    stage: ExecutionStage | None
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
    resource_lane: str | None = None
    payload_bytes: int | None = None
    logical_input_bytes_per_rank: tuple[tuple[int, int], ...] = ()
    logical_output_bytes_per_rank: tuple[tuple[int, int], ...] = ()
    layer_type: AFLayerType | None = None
    logical_stage_id: str | None = None
    leg_id: str | None = None
    source_endpoint: AFEndpoint | None = None
    target_endpoint: AFEndpoint | None = None
    participant_endpoints: tuple[AFEndpoint, ...] = ()
    input_producer_ids: tuple[tuple[str, str], ...] = ()
    tensor_slices: tuple[AFTensorSlice, ...] = ()
    source_layout: AFTensorLayout | None = None
    target_layout: AFTensorLayout | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    routing_edges: tuple[AFRoutingEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ConfigurationError("execution node id must be non-empty")
        if not self.operation:
            raise ConfigurationError(
                f"execution node {self.node_id!r} operation must be non-empty"
            )
        if self.kind not in {"compute", "communication"}:
            raise ConfigurationError(
                f"execution node {self.node_id!r} has unsupported kind {self.kind!r}"
            )
        if self.kind == "compute" and len(self.physical_workers) != 1:
            raise ConfigurationError(
                f"compute node {self.node_id!r} must bind exactly one physical worker"
            )
        if self.kind == "compute" and self.stage is None:
            raise ConfigurationError(
                f"compute node {self.node_id!r} must declare a compute stage"
            )
        if self.stage is not None and self.stage not in _EXECUTION_STAGES:
            raise ConfigurationError(
                f"execution node {self.node_id!r} has unsupported stage {self.stage!r}"
            )
        if self.layer_type is not None and self.layer_type not in {"dense", "moe"}:
            raise ConfigurationError(
                f"execution node {self.node_id!r} has unsupported layer type "
                f"{self.layer_type!r}"
            )
        if self.logical_stage_id is not None and not self.logical_stage_id:
            raise ConfigurationError(
                f"execution node {self.node_id!r} logical_stage_id must be non-empty"
            )
        if self.payload_bytes is not None and (
            type(self.payload_bytes) is not int or self.payload_bytes < 0
        ):
            raise ConfigurationError(
                f"execution node {self.node_id!r} payload_bytes must be non-negative"
            )
        for field_name in ("source_layout", "target_layout"):
            value = getattr(self, field_name)
            if value is not None and value not in {
                "sharded",
                "replicated",
                "partial",
            }:
                raise ConfigurationError(
                    f"execution node {self.node_id!r} {field_name} has unsupported "
                    f"layout {value!r}"
                )
        for field_name in (
            "predecessor_ids",
            "logical_participants",
            "domain_ranks",
            "physical_workers",
            "tp_group",
            "ep_group",
            "participant_endpoints",
            "input_producer_ids",
            "tensor_slices",
            "logical_input_bytes_per_rank",
            "logical_output_bytes_per_rank",
            "routing_edges",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if self.input_producer_ids:
            endpoints = tuple(endpoint for endpoint, _ in self.input_producer_ids)
            if (
                self.kind != "communication"
                or endpoints != self.physical_workers
                or len(set(endpoints)) != len(endpoints)
                or any(
                    producer not in self.predecessor_ids
                    for _, producer in self.input_producer_ids
                )
            ):
                raise ConfigurationError(
                    "collective input bindings must uniquely cover endpoints in participant order and reference predecessors"
                )
        elif self.metadata.get("collective_input_binding_required"):
            raise ConfigurationError("collective is missing required input bindings")


@dataclass(frozen=True)
class AFExecutionDAG:
    nodes: tuple[ExecutionNode, ...]
    terminal_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        object.__setattr__(self, "nodes", nodes)
        if not nodes:
            raise ConfigurationError("AF execution DAG must contain at least one node")
        if not all(isinstance(node, ExecutionNode) for node in nodes):
            raise ConfigurationError(
                "AF execution DAG nodes must contain ExecutionNode objects"
            )
        ids = [node.node_id for node in nodes]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("AF execution DAG contains duplicate node ids")
        all_ids = set(ids)
        known: set[str] = set()
        referenced: set[str] = set()
        for node in nodes:
            if len(node.predecessor_ids) != len(set(node.predecessor_ids)):
                raise ConfigurationError(
                    f"execution node {node.node_id!r} contains duplicate predecessors"
                )
            missing = set(node.predecessor_ids) - all_ids
            if missing:
                raise ConfigurationError(
                    f"execution node {node.node_id!r} has missing or cyclic "
                    f"predecessors {sorted(missing)}"
                )
            out_of_order = set(node.predecessor_ids) - known
            if out_of_order:
                raise ConfigurationError(
                    f"execution node {node.node_id!r} has out-of-order or cyclic "
                    f"predecessors {sorted(out_of_order)}"
                )
            referenced.update(node.predecessor_ids)
            known.add(node.node_id)

        sink_ids = tuple(node_id for node_id in ids if node_id not in referenced)
        supplied_terminals = tuple(self.terminal_node_ids)
        if supplied_terminals:
            if len(supplied_terminals) != len(set(supplied_terminals)):
                raise ConfigurationError(
                    "AF execution DAG contains duplicate terminal node ids"
                )
            missing_terminals = set(supplied_terminals) - all_ids
            if missing_terminals:
                raise ConfigurationError(
                    "AF execution DAG terminal nodes do not exist: "
                    f"{sorted(missing_terminals)}"
                )
            if set(supplied_terminals) != set(sink_ids):
                raise ConfigurationError(
                    "AF execution DAG terminal nodes must equal all sink nodes: "
                    f"expected={list(sink_ids)!r}, got={list(supplied_terminals)!r}"
                )
        object.__setattr__(self, "terminal_node_ids", sink_ids)
        self.validate_runtime_semantics()

    def validate_runtime_semantics(self) -> None:
        runtime_nodes = tuple(
            node for node in self.nodes if node.layer_type is not None
        )
        if not runtime_nodes:
            return
        if len(runtime_nodes) != len(self.nodes):
            raise ConfigurationError(
                "AF runtime DAG cannot mix typed and legacy execution nodes"
            )
        layer_types = {node.layer_type for node in runtime_nodes}
        if len(layer_types) != 1:
            raise ConfigurationError(
                f"AF runtime DAG must describe one layer type, got {sorted(layer_types)}"
            )
        layer_type = next(iter(layer_types))
        attention_nodes = tuple(
            node for node in runtime_nodes if node.stage == "attention"
        )
        if not attention_nodes:
            raise ConfigurationError("AF runtime DAG must contain A Attention work")
        a_domains = {node.source_domain for node in attention_nodes}
        if len(a_domains) != 1:
            raise ConfigurationError(
                f"AF runtime DAG Attention work spans multiple A domains: {a_domains}"
            )
        a_domain = next(iter(a_domains))
        attention_counts = {
            node.local_work.original_token_count
            for node in attention_nodes
            if node.local_work is not None
        }
        if len(attention_counts) != 1 or None in attention_counts:
            raise ConfigurationError(
                "AF runtime Attention ranks must agree on original token count"
            )
        original_token_count = next(iter(attention_counts))
        residual_nodes = tuple(
            node for node in runtime_nodes if node.stage == "residual"
        )
        for node in residual_nodes:
            if node.source_domain != a_domain or node.target_domain != a_domain:
                raise ConfigurationError(
                    f"AF residual node {node.node_id!r} must execute in A domain "
                    f"{a_domain!r}, got source={node.source_domain!r}, "
                    f"target={node.target_domain!r}"
                )
        attention_ranks = {node.domain_ranks[0] for node in attention_nodes}
        residual_ranks = {node.domain_ranks[0] for node in residual_nodes}
        if residual_ranks != attention_ranks or len(residual_nodes) != len(
            residual_ranks
        ):
            raise ConfigurationError(
                "AF runtime residual must execute exactly once on every A tensor owner rank"
            )
        for node in residual_nodes:
            work = node.local_work
            if work is None or (
                work.token_count != original_token_count
                or work.original_token_count != original_token_count
                or work.output_token_count != original_token_count
            ):
                raise ConfigurationError(
                    "AF runtime residual must process and produce the original token "
                    f"count on node={node.node_id!r}, expected={original_token_count}"
                )

        by_id = {node.node_id: node for node in runtime_nodes}
        for terminal_id in self.terminal_node_ids:
            terminal = by_id[terminal_id]
            if terminal.source_domain != a_domain or terminal.target_domain != a_domain:
                raise ConfigurationError(
                    f"AF runtime terminal {terminal_id!r} must be A-side, got "
                    f"source={terminal.source_domain!r}, "
                    f"target={terminal.target_domain!r}"
                )
            if terminal.stage not in {"residual", "layout"} and (
                terminal.operation != "all_gather"
            ):
                raise ConfigurationError(
                    f"AF runtime terminal {terminal_id!r} must complete residual/layout"
                )

        transfer_copies: set[tuple[object, ...]] = set()
        for node in runtime_nodes:
            if node.operation not in {"af_a_to_f", "af_f_to_a"}:
                continue
            for tensor_slice in node.tensor_slices:
                copy_key = (
                    node.logical_stage_id,
                    node.direction,
                    node.source_endpoint,
                    node.target_endpoint,
                    tensor_slice.tensor_item_id,
                    tensor_slice.interval,
                    tensor_slice.partial_id,
                )
                if copy_key in transfer_copies:
                    raise ConfigurationError(
                        "AF runtime DAG contains a duplicate physical tensor copy: "
                        f"{copy_key!r}"
                    )
                transfer_copies.add(copy_key)
        for operation in ("af_a_to_f", "af_f_to_a"):
            stage_ids = {
                node.logical_stage_id
                for node in runtime_nodes
                if node.operation == operation
            }
            if len(stage_ids) != 1 or None in stage_ids:
                raise ConfigurationError(
                    f"AF runtime DAG requires exactly one logical {operation} stage, "
                    f"got {stage_ids!r}"
                )

        if layer_type == "dense":
            forbidden = {
                "router",
                "routed_expert",
                "shared_expert",
                "weighted_combine",
            }
            present = {node.stage for node in runtime_nodes} & forbidden
            if present:
                raise ConfigurationError(
                    f"Dense AF runtime DAG contains MoE-only stages: {sorted(present)}"
                )
            return

        routers = tuple(node for node in runtime_nodes if node.stage == "router")
        if len(routers) != 1 or routers[0].local_work is None:
            raise ConfigurationError(
                "MoE AF runtime DAG must contain exactly one Router work item"
            )
        router_work = routers[0].local_work
        original_tokens = router_work.original_token_count
        if original_tokens is None:
            raise ConfigurationError("MoE Router must declare original token count")
        if original_tokens != original_token_count:
            raise ConfigurationError(
                "MoE Router token count must match A Attention token count"
            )
        expected_edges = tuple(router_work.routing_edges)
        expected_edge_set = set(expected_edges)
        if len(expected_edges) != len(expected_edge_set):
            raise ConfigurationError(
                "MoE Router contains duplicate routing assignments"
            )
        if router_work.assignment_count != len(expected_edges):
            raise ConfigurationError(
                "MoE Router assignment count does not match its routing edges"
            )

        routed_nodes = tuple(
            node for node in runtime_nodes if node.stage == "routed_expert"
        )
        edge_consumers: dict[AFRoutingEdge, list[ExecutionNode]] = {}
        for node in routed_nodes:
            work = node.local_work
            if work is None or work.assignment_count != len(work.routing_edges):
                raise ConfigurationError(
                    f"Routed Expert node {node.node_id!r} has inconsistent workload"
                )
            if len(work.routing_edges) != len(set(work.routing_edges)):
                raise ConfigurationError(
                    f"Routed Expert node {node.node_id!r} consumes an assignment twice"
                )
            for edge in work.routing_edges:
                if edge.expert_id not in work.expert_ids:
                    raise ConfigurationError(
                        f"Routed Expert node {node.node_id!r} consumes edge for "
                        f"non-resident Expert {edge.expert_id}"
                    )
                edge_consumers.setdefault(edge, []).append(node)
        if set(edge_consumers) != expected_edge_set:
            missing = expected_edge_set - set(edge_consumers)
            extra = set(edge_consumers) - expected_edge_set
            raise ConfigurationError(
                "Routed Expert branches must consume every Router assignment exactly "
                f"for their TP group: missing={len(missing)}, extra={len(extra)}"
            )
        for edge, consumers in edge_consumers.items():
            groups = {node.tp_group for node in consumers}
            if len(groups) != 1:
                raise ConfigurationError(
                    f"routing assignment {edge!r} spans inconsistent Expert TP groups"
                )
            expected_ranks = set(next(iter(groups)))
            actual_ranks = [node.domain_ranks[0] for node in consumers]
            if (
                len(actual_ranks) != len(set(actual_ranks))
                or set(actual_ranks) != expected_ranks
            ):
                raise ConfigurationError(
                    "routing assignment must be consumed once on every Expert TP "
                    f"rank: expert={edge.expert_id}, expected={sorted(expected_ranks)}, "
                    f"actual={sorted(actual_ranks)}"
                )

        shared_nodes = tuple(
            node for node in runtime_nodes if node.stage == "shared_expert"
        )
        if shared_nodes:
            shared_groups = {node.tp_group for node in shared_nodes}
            shared_ranks = [node.domain_ranks[0] for node in shared_nodes]
            if len(shared_groups) != 1 or len(shared_ranks) != len(set(shared_ranks)):
                raise ConfigurationError(
                    "Shared Expert branch must contain one work item per F TP rank"
                )
            if set(shared_ranks) != set(next(iter(shared_groups))):
                raise ConfigurationError(
                    "Shared Expert branch does not cover its complete F TP group"
                )
            for node in shared_nodes:
                work = node.local_work
                if work is None or (
                    work.token_count != original_tokens
                    or work.original_token_count != original_tokens
                    or work.output_token_count != original_tokens
                ):
                    raise ConfigurationError(
                        "Shared Expert must process every original token exactly once "
                        f"per TP shard: node={node.node_id!r}, expected={original_tokens}"
                    )

        combine_nodes = tuple(
            node for node in runtime_nodes if node.stage == "weighted_combine"
        )
        combine_ranks = {node.domain_ranks[0] for node in combine_nodes}
        if combine_ranks != attention_ranks or len(combine_nodes) != len(combine_ranks):
            raise ConfigurationError(
                "MoE weighted combine must execute once on every A tensor owner"
            )
        for node in combine_nodes:
            expected_domain = (
                routers[0].source_domain
                if node.operation == "f_weighted_combine"
                else a_domain
            )
            if node.source_domain != expected_domain:
                raise ConfigurationError("MoE combine is on the wrong protocol domain")
            if node.operation == "f_weighted_combine":
                if expected_domain == a_domain or len(combine_nodes) != 1:
                    raise ConfigurationError(
                        "native combine requires exactly one F owner"
                    )
                for transfer in runtime_nodes:
                    if transfer.operation != "af_f_to_a":
                        continue
                    pending = list(transfer.predecessor_ids)
                    seen = set()
                    while pending:
                        parent = pending.pop()
                        if parent not in seen:
                            seen.add(parent)
                            pending.extend(by_id[parent].predecessor_ids)
                    if node.node_id not in seen:
                        raise ConfigurationError(
                            "native hidden return must depend on F combine"
                        )
        expected_input_edges = len(expected_edges) + (
            original_tokens if shared_nodes else 0
        )
        for node in combine_nodes:
            work = node.local_work
            if work is None or (
                work.original_token_count != original_tokens
                or work.assignment_count != len(expected_edges)
                or work.input_edge_count != expected_input_edges
                or work.output_token_count != original_tokens
                or len(work.routing_edges) != len(expected_edges)
                or set(work.routing_edges) != expected_edge_set
            ):
                raise ConfigurationError(
                    "MoE weighted combine must consume all routed/shared inputs and "
                    f"produce the original token count on node={node.node_id!r}"
                )

    @property
    def final_node_id(self) -> str:
        if len(self.terminal_node_ids) != 1:
            raise ConfigurationError(
                "AF execution DAG has multiple terminal nodes; use terminal_node_ids"
            )
        return self.terminal_node_ids[0]


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
    expert_parallel_mode: Literal["expert_tp", "rank_reuse"]

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


class ParallelExecutionPlanner(ConfigurationOwner):
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
        from TokenSim.af.templates import TemplateCache

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
            from TokenSim.af.estimate_cache import identity

            if identity(link) != identity(configured):
                raise ConfigurationError(
                    "AF link configuration changed; create a new runtime"
                )
            link = configured
        formal_runtime = (
            self.runtime_plan is not None
            and (block.phase == "decode" or self.runtime_plan.layers[0].mla is not None)
            and block.placement is not None
            and not block.placement.a_domain.startswith("implicit:")
            and not block.placement.f_domain.startswith("implicit:")
        )
        if formal_runtime:
            if self.cluster_config.af.dag_template_cache:
                return self.template_cache.build(self, block, layer_id, link)
            return self._build_runtime_layer(block, layer_id, link)
        return self._build_legacy_layer(block, layer_id, link)

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

    def _build_runtime_layer(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
    ) -> AFExecutionDAG:
        if block.placement is None:
            raise ConfigurationError(
                f"AF block {block.block_id} has no preselected DP placement"
            )
        layer = self.runtime_plan.layer_plan(layer_id)
        snapshot = block.routing_snapshot_for_layer(layer_id)
        if layer.layer_type == "dense" and snapshot is not None:
            raise ConfigurationError(
                f"Dense AF layer {layer_id} must not contain routing state"
            )
        if layer.layer_type == "moe" and snapshot is None:
            raise ConfigurationError(
                f"MoE AF layer {layer_id} requires a frozen routing snapshot"
            )

        a_domain = self.cluster_config.execution_domains[block.placement.a_domain]
        f_domain = self.cluster_config.execution_domains[block.placement.f_domain]
        a_instance = self._instance(a_domain, block.placement.a_instance)
        f_instance = self._instance(f_domain, block.placement.f_instance)
        a_bindings = self.runtime_plan.bindings_for_instance(
            a_domain.name, a_instance.name
        )
        f_service_bindings = self.runtime_plan.bindings_for_instance(
            f_domain.name, f_instance.name
        )
        if (
            f_domain.capability == "af_ffn"
            and f_domain.parallel.expert_parallel_size > 1
        ):
            # Dense FFN and shared-expert work is replicated across the EP
            # ranks.  Keep one representative per base TP coordinate for
            # those stages; routed experts below still resolve their owning EP
            # rank through ``expert_execution``.
            f_by_tp: dict[int, Any] = {}
            for binding in sorted(
                f_service_bindings, key=lambda item: item.domain_rank
            ):
                f_by_tp.setdefault(binding.tp_rank, binding)
            f_base_bindings = tuple(binding for _, binding in sorted(f_by_tp.items()))
        else:
            f_base_bindings = f_service_bindings
        prefix = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}"
        nodes: list[ExecutionNode] = []

        attention_ids: dict[int, str] = {}
        for binding in a_bindings:
            node_id = f"{prefix}:attention:r{binding.domain_rank}"
            attention_ids[binding.domain_rank] = node_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="attention",
                    stage="attention",
                    predecessors=(),
                    binding=binding,
                    layer_type=layer.layer_type,
                    layer_id=layer_id,
                    token_count=block.token_count,
                    original_token_count=block.token_count,
                    output_token_count=block.token_count,
                )
            )
        a_collective = self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:attention:collective",
            domain=a_domain,
            dp_rank=block.placement.a_dp_rank,
            bindings=a_bindings,
            producer_ids=attention_ids,
            after_stage="a_compute",
            token_count=block.token_count,
            layer_type=layer.layer_type,
            required=len(a_bindings) > 1,
            producer_layout="partial",
            required_target_layout=(
                "replicated" if layer.layer_type == "moe" else None
            ),
        )

        # DeepSeek-V3 applies one RMSNorm after attention and before the AF
        # transfer/router.  Keep it as an explicit A-side node so its service
        # time contends for the same worker and appears in the trace.
        post_attention_ids: dict[int, str] = {}
        post_attention_predecessors = (
            (a_collective.node_id,)
            if a_collective is not None
            else tuple(attention_ids.values())
        )
        for binding in a_bindings:
            node_id = f"{prefix}:post_attention_norm:r{binding.domain_rank}"
            post_attention_ids[binding.domain_rank] = node_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="post_attention_norm",
                    stage="layout",
                    predecessors=post_attention_predecessors,
                    binding=binding,
                    layer_type=layer.layer_type,
                    layer_id=layer_id,
                    token_count=block.token_count,
                    original_token_count=block.token_count,
                    output_token_count=block.token_count,
                    shape={
                        "tokens": block.token_count,
                        "hidden_size": self.hidden_size,
                    },
                )
            )

        if layer.layer_type == "dense":
            dag = self._build_runtime_dense_tail(
                block=block,
                layer_id=layer_id,
                layer_type=layer.layer_type,
                link=link,
                prefix=prefix,
                nodes=nodes,
                a_domain=a_domain,
                f_domain=f_domain,
                a_bindings=a_bindings,
                f_bindings=f_base_bindings,
                attention_ids=post_attention_ids,
                a_collective=a_collective,
            )
        else:
            dag = self._build_runtime_moe_tail(
                block=block,
                layer_id=layer_id,
                layer=layer,
                snapshot=snapshot,
                link=link,
                prefix=prefix,
                nodes=nodes,
                a_domain=a_domain,
                f_domain=f_domain,
                a_bindings=a_bindings,
                f_service_bindings=f_service_bindings,
                attention_ids=post_attention_ids,
                a_collective=a_collective,
            )
        for node in dag.nodes:
            self.runtime_plan.validate_execution_node(node)
        return dag

    def _build_runtime_dense_tail(
        self,
        *,
        block: AFBlock,
        layer_id: int,
        layer_type: str,
        link: AFLinkConfig,
        prefix: str,
        nodes: list[ExecutionNode],
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
        a_bindings: tuple[Any, ...],
        f_bindings: tuple[Any, ...],
        attention_ids: Mapping[int, str],
        a_collective: ExecutionNode | None,
    ) -> AFExecutionDAG:
        tokens = self._block_tokens(block)
        dense_ids = {
            binding.domain_rank: f"{prefix}:dense:r{binding.domain_rank}"
            for binding in f_bindings
        }
        a_to_f_copies = self._activation_copies(
            tokens=tokens,
            a_bindings=a_bindings,
            target_specs=tuple(
                (binding, dense_ids[binding.domain_rank], tokens)
                for binding in f_bindings
            ),
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
            copies=a_to_f_copies,
            link=link,
            layer_type=layer_type,
            layer_id=layer_id,
        )

        dense_compute_ids: dict[int, str] = {}
        for binding in f_bindings:
            node_id = dense_ids[binding.domain_rank]
            dense_compute_ids[binding.domain_rank] = node_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="dense_ffn",
                    stage="dense_ffn",
                    predecessors=arrivals[node_id],
                    binding=binding,
                    layer_type=layer_type,
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    output_token_count=len(tokens),
                )
            )
        f_collective = self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:dense:collective",
            domain=f_domain,
            dp_rank=block.placement.f_dp_rank,
            bindings=f_bindings,
            producer_ids=dense_compute_ids,
            after_stage="expert_compute",
            token_count=len(tokens),
            layer_type=layer_type,
            required=len(f_bindings) > 1,
            producer_layout="partial",
        )

        residual_ids = {
            binding.domain_rank: f"{prefix}:residual:r{binding.domain_rank}"
            for binding in a_bindings
        }
        f_to_a_copies = self._output_copies(
            items=tuple(
                (token, f"request:{token[0]}:token:{token[1]}:dense", None, None)
                for token in tokens
            ),
            source_bindings=f_bindings,
            producer_ids=dense_compute_ids,
            collective=f_collective,
            target_bindings=a_bindings,
            consumer_ids=residual_ids,
            tensor_layout="replicated",
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).f_to_a_dtype,
        )
        returned = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:f_to_a",
            operation="af_f_to_a",
            direction="f_to_a",
            copies=f_to_a_copies,
            link=link,
            layer_type=layer_type,
            layer_id=layer_id,
        )
        residual_nodes: dict[int, str] = {}
        for binding in a_bindings:
            node_id = residual_ids[binding.domain_rank]
            residual_nodes[binding.domain_rank] = node_id
            slices = tuple(
                copy.tensor_slice
                for copy in f_to_a_copies
                if copy.target_binding.domain_rank == binding.domain_rank
            )
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="residual",
                    stage="residual",
                    predecessors=returned[node_id],
                    binding=binding,
                    layer_type=layer_type,
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    output_token_count=len(tokens),
                    tensor_slices=slices,
                    source_layout="sharded",
                    target_layout="sharded",
                )
            )
        self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:residual:layout",
            domain=a_domain,
            dp_rank=block.placement.a_dp_rank,
            bindings=a_bindings,
            producer_ids=residual_nodes,
            after_stage="residual",
            token_count=len(tokens),
            layer_type=layer_type,
            required=len(a_bindings) > 1,
            producer_layout="sharded",
            required_target_layout="replicated",
        )
        return AFExecutionDAG(tuple(nodes))

    def _build_runtime_moe_tail(
        self,
        *,
        block: AFBlock,
        layer_id: int,
        layer: Any,
        snapshot: Any,
        link: AFLinkConfig,
        prefix: str,
        nodes: list[ExecutionNode],
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
        a_bindings: tuple[Any, ...],
        f_service_bindings: tuple[Any, ...],
        attention_ids: Mapping[int, str],
        a_collective: ExecutionNode | None,
    ) -> AFExecutionDAG:
        tokens = tuple(snapshot.original_tokens)
        if self.cluster_config.af.execution_protocol == "vllm_afd":
            return self._build_native_moe_tail(
                layer_id=layer_id,
                layer=layer,
                snapshot=snapshot,
                link=link,
                prefix=prefix,
                nodes=nodes,
                a_bindings=a_bindings,
                f_bindings=f_service_bindings,
                attention_ids=attention_ids,
                a_collective=a_collective,
            )
        a_leader = min(a_bindings, key=lambda item: item.tp_rank)
        # ``attention_ids`` is the post-attention norm frontier in the MLA
        # runtime.  Using the collective directly here would leave those norm
        # nodes as orphan terminals when TP>1.
        a_ready = tuple(attention_ids.values())
        routing_edges = tuple(
            self._routing_edge(assignment) for assignment in snapshot.assignments
        )
        router_id = f"{prefix}:router"
        router_on_f = self.cluster_config.af.router_location == "f"
        if router_on_f and f_domain.parallel.data_parallel_size != 1:
            raise ConfigurationError("F-side leader Router requires one F DP replica")
        if (
            not router_on_f
            and self.cluster_config.af.transport_router_logits
            and (len(a_bindings) != 1 or len(f_service_bindings) != 1)
        ):
            raise ConfigurationError(
                "Transporting A-side Router logits requires one A rank and one F service rank"
            )
        router_binding = f_service_bindings[0] if router_on_f else a_leader
        router_node = self._runtime_compute_node(
            node_id=router_id,
            operation="router_topk",
            stage="router",
            predecessors=a_ready,
            binding=router_binding,
            layer_type="moe",
            layer_id=layer_id,
            token_count=len(tokens),
            original_token_count=len(tokens),
            assignment_count=len(snapshot.assignments),
            output_token_count=len(tokens),
            expert_ids=tuple(expert for expert, _ in snapshot.expert_histogram),
            shape={
                "tokens": len(tokens),
                "hidden_size": self.hidden_size,
                "num_experts": layer.num_experts,
                "top_k": layer.top_k,
            },
            routing_edges=routing_edges,
        )
        if not router_on_f:
            nodes.append(router_node)

        assignments_by_expert: dict[int, list[Any]] = {}
        for assignment in snapshot.assignments:
            assignments_by_expert.setdefault(assignment.expert_id, []).append(
                assignment
            )
        routed_specs: list[tuple[Any, str, tuple[tuple[int, int], ...]]] = []
        routed_edges_by_consumer: dict[str, tuple[AFRoutingEdge, ...]] = {}
        routed_nodes: list[
            tuple[int, tuple[Any, ...], dict[int, str], tuple[Any, ...]]
        ] = []
        for expert_id, assignments in sorted(assignments_by_expert.items()):
            bindings = self.runtime_plan.expert_execution(
                f_domain.name,
                expert_id,
                block.placement.f_dp_rank,
            )
            ids = {
                binding.domain_rank: (
                    f"{prefix}:routed:e{expert_id}:r{binding.domain_rank}"
                )
                for binding in bindings
            }
            expert_tokens = tuple(
                (assignment.request_id, assignment.token_id)
                for assignment in assignments
            )
            for binding in bindings:
                routed_specs.append((binding, ids[binding.domain_rank], expert_tokens))
                routed_edges_by_consumer[ids[binding.domain_rank]] = tuple(
                    self._routing_edge(assignment) for assignment in assignments
                )
            routed_nodes.append((expert_id, bindings, ids, tuple(assignments)))

        shared_ids: dict[int, str] = {}
        shared_specs: list[tuple[Any, str, tuple[tuple[int, int], ...]]] = []
        if layer.num_shared_experts > 0:
            shared_bindings = self.runtime_plan.component_execution(
                f_domain.name,
                layer_id,
                "shared_expert",
                block.placement.f_dp_rank,
            )
            shared_ids = {
                binding.domain_rank: f"{prefix}:shared:r{binding.domain_rank}"
                for binding in shared_bindings
            }
            shared_specs.extend(
                (binding, shared_ids[binding.domain_rank], tokens)
                for binding in shared_bindings
            )
        else:
            shared_bindings = ()

        # With an F-side leader Router, every F rank receives the full block
        # independently of the sampled routing. Selecting remote destinations
        # before the Router finishes would grant the scheduler free lookahead.
        multi_f_router = router_on_f and (
            len(f_service_bindings) > 1 or len(a_bindings) > 1
        )
        activation_specs = (
            tuple(
                (binding, f"{prefix}:f_input:r{binding.domain_rank}", tokens)
                for binding in f_service_bindings
            )
            if multi_f_router
            else tuple(routed_specs + shared_specs)
        )
        a_to_f_copies = self._activation_copies(
            tokens=tokens,
            a_bindings=a_bindings,
            target_specs=activation_specs,
            producer_ids=attention_ids,
            collective=a_collective,
            barrier_predecessor=None if router_on_f else router_id,
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype,
            routing_edges_by_consumer=(
                None if multi_f_router else routed_edges_by_consumer
            ),
        )
        arrivals = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:a_to_f",
            operation="af_a_to_f",
            direction="a_to_f",
            copies=a_to_f_copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )

        if not router_on_f and self.cluster_config.af.transport_router_logits:
            dtype = self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype
            target = f_service_bindings[0]
            logits_copies = tuple(
                _TransferCopy(
                    source_binding=a_leader,
                    target_binding=target,
                    tensor_slice=AFTensorSlice(
                        tensor_item_id=f"request:{request}:token:{token}:router_logits",
                        interval=(0, layer.num_experts),
                        layout="replicated",
                        owner_domain=a_leader.domain,
                        owner_domain_rank=a_leader.domain_rank,
                        owner_dp_rank=a_leader.dp_rank,
                        owner_tp_rank=a_leader.tp_rank,
                        owner_physical_worker=a_leader.physical_identity,
                        canonical_producer=router_id,
                        element_count=layer.num_experts,
                        payload_bytes=layer.num_experts
                        * dtype_size_bytes(dtype, role="a_to_f_dtype"),
                    ),
                    predecessor_id=router_id,
                    consumer_id="expert_logits",
                    source_layout="replicated",
                    target_layout="replicated",
                )
                for request, token in tokens
            )
            logits_arrivals = self._append_transfer_stage(
                nodes,
                logical_stage_id=f"{prefix}:router_logits",
                operation="af_router_logits",
                direction="a_to_f",
                copies=logits_copies,
                link=link,
                layer_type="moe",
                layer_id=layer_id,
            )
            arrivals = {
                name: tuple(dict.fromkeys((*preds, *logits_arrivals["expert_logits"])))
                for name, preds in arrivals.items()
            }
        combine_routing_ready = {}
        if router_on_f:
            leader_arrivals = (
                arrivals[f"{prefix}:f_input:r{router_binding.domain_rank}"]
                if multi_f_router
                else tuple(
                    dict.fromkeys(
                        pred
                        for predecessors in arrivals.values()
                        for pred in predecessors
                    )
                )
            )
            router_node = replace(router_node, predecessor_ids=leader_arrivals)
            nodes.append(router_node)
            routing_ready = {router_binding.domain_rank: router_id}
            # A-side weighted Combine also needs the selected IDs/weights.
            for binding in (*f_service_bindings, *a_bindings):
                if (
                    binding.domain == f_domain.name
                    and binding.domain_rank == router_binding.domain_rank
                ):
                    continue
                node_id = (
                    f"{prefix}:routing_metadata:{binding.domain}:r{binding.domain_rank}"
                )
                source = self._endpoint(router_binding)
                target = self._endpoint(binding)
                nodes.append(
                    ExecutionNode(
                        node_id=node_id,
                        operation="af_router_metadata",
                        kind="communication",
                        predecessor_ids=(router_id,),
                        stage=None,
                        direction=(
                            "f_to_f" if binding.domain == f_domain.name else "f_to_a"
                        ),
                        source_domain=f_domain.name,
                        target_domain=binding.domain,
                        logical_participants=(
                            router_binding.instance,
                            binding.instance,
                        ),
                        domain_ranks=(router_binding.domain_rank, binding.domain_rank),
                        physical_workers=(
                            router_binding.physical_identity,
                            binding.physical_identity,
                        ),
                        dp_rank=binding.dp_rank,
                        network=link.network,
                        resource_group=link.resource_group,
                        resource_lane="shared" if link.duplex == "half" else "f_to_a",
                        # Explicit protocol: int32 expert IDs + float32 weights.
                        payload_bytes=len(tokens) * layer.top_k * 8,
                        layer_type="moe",
                        logical_stage_id=f"{prefix}:routing_metadata",
                        leg_id=node_id,
                        source_endpoint=source,
                        target_endpoint=target,
                        participant_endpoints=(source, target),
                        metadata={
                            "original_token_count": len(tokens),
                            "assignment_count": len(snapshot.assignments),
                        },
                    )
                )
                if binding.domain == f_domain.name:
                    routing_ready[binding.domain_rank] = node_id
                else:
                    combine_routing_ready[binding.domain_rank] = node_id
            if multi_f_router:
                arrivals = {
                    consumer: tuple(
                        dict.fromkeys(
                            (
                                *arrivals[f"{prefix}:f_input:r{binding.domain_rank}"],
                                routing_ready[binding.domain_rank],
                            )
                        )
                    )
                    for binding, consumer, _ in routed_specs + shared_specs
                }
            else:
                arrivals = {
                    name: tuple(dict.fromkeys((*preds, router_id)))
                    for name, preds in arrivals.items()
                }

        routed_outputs: list[_TransferCopy] = []
        combine_ids = {
            binding.domain_rank: f"{prefix}:combine:r{binding.domain_rank}"
            for binding in a_bindings
        }
        f_to_a_dtype = self.runtime_plan.dtypes_for_link(link.name).f_to_a_dtype
        for expert_id, bindings, ids, assignments in routed_nodes:
            producer_ids: dict[int, str] = {}
            for binding in bindings:
                node_id = ids[binding.domain_rank]
                producer_ids[binding.domain_rank] = node_id
                nodes.append(
                    self._runtime_compute_node(
                        node_id=node_id,
                        operation="routed_expert",
                        stage="routed_expert",
                        predecessors=arrivals[node_id],
                        binding=binding,
                        layer_type="moe",
                        layer_id=layer_id,
                        token_count=len(assignments),
                        original_token_count=len(
                            {(item.request_id, item.token_id) for item in assignments}
                        ),
                        assignment_count=len(assignments),
                        output_token_count=len(assignments),
                        expert_ids=(expert_id,),
                        expert_workload=((expert_id, len(assignments)),),
                        routing_edges=tuple(
                            self._routing_edge(assignment) for assignment in assignments
                        ),
                    )
                )
            collective = self._append_runtime_collective(
                nodes,
                node_id=f"{prefix}:routed:e{expert_id}:collective",
                domain=f_domain,
                dp_rank=bindings[0].dp_rank,
                bindings=bindings,
                producer_ids=producer_ids,
                after_stage="expert_compute",
                token_count=len(assignments),
                layer_type="moe",
                required=len(bindings) > 1,
                producer_layout="partial",
                routing_edges=tuple(
                    self._routing_edge(assignment) for assignment in assignments
                ),
            )
            items = tuple(
                (
                    (assignment.request_id, assignment.token_id),
                    f"request:{assignment.request_id}:token:{assignment.token_id}:"
                    f"routed:k{assignment.top_k_index}:expert:{expert_id}",
                    f"expert:{expert_id}:k{assignment.top_k_index}:"
                    f"weight:{assignment.routing_weight:.17g}",
                    self._routing_edge(assignment),
                )
                for assignment in assignments
            )
            routed_outputs.extend(
                self._output_copies(
                    items=items,
                    source_bindings=bindings,
                    producer_ids=producer_ids,
                    collective=collective,
                    target_bindings=a_bindings,
                    consumer_ids=combine_ids,
                    tensor_layout="partial",
                    transfer_dtype=f_to_a_dtype,
                )
            )

        shared_outputs: list[_TransferCopy] = []
        if shared_bindings:
            shared_compute_ids: dict[int, str] = {}
            for binding in shared_bindings:
                node_id = shared_ids[binding.domain_rank]
                shared_compute_ids[binding.domain_rank] = node_id
                nodes.append(
                    self._runtime_compute_node(
                        node_id=node_id,
                        operation="shared_expert",
                        stage="shared_expert",
                        predecessors=arrivals[node_id],
                        binding=binding,
                        layer_type="moe",
                        layer_id=layer_id,
                        token_count=len(tokens),
                        original_token_count=len(tokens),
                        output_token_count=len(tokens),
                    )
                )
            shared_collective = self._append_runtime_collective(
                nodes,
                node_id=f"{prefix}:shared:collective",
                domain=f_domain,
                dp_rank=block.placement.f_dp_rank,
                bindings=shared_bindings,
                producer_ids=shared_compute_ids,
                after_stage="expert_compute",
                token_count=len(tokens),
                layer_type="moe",
                required=len(shared_bindings) > 1,
                producer_layout="partial",
            )
            shared_outputs.extend(
                self._output_copies(
                    items=tuple(
                        (
                            token,
                            f"request:{token[0]}:token:{token[1]}:shared",
                            "shared",
                            None,
                        )
                        for token in tokens
                    ),
                    source_bindings=shared_bindings,
                    producer_ids=shared_compute_ids,
                    collective=shared_collective,
                    target_bindings=a_bindings,
                    consumer_ids=combine_ids,
                    tensor_layout="partial",
                    transfer_dtype=f_to_a_dtype,
                )
            )

        f_to_a_copies = tuple(routed_outputs + shared_outputs)
        returned = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:f_to_a",
            operation="af_f_to_a",
            direction="f_to_a",
            copies=f_to_a_copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        residual_ids: dict[int, str] = {}
        for binding in a_bindings:
            combine_id = combine_ids[binding.domain_rank]
            owner_slices = tuple(
                copy.tensor_slice
                for copy in f_to_a_copies
                if copy.target_binding.domain_rank == binding.domain_rank
            )
            nodes.append(
                self._runtime_compute_node(
                    node_id=combine_id,
                    operation="weighted_combine",
                    stage="weighted_combine",
                    predecessors=returned[combine_id]
                    + (
                        (combine_routing_ready[binding.domain_rank],)
                        if binding.domain_rank in combine_routing_ready
                        else ()
                    ),
                    binding=binding,
                    layer_type="moe",
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    assignment_count=len(snapshot.assignments),
                    input_edge_count=(
                        len(snapshot.assignments)
                        + (len(tokens) if shared_bindings else 0)
                    ),
                    output_token_count=len(tokens),
                    tensor_slices=owner_slices,
                    source_layout="partial",
                    target_layout="sharded",
                    routing_edges=routing_edges,
                )
            )
            residual_id = f"{prefix}:residual:r{binding.domain_rank}"
            residual_ids[binding.domain_rank] = residual_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=residual_id,
                    operation="residual",
                    stage="residual",
                    predecessors=(combine_id,),
                    binding=binding,
                    layer_type="moe",
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    output_token_count=len(tokens),
                    source_layout="sharded",
                    target_layout="sharded",
                )
            )
        self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:residual:layout",
            domain=a_domain,
            dp_rank=block.placement.a_dp_rank,
            bindings=a_bindings,
            producer_ids=residual_ids,
            after_stage="residual",
            token_count=len(tokens),
            layer_type="moe",
            required=len(a_bindings) > 1,
            producer_layout="sharded",
            required_target_layout="replicated",
        )
        if self.cluster_config.af.routed_expert_execution == "grouped":
            nodes = list(group_routed_expert_nodes(tuple(nodes)))
        return AFExecutionDAG(tuple(nodes))

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

    def _runtime_compute_node(
        self,
        *,
        node_id: str,
        operation: str,
        stage: ExecutionStage,
        predecessors: tuple[str, ...],
        binding: Any,
        layer_type: str,
        layer_id: int,
        token_count: int,
        original_token_count: int | None = None,
        assignment_count: int | None = None,
        input_edge_count: int | None = None,
        output_token_count: int | None = None,
        expert_ids: tuple[int, ...] = (),
        expert_workload: tuple[tuple[int, int], ...] = (),
        tensor_slices: tuple[AFTensorSlice, ...] = (),
        source_layout: str | None = None,
        target_layout: str | None = None,
        routing_edges: tuple[AFRoutingEdge, ...] = (),
        source_dtype: str | None = None,
        target_dtype: str | None = None,
        shape: Mapping[str, int] | None = None,
    ) -> ExecutionNode:
        local_shape = dict(shape or {})
        if local_shape and self.runtime_plan is not None:
            base_shape = self.runtime_plan.local_shape(
                binding.domain, binding.domain_rank, layer_id, operation
            )
            for name in ("global_hidden_size", "tp_size"):
                local_shape.setdefault(name, base_shape.get(name, self.hidden_size))
        if not local_shape and operation in {
            "attention",
            "router",
            "dense_ffn",
            "routed_expert",
            "shared_expert",
        }:
            local_shape = self.runtime_plan.local_shape(
                binding.domain,
                binding.domain_rank,
                layer_id,
                operation,
            )
            local_shape["tokens"] = token_count
        if not local_shape and operation in {
            "weighted_combine",
            "residual",
            "layout",
        }:
            owner = self._ownership(binding)
            local_shape = {
                "tokens": token_count,
                "global_hidden_size": self.hidden_size,
                "hidden_size": owner.interval.size,
                "elements": token_count * owner.interval.size,
            }
        if not local_shape:
            local_shape = {
                "tokens": token_count,
                "hidden_size": self.runtime_plan.local_shape(
                    binding.domain,
                    binding.domain_rank,
                    layer_id,
                    (
                        "attention"
                        if binding.capability == "af_attention"
                        else "dense_ffn"
                    ),
                )["hidden_size"],
            }
        local_shape.setdefault("global_hidden_size", self.hidden_size)
        local_shape.setdefault("tp_size", max(1, len(binding.tp_group)))
        work = RankLocalWork(
            token_count=token_count,
            tensor_shape=tuple(sorted(local_shape.items())),
            expert_ids=expert_ids,
            expert_workload=expert_workload,
            original_token_count=original_token_count,
            assignment_count=assignment_count,
            input_edge_count=input_edge_count,
            output_token_count=output_token_count,
            tensor_slices=tensor_slices,
            owner_endpoint=self._endpoint(binding),
            source_layout=source_layout,
            target_layout=target_layout,
            routing_edges=routing_edges,
            source_dtype=source_dtype,
            target_dtype=target_dtype,
        )
        return ExecutionNode(
            node_id=node_id,
            operation=operation,
            kind="compute",
            predecessor_ids=tuple(dict.fromkeys(predecessors)),
            stage=stage,
            direction=None,
            source_domain=binding.domain,
            target_domain=binding.domain,
            logical_participants=(binding.instance,),
            domain_ranks=(binding.domain_rank,),
            physical_workers=(binding.physical_identity,),
            dp_rank=binding.dp_rank,
            tp_rank=binding.tp_rank,
            ep_rank=binding.ep_rank,
            tp_group=(
                (binding.domain_rank,)
                if stage == "routed_expert"
                and self.runtime_plan.domain_runtime(
                    binding.domain
                ).expert_parallel_mode
                == "rank_reuse"
                else binding.tp_group
            ),
            ep_group=binding.ep_group,
            local_work=work,
            layer_type=layer_type,
            source_endpoint=self._endpoint(binding),
            target_endpoint=self._endpoint(binding),
            tensor_slices=tensor_slices,
            source_layout=source_layout,
            target_layout=target_layout,
            routing_edges=routing_edges,
        )

    def _append_runtime_collective(
        self,
        nodes: list[ExecutionNode],
        *,
        node_id: str,
        domain: ExecutionDomainConfig,
        dp_rank: int,
        bindings: tuple[Any, ...],
        producer_ids: Mapping[int, str],
        after_stage: str,
        token_count: int,
        layer_type: str,
        required: bool,
        producer_layout: AFTensorLayout,
        required_target_layout: AFTensorLayout | None = None,
        routing_edges: tuple[AFRoutingEdge, ...] = (),
    ) -> ExecutionNode | None:
        if len(bindings) <= 1:
            return None
        plan = self.runtime_plan.collective_plan(domain.name, dp_rank, after_stage)
        if plan is None:
            if required:
                raise ConfigurationError(
                    f"AF layer requires collective after_stage={after_stage!r} for "
                    f"domain={domain.name!r}, dp_rank={dp_rank}, but no executable "
                    "collective plan is configured"
                )
            return None
        by_rank = {binding.domain_rank: binding for binding in bindings}
        try:
            participants = tuple(by_rank[rank] for rank in plan.participant_ranks)
        except KeyError as exc:
            raise ConfigurationError(
                f"collective {plan.operation!r} for domain={domain.name!r}, "
                f"dp_rank={dp_rank} references rank outside the execution branch"
            ) from exc
        if required and set(plan.participant_ranks) != set(by_rank):
            raise ConfigurationError(
                f"collective {plan.operation!r} after_stage={after_stage!r} for "
                f"domain={domain.name!r}, dp_rank={dp_rank} performs a required "
                "layout transition and must cover the complete execution branch: "
                f"expected_ranks={sorted(by_rank)}, "
                f"participant_ranks={list(plan.participant_ranks)}"
            )
        activation_bytes = dtype_size_bytes(
            self.runtime_plan.effective_dtypes.activation_dtype,
            role="activation_dtype",
        )
        local_width = self.hidden_size // max(1, len(bindings))
        full_bytes = token_count * self.hidden_size * activation_bytes
        local_bytes = token_count * local_width * activation_bytes
        transitions = {
            "all_gather": ("sharded", "replicated"),
            "reduce_scatter": ("partial", "sharded"),
            "all_reduce": ("partial", "replicated"),
        }
        source_layout, target_layout = transitions[plan.operation]
        if source_layout != producer_layout:
            raise ConfigurationError(
                f"collective {plan.operation!r} after_stage={after_stage!r} for "
                f"domain={domain.name!r}, dp_rank={dp_rank} expects "
                f"source_layout={source_layout!r}, but producer emits "
                f"{producer_layout!r}"
            )
        if (
            required_target_layout is not None
            and target_layout != required_target_layout
        ):
            raise ConfigurationError(
                f"collective {plan.operation!r} after_stage={after_stage!r} for "
                f"domain={domain.name!r}, dp_rank={dp_rank} produces "
                f"target_layout={target_layout!r}, but downstream requires "
                f"{required_target_layout!r}"
            )
        if plan.operation == "all_gather":
            input_bytes, output_bytes = local_bytes, full_bytes
        elif plan.operation == "reduce_scatter":
            input_bytes, output_bytes = full_bytes, local_bytes
        else:
            input_bytes = output_bytes = full_bytes
        node = ExecutionNode(
            node_id=node_id,
            operation=plan.operation,
            kind="communication",
            predecessor_ids=tuple(
                producer_ids[item.domain_rank] for item in participants
            ),
            input_producer_ids=tuple(
                (item.physical_identity, producer_ids[item.domain_rank])
                for item in participants
            ),
            stage=None,
            direction=plan.operation,
            source_domain=domain.name,
            target_domain=domain.name,
            logical_participants=tuple(
                dict.fromkeys(item.instance for item in participants)
            ),
            domain_ranks=plan.participant_ranks,
            physical_workers=plan.physical_workers,
            dp_rank=dp_rank,
            tp_group=plan.participant_ranks,
            network=plan.network,
            resource_group=plan.resource_group,
            resource_lane=plan.resource_lane,
            payload_bytes=(
                plan.payload_bytes if plan.payload_bytes is not None else input_bytes
            ),
            logical_input_bytes_per_rank=tuple(
                (item.domain_rank, input_bytes) for item in participants
            ),
            logical_output_bytes_per_rank=tuple(
                (item.domain_rank, output_bytes) for item in participants
            ),
            layer_type=layer_type,
            logical_stage_id=node_id,
            participant_endpoints=tuple(self._endpoint(item) for item in participants),
            source_layout=source_layout,
            target_layout=target_layout,
            metadata={
                "after_stage": after_stage,
                "collective_input_binding_required": True,
                "original_token_count": (
                    len({(edge.request_id, edge.token_id) for edge in routing_edges})
                    if routing_edges
                    else token_count
                ),
                "assignment_count": (len(routing_edges) if routing_edges else None),
            },
            routing_edges=routing_edges,
        )
        nodes.append(node)
        return node

    def _activation_copies(
        self,
        *,
        tokens: tuple[tuple[int, int], ...],
        a_bindings: tuple[Any, ...],
        target_specs: tuple[tuple[Any, str, tuple[tuple[int, int], ...]], ...],
        producer_ids: Mapping[int, str],
        collective: ExecutionNode | None,
        barrier_predecessor: str | None,
        transfer_dtype: str,
        routing_edges_by_consumer: (
            Mapping[str, tuple[AFRoutingEdge, ...]] | None
        ) = None,
    ) -> tuple[_TransferCopy, ...]:
        a_ownerships = self.runtime_plan.ownerships_for_replica(
            a_bindings[0].domain, a_bindings[0].dp_rank
        )
        source_by_rank = {item.domain_rank: item for item in a_bindings}
        ownership_by_rank = {item.domain_rank: item for item in a_ownerships}
        source_layout = (
            collective.target_layout
            if collective is not None
            else "replicated" if len(a_bindings) == 1 else "sharded"
        )
        if source_layout not in {"sharded", "replicated"}:
            raise ConfigurationError(
                "AF FFN input requires a sharded or replicated A activation, "
                f"got source_layout={source_layout!r}"
            )
        replicated = source_layout == "replicated"
        canonical_candidates = (
            tuple(
                item
                for item in a_bindings
                if item.domain_rank in collective.domain_ranks
            )
            if replicated and collective is not None
            else a_bindings
        )
        if not canonical_candidates:
            raise ConfigurationError("AF activation has no canonical source candidate")
        canonical = min(canonical_candidates, key=lambda item: item.domain_rank)
        eligible_ranks = (
            set(collective.domain_ranks)
            if collective is not None
            else set(source_by_rank)
        )
        sharded_owners = tuple(
            item for item in a_ownerships if item.domain_rank in eligible_ranks
        )
        byte_width = dtype_size_bytes(transfer_dtype, role="a_to_f_dtype")
        copies: list[_TransferCopy] = []
        edges_by_consumer = dict(routing_edges_by_consumer or {})
        for target, consumer_id, consumer_tokens in target_specs:
            for token in consumer_tokens:
                matching_edges = tuple(
                    edge
                    for edge in edges_by_consumer.get(consumer_id, ())
                    if (edge.request_id, edge.token_id) == token
                )
                source_parts = (
                    (
                        (
                            canonical,
                            ownership_by_rank[canonical.domain_rank],
                            (0, self.hidden_size),
                        ),
                    )
                    if replicated
                    else tuple(
                        (
                            source_by_rank[a_owner.domain_rank],
                            a_owner,
                            (a_owner.interval.start, a_owner.interval.end),
                        )
                        for a_owner in sharded_owners
                    )
                )
                for source, a_owner, interval in source_parts:
                    predecessor = barrier_predecessor or (
                        collective.node_id
                        if collective is not None
                        else producer_ids[source.domain_rank]
                    )
                    tensor_slice = AFTensorSlice(
                        tensor_item_id=f"request:{token[0]}:token:{token[1]}:activation",
                        interval=interval,
                        layout=source_layout,
                        owner_domain=a_owner.domain,
                        owner_domain_rank=a_owner.domain_rank,
                        owner_dp_rank=a_owner.dp_rank,
                        owner_tp_rank=a_owner.tp_rank,
                        owner_physical_worker=a_owner.physical_worker,
                        canonical_producer=(
                            canonical.physical_identity if replicated else None
                        ),
                        element_count=interval[1] - interval[0],
                        payload_bytes=(interval[1] - interval[0]) * byte_width,
                    )
                    copies.append(
                        _TransferCopy(
                            source,
                            target,
                            tensor_slice,
                            predecessor,
                            consumer_id,
                            source_layout,
                            "replicated",
                            matching_edges,
                        )
                    )
        self._validate_copy_coverage(
            copies,
            expected_intervals={
                (
                    consumer_id,
                    target.domain_rank,
                    f"request:{token[0]}:token:{token[1]}:activation",
                    None,
                ): (0, self.hidden_size)
                for target, consumer_id, consumer_tokens in target_specs
                for token in consumer_tokens
            },
            preserve_source_partials=False,
            context="A-to-F FFN input",
        )
        return tuple(copies)

    def _output_copies(
        self,
        *,
        items: tuple[
            tuple[tuple[int, int], str, str | None, AFRoutingEdge | None], ...
        ],
        source_bindings: tuple[Any, ...],
        producer_ids: Mapping[int, str],
        collective: ExecutionNode | None,
        target_bindings: tuple[Any, ...],
        consumer_ids: Mapping[int, str],
        tensor_layout: str,
        transfer_dtype: str,
    ) -> tuple[_TransferCopy, ...]:
        source_layout = (
            collective.target_layout
            if collective is not None
            else "replicated" if len(source_bindings) == 1 else tensor_layout
        )
        replicated = source_layout == "replicated"
        eligible_sources = tuple(
            item
            for item in source_bindings
            if collective is None or item.domain_rank in collective.domain_ranks
        )
        canonical_candidates = (
            eligible_sources if collective is not None else source_bindings
        )
        if not canonical_candidates:
            raise ConfigurationError("AF output has no canonical source candidate")
        canonical = min(canonical_candidates, key=lambda item: item.domain_rank)
        source_owners = {
            item.domain_rank: self._ownership(item) for item in source_bindings
        }
        byte_width = dtype_size_bytes(transfer_dtype, role="f_to_a_dtype")
        copies: list[_TransferCopy] = []
        target_layout = "partial" if tensor_layout == "partial" else "sharded"
        expected_intervals: dict[tuple[object, ...], tuple[int, int]] = {}
        for _token, tensor_item_id, partial_id, routing_edge in items:
            for target in target_bindings:
                target_owner = self._ownership(target)
                sources = (canonical,) if replicated else eligible_sources
                for source in sources:
                    interval = target_owner.interval
                    if source_layout == "sharded":
                        interval = source_owners[
                            source.domain_rank
                        ].interval.intersection(target_owner.interval)
                        if interval is None:
                            continue
                    predecessor = (
                        collective.node_id
                        if collective is not None
                        else producer_ids[source.domain_rank]
                    )
                    tensor_slice = AFTensorSlice(
                        tensor_item_id=tensor_item_id,
                        interval=(interval.start, interval.end),
                        layout=source_layout,
                        owner_domain=target.domain,
                        owner_domain_rank=target.domain_rank,
                        owner_dp_rank=target.dp_rank,
                        owner_tp_rank=target.tp_rank,
                        owner_physical_worker=target.physical_identity,
                        partial_id=partial_id,
                        canonical_producer=(
                            canonical.physical_identity if replicated else None
                        ),
                        routing_weight=(
                            routing_edge.routing_weight
                            if routing_edge is not None
                            else None
                        ),
                        element_count=interval.size,
                        payload_bytes=interval.size * byte_width,
                    )
                    copies.append(
                        _TransferCopy(
                            source,
                            target,
                            tensor_slice,
                            predecessor,
                            consumer_ids[target.domain_rank],
                            source_layout,
                            target_layout,
                            (routing_edge,) if routing_edge is not None else (),
                        )
                    )
                coverage_key: tuple[object, ...] = (
                    consumer_ids[target.domain_rank],
                    target.domain_rank,
                    tensor_item_id,
                    partial_id,
                )
                expected_intervals[coverage_key] = (
                    target_owner.interval.start,
                    target_owner.interval.end,
                )
        self._validate_copy_coverage(
            copies,
            expected_intervals=expected_intervals,
            preserve_source_partials=source_layout == "partial",
            context="F-to-A output",
        )
        return tuple(copies)

    def _append_transfer_stage(
        self,
        nodes: list[ExecutionNode],
        *,
        logical_stage_id: str,
        operation: str,
        direction: str,
        copies: tuple[_TransferCopy, ...],
        link: AFLinkConfig,
        layer_type: str,
        layer_id: int,
    ) -> dict[str, tuple[str, ...]]:
        if not copies:
            raise ConfigurationError(
                f"AF logical transfer stage {logical_stage_id!r} has no endpoint copies"
            )
        effective = self.runtime_plan.dtypes_for_link(link.name)
        activation_dtype = effective.activation_dtype
        transfer_dtype = (
            effective.a_to_f_dtype if direction == "a_to_f" else effective.f_to_a_dtype
        )
        needs_cast = activation_dtype != transfer_dtype

        records: dict[tuple[object, ...], dict[str, object]] = {}
        for copy in copies:
            key = (
                copy.source_binding.domain,
                copy.source_binding.domain_rank,
                copy.target_binding.domain,
                copy.target_binding.domain_rank,
                copy.tensor_slice.tensor_item_id,
                copy.tensor_slice.interval,
                copy.tensor_slice.partial_id,
                copy.source_layout,
                copy.target_layout,
            )
            record = records.setdefault(
                key,
                {
                    "copy": copy,
                    "predecessors": [],
                    "consumers": [],
                    "routing_edges": [],
                },
            )
            record["predecessors"].append(copy.predecessor_id)
            record["consumers"].append(copy.consumer_id)
            record["routing_edges"].extend(copy.routing_edges)

        encode_ids: dict[tuple[object, ...], str] = {}
        if needs_cast:
            encode_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
            for record in records.values():
                copy = record["copy"]
                encode_key = (
                    copy.source_binding.physical_identity,
                    copy.tensor_slice.tensor_item_id,
                    copy.tensor_slice.interval,
                    copy.tensor_slice.partial_id,
                    activation_dtype,
                    transfer_dtype,
                )
                encode_groups.setdefault(encode_key, []).append(record)
            for index, (encode_key, encode_records) in enumerate(encode_groups.items()):
                record = encode_records[0]
                copy = record["copy"]
                node_id = f"{logical_stage_id}:encode:{index}"
                encode_ids[encode_key] = node_id
                encode_edges = tuple(
                    dict.fromkeys(
                        edge
                        for item in encode_records
                        for edge in item["routing_edges"]
                    )
                )
                nodes.append(
                    self._runtime_compute_node(
                        node_id=node_id,
                        operation="encode",
                        stage="encode",
                        predecessors=tuple(
                            dict.fromkeys(
                                predecessor
                                for item in encode_records
                                for predecessor in item["predecessors"]
                            )
                        ),
                        binding=copy.source_binding,
                        layer_type=layer_type,
                        layer_id=layer_id,
                        token_count=1,
                        original_token_count=1,
                        output_token_count=1,
                        tensor_slices=(copy.tensor_slice,),
                        source_layout=copy.source_layout,
                        target_layout=copy.source_layout,
                        routing_edges=encode_edges,
                        source_dtype=activation_dtype,
                        target_dtype=transfer_dtype,
                        shape={
                            "tokens": 1,
                            "elements": copy.tensor_slice.element_count or 0,
                        },
                    )
                )

        grouped: dict[
            tuple[object, ...], list[tuple[tuple[object, ...], dict[str, object]]]
        ] = {}
        record_predecessors: dict[tuple[object, ...], tuple[str, ...]] = {}
        for record_key, record in records.items():
            copy = record["copy"]
            direct_predecessors = tuple(dict.fromkeys(record["predecessors"]))
            if len(direct_predecessors) != 1:
                raise ConfigurationError(
                    f"AF tensor copy in logical stage {logical_stage_id!r} must "
                    "resolve to exactly one data producer before transfer: "
                    f"item={copy.tensor_slice.tensor_item_id!r}, "
                    f"source_rank={copy.source_binding.domain_rank}, "
                    f"target_rank={copy.target_binding.domain_rank}, "
                    f"predecessors={direct_predecessors!r}"
                )
            if needs_cast:
                encode_key = (
                    copy.source_binding.physical_identity,
                    copy.tensor_slice.tensor_item_id,
                    copy.tensor_slice.interval,
                    copy.tensor_slice.partial_id,
                    activation_dtype,
                    transfer_dtype,
                )
                actual_predecessors = (encode_ids[encode_key],)
            else:
                actual_predecessors = direct_predecessors
            record_predecessors[record_key] = actual_predecessors
            endpoint_key = (
                copy.source_binding.domain,
                copy.source_binding.domain_rank,
                copy.target_binding.domain,
                copy.target_binding.domain_rank,
                copy.source_layout,
                copy.target_layout,
                actual_predecessors,
            )
            grouped.setdefault(endpoint_key, []).append((record_key, record))

        leg_for_record: dict[tuple[object, ...], str] = {}
        for leg_index, group in enumerate(grouped.values()):
            first = group[0][1]["copy"]
            leg_id = f"{logical_stage_id}:leg:{leg_index}"
            slices = tuple(record["copy"].tensor_slice for _, record in group)
            routing_edges = tuple(
                dict.fromkeys(
                    edge for _, record in group for edge in record["routing_edges"]
                )
            )
            predecessors = record_predecessors[group[0][0]]
            for record_key, _record in group:
                leg_for_record[record_key] = leg_id
            source_endpoint = self._endpoint(first.source_binding)
            target_endpoint = self._endpoint(first.target_binding)
            nodes.append(
                ExecutionNode(
                    node_id=leg_id,
                    operation=operation,
                    kind="communication",
                    predecessor_ids=predecessors,
                    stage=None,
                    direction=direction,
                    source_domain=first.source_binding.domain,
                    target_domain=first.target_binding.domain,
                    logical_participants=(
                        first.source_binding.instance,
                        first.target_binding.instance,
                    ),
                    domain_ranks=(
                        first.source_binding.domain_rank,
                        first.target_binding.domain_rank,
                    ),
                    physical_workers=(
                        first.source_binding.physical_identity,
                        first.target_binding.physical_identity,
                    ),
                    dp_rank=first.target_binding.dp_rank,
                    tp_rank=first.target_binding.tp_rank,
                    ep_rank=first.target_binding.ep_rank,
                    network=link.network,
                    resource_group=link.resource_group,
                    resource_lane=("shared" if link.duplex == "half" else direction),
                    payload_bytes=sum(item.payload_bytes or 0 for item in slices),
                    layer_type=layer_type,
                    logical_stage_id=logical_stage_id,
                    leg_id=leg_id,
                    source_endpoint=source_endpoint,
                    target_endpoint=target_endpoint,
                    participant_endpoints=(source_endpoint, target_endpoint),
                    tensor_slices=slices,
                    source_layout=first.source_layout,
                    target_layout=first.target_layout,
                    routing_edges=routing_edges,
                    metadata={
                        "original_token_count": (
                            len(
                                {
                                    (edge.request_id, edge.token_id)
                                    for edge in routing_edges
                                }
                            )
                            if routing_edges
                            else None
                        ),
                        "assignment_count": (
                            len(routing_edges) if routing_edges else None
                        ),
                        "payload_breakdown": tuple(
                            {
                                "tensor_item_id": item.tensor_item_id,
                                "interval": item.interval,
                                "partial_id": item.partial_id,
                                "routing_weight": item.routing_weight,
                                "payload_bytes": item.payload_bytes,
                            }
                            for item in slices
                        ),
                    },
                )
            )

        arrivals: dict[str, list[str]] = {}
        decode_ids: dict[tuple[object, ...], str] = {}
        for record_index, (record_key, record) in enumerate(records.items()):
            copy = record["copy"]
            leg_id = leg_for_record[record_key]
            arrival_id = leg_id
            if needs_cast:
                decode_key = (
                    copy.target_binding.physical_identity,
                    copy.tensor_slice.tensor_item_id,
                    copy.tensor_slice.interval,
                    copy.tensor_slice.partial_id,
                    transfer_dtype,
                    activation_dtype,
                )
                arrival_id = decode_ids.get(decode_key)
                if arrival_id is None:
                    arrival_id = f"{logical_stage_id}:decode:{record_index}"
                    decode_ids[decode_key] = arrival_id
                    nodes.append(
                        self._runtime_compute_node(
                            node_id=arrival_id,
                            operation="decode",
                            stage="decode",
                            predecessors=(leg_id,),
                            binding=copy.target_binding,
                            layer_type=layer_type,
                            layer_id=layer_id,
                            token_count=1,
                            original_token_count=1,
                            output_token_count=1,
                            tensor_slices=(copy.tensor_slice,),
                            source_layout=copy.target_layout,
                            target_layout=copy.target_layout,
                            routing_edges=tuple(dict.fromkeys(record["routing_edges"])),
                            source_dtype=transfer_dtype,
                            target_dtype=activation_dtype,
                            shape={
                                "tokens": 1,
                                "elements": copy.tensor_slice.element_count or 0,
                            },
                        )
                    )
            for consumer in dict.fromkeys(record["consumers"]):
                arrivals.setdefault(consumer, []).append(arrival_id)
        return {
            consumer: tuple(dict.fromkeys(node_ids))
            for consumer, node_ids in arrivals.items()
        }

    @staticmethod
    def _routing_edge(assignment: Any) -> AFRoutingEdge:
        return AFRoutingEdge(
            iteration_id=assignment.iteration_id,
            phase=assignment.phase,
            decode_step=assignment.decode_step,
            layer_id=assignment.layer_id,
            request_id=assignment.request_id,
            token_id=assignment.token_id,
            top_k_index=assignment.top_k_index,
            expert_id=assignment.expert_id,
            routing_weight=assignment.routing_weight,
            a_dp_rank=assignment.a_dp_rank,
            a_tp_rank=assignment.a_tp_rank,
        )

    @staticmethod
    def _validate_copy_coverage(
        copies: list[_TransferCopy],
        *,
        expected_intervals: Mapping[tuple[object, ...], tuple[int, int]],
        preserve_source_partials: bool,
        context: str,
    ) -> None:
        grouped: dict[tuple[object, ...], list[_TransferCopy]] = {}
        for copy in copies:
            key = (
                copy.consumer_id,
                copy.target_binding.domain_rank,
                copy.tensor_slice.tensor_item_id,
                copy.tensor_slice.partial_id,
            )
            grouped.setdefault(key, []).append(copy)

        for key, expected in expected_intervals.items():
            candidates = grouped.get(key, [])
            if not candidates:
                raise ConfigurationError(
                    f"{context} has no tensor slices for consumer={key[0]!r}, "
                    f"target_rank={key[1]}, item={key[2]!r}"
                )
            copy_groups = (
                {
                    copy.source_binding.domain_rank: tuple(
                        item
                        for item in candidates
                        if item.source_binding.domain_rank
                        == copy.source_binding.domain_rank
                    )
                    for copy in candidates
                }.values()
                if preserve_source_partials
                else (tuple(candidates),)
            )
            for copy_group in copy_groups:
                intervals = sorted(item.tensor_slice.interval for item in copy_group)
                cursor = expected[0]
                for start, end in intervals:
                    if start != cursor:
                        raise ConfigurationError(
                            f"{context} has a gap or overlap for consumer={key[0]!r}, "
                            f"target_rank={key[1]}, item={key[2]!r}: "
                            f"expected_start={cursor}, actual=[{start}, {end})"
                        )
                    cursor = end
                if cursor != expected[1]:
                    raise ConfigurationError(
                        f"{context} does not cover the required interval for "
                        f"consumer={key[0]!r}, target_rank={key[1]}, "
                        f"item={key[2]!r}: covered_end={cursor}, "
                        f"required_end={expected[1]}"
                    )

    def _ownership(self, binding: Any) -> Any:
        for ownership in self.runtime_plan.ownerships_for_replica(
            binding.domain, binding.dp_rank
        ):
            if ownership.domain_rank == binding.domain_rank:
                return ownership
        # In a rank-reuse F pool every EP rank keeps a replicated copy of the
        # residual tensor. ``ownerships_for_replica`` intentionally collapses
        # those identical intervals for coverage validation, but execution
        # still needs the rank-specific physical owner when constructing
        # output copies. Resolve that replicated record here.
        domain_plan = self.runtime_plan.domain_runtime(binding.domain)
        if (
            domain_plan.capability == "af_ffn"
            and domain_plan.expert_parallel_mode == "rank_reuse"
        ):
            for ownership in self.runtime_plan.tensor_ownerships:
                if (
                    ownership.domain == binding.domain
                    and ownership.domain_rank == binding.domain_rank
                    and ownership.dp_rank == binding.dp_rank
                ):
                    return ownership
        raise ConfigurationError(
            f"AF runtime has no tensor ownership for domain={binding.domain!r}, "
            f"rank={binding.domain_rank}"
        )

    @staticmethod
    def _endpoint(binding: Any) -> AFEndpoint:
        return AFEndpoint(
            domain=binding.domain,
            domain_rank=binding.domain_rank,
            physical_worker=binding.physical_identity,
            device_type=binding.hardware,
            dp_rank=binding.dp_rank,
            tp_rank=binding.tp_rank,
            ep_rank=binding.ep_rank,
        )

    @staticmethod
    def _block_tokens(block: AFBlock) -> tuple[tuple[int, int], ...]:
        if block.phase == "decode":
            return tuple(
                (request_id, max(0, context_len - 1))
                for request_id, context_len in zip(
                    block.request_ids, block.context_lengths
                )
            )
        tokens: list[tuple[int, int]] = []
        for request_id, context_len in zip(block.request_ids, block.context_lengths):
            tokens.extend((request_id, token_id) for token_id in range(context_len))
            if len(tokens) >= block.token_count:
                break
        if len(tokens) != block.token_count:
            raise ConfigurationError(
                f"AF block {block.block_id} cannot reconstruct {block.token_count} "
                "Prefill token identities from its context lengths"
            )
        return tuple(tokens)

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


# Import compatibility for callers migrating from the previous linear planner.
AFExecutionStep = ExecutionNode
AFExecutionPlanBuilder = ParallelExecutionPlanner


def group_routed_expert_nodes(
    nodes: tuple[ExecutionNode, ...]
) -> tuple[ExecutionNode, ...]:
    """Group the independent expert branches of one block/layer runtime DAG.

    Keep every routing edge and output transfer/collective. Only the compute
    execution boundary changes; the group waits for all of its inputs.
    """
    groups: dict[tuple, list[ExecutionNode]] = {}
    for node in nodes:
        if node.stage != "routed_expert" or node.kind != "compute":
            continue
        work = node.local_work
        if work is None:
            raise ConfigurationError("routed expert missing local work")
        # The prefix encodes iteration/block/layer. Domain/rank and layouts are
        # included even if two logical workers share a physical GPU.
        key = (
            node.node_id.split(":routed:")[0],
            node.physical_workers,
            node.source_domain,
            node.target_domain,
            node.domain_ranks,
            node.dp_rank,
            node.tp_rank,
            node.ep_rank,
            node.tp_group,
            node.ep_group,
            node.layer_type,
            node.logical_participants,
            work.source_dtype,
            work.target_dtype,
            work.source_layout,
            work.target_layout,
            tuple((k, v) for k, v in work.tensor_shape if k != "tokens"),
        )
        groups.setdefault(key, []).append(node)
    aliases = {}
    replacements = {}
    by_id = {node.node_id: node for node in nodes}
    for group in groups.values():
        if len(group) < 2:
            continue
        member_ids = {node.node_id for node in group}
        pending = [pred for node in group for pred in node.predecessor_ids]
        visited = set()
        while pending:
            pred = pending.pop()
            if pred in member_ids:
                raise ConfigurationError("cannot group dependent routed expert nodes")
            if pred not in visited:
                visited.add(pred)
                pending.extend(by_id[pred].predecessor_ids)
        first = group[0]
        edges = tuple(edge for node in group for edge in node.local_work.routing_edges)
        workload = tuple(
            sorted(pair for node in group for pair in node.local_work.expert_workload)
        )
        if len({expert for expert, _ in workload}) != len(workload):
            raise ConfigurationError("expert group contains duplicate expert workloads")
        count = sum(value for _, value in workload)
        shape = dict(first.local_work.tensor_shape)
        shape["tokens"] = count
        work = replace(
            first.local_work,
            token_count=count,
            tensor_shape=tuple(sorted(shape.items())),
            expert_ids=tuple(expert for expert, _ in workload),
            expert_workload=workload,
            assignment_count=count,
            original_token_count=len(
                {(edge.request_id, edge.token_id) for edge in edges}
            ),
            output_token_count=count,
            routing_edges=edges,
            tensor_slices=tuple(
                item for node in group for item in node.local_work.tensor_slices
            ),
        )
        group_id = (
            first.node_id.split(":routed:")[0]
            + ":routed:group:"
            + first.node_id.split(":routed:")[1]
        )
        aliases.update({node.node_id: group_id for node in group})
        replacements[first.node_id] = replace(
            first,
            node_id=group_id,
            local_work=work,
            routing_edges=edges,
            tensor_slices=tuple(item for node in group for item in node.tensor_slices),
            predecessor_ids=tuple(
                dict.fromkeys(pred for node in group for pred in node.predecessor_ids)
            ),
            metadata={
                **first.metadata,
                "execution_unit": "expert_group",
                "member_node_ids": tuple(node.node_id for node in group),
            },
        )
    result = []
    for node in nodes:
        if node.node_id in aliases and node.node_id not in replacements:
            continue
        node = replacements.get(node.node_id, node)
        result.append(
            replace(
                node,
                predecessor_ids=tuple(
                    dict.fromkeys(
                        aliases.get(pred, pred) for pred in node.predecessor_ids
                    )
                ),
            )
        )
    return tuple(result)
