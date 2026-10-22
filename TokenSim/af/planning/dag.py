"""Planning dag."""

from __future__ import annotations

from dataclasses import dataclass

from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.af.planning.reuse_validation import validate_dense_reuse
from TokenSim.af.providers.queries import AFRoutingEdge
from TokenSim.errors import ConfigurationError


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
        local_reuse = any(
            node.metadata.get("colocated_dense_reuse") for node in runtime_nodes
        )
        if local_reuse:
            validate_dense_reuse(runtime_nodes)
        for operation in () if local_reuse else ("af_a_to_f", "af_f_to_a"):
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
        expected_ranks = (
            {routers[0].domain_ranks[0]}
            if combine_nodes
            and all(node.operation == "f_weighted_combine" for node in combine_nodes)
            else attention_ranks
        )
        if combine_ranks != expected_ranks or len(combine_nodes) != len(combine_ranks):
            raise ConfigurationError(
                "MoE weighted combine must execute once on every protocol owner"
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
