"""Planning grouping."""

from __future__ import annotations

from dataclasses import replace

from TokenSim.af.planning_nodes import ExecutionNode
from TokenSim.errors import ConfigurationError


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
