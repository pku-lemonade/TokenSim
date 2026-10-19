"""Batch transfers and collectives whose physical producers are already shared."""

from dataclasses import replace


def _rewrite(nodes, aliases):
    return tuple(
        replace(
            node,
            predecessor_ids=tuple(
                dict.fromkeys(aliases.get(key, key) for key in node.predecessor_ids)
            ),
            input_producer_ids=tuple(
                (endpoint, aliases.get(producer, producer))
                for endpoint, producer in node.input_producer_ids
            ),
        )
        for node in nodes
    )


def _merge(nodes, operations):
    groups = {}
    for node in nodes:
        if node.operation not in operations:
            continue
        key = (
            node.operation,
            node.predecessor_ids,
            node.physical_workers,
            node.source_domain,
            node.target_domain,
            node.source_layout,
            node.target_layout,
            node.network,
            node.resource_group,
            node.resource_lane,
            node.dp_rank,
        )
        groups.setdefault(key, []).append(node)
    aliases, replacements = {}, {}
    for group in groups.values():
        if len(group) < 2:
            continue
        first = group[0]
        node_id = first.node_id + ":batch"
        aliases.update({node.node_id: node_id for node in group})
        input_bytes, output_bytes = {}, {}
        for node in group:
            for rank, size in node.logical_input_bytes_per_rank:
                input_bytes[rank] = input_bytes.get(rank, 0) + size
            for rank, size in node.logical_output_bytes_per_rank:
                output_bytes[rank] = output_bytes.get(rank, 0) + size
        edges = tuple(
            dict.fromkeys(edge for node in group for edge in node.routing_edges)
        )
        replacements[first.node_id] = replace(
            first,
            node_id=node_id,
            payload_bytes=sum(node.payload_bytes or 0 for node in group),
            tensor_slices=tuple(item for node in group for item in node.tensor_slices),
            routing_edges=edges,
            logical_input_bytes_per_rank=tuple(sorted(input_bytes.items())),
            logical_output_bytes_per_rank=tuple(sorted(output_bytes.items())),
            metadata={
                **first.metadata,
                "coalesced_nodes": tuple(node.node_id for node in group),
                "original_token_count": (
                    len({(edge.request_id, edge.token_id) for edge in edges})
                    if edges
                    else first.metadata.get("original_token_count")
                ),
                "assignment_count": (
                    len(edges) if edges else first.metadata.get("assignment_count")
                ),
            },
        )
    result = tuple(
        replacements.get(node.node_id, node)
        for node in nodes
        if node.node_id not in aliases or node.node_id in replacements
    )
    return _rewrite(result, aliases)


def coalesce_ready_transfers(nodes):
    # Grouped expert compute makes disjoint expert outputs ready together.
    # Concatenated reductions retain per-expert tensor boundaries and total bytes.
    nodes = _merge(nodes, {"all_reduce", "reduce_scatter", "all_gather"})
    return _merge(nodes, {"af_f_to_a", "af_a_to_f"})
