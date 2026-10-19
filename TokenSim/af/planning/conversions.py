"""Batch tensor conversions at shared producer and receive boundaries."""

from dataclasses import replace

from TokenSim.af.planning.coalesce import _merge, rewrite_dependencies


def _batch_stage(nodes, stage):
    groups = {}
    for node in nodes:
        if node.stage != stage:
            continue
        work = node.local_work
        key = (
            node.predecessor_ids,
            node.physical_workers,
            node.source_domain,
            node.domain_ranks,
            work.source_layout,
            work.target_layout,
            work.source_dtype,
            work.target_dtype,
            tuple(
                (k, v) for k, v in work.tensor_shape if k not in {"tokens", "elements"}
            ),
        )
        groups.setdefault(key, []).append(node)
    aliases, replacements = {}, {}
    for group in groups.values():
        if len(group) < 2:
            continue
        first = group[0]
        slices = tuple(dict.fromkeys(s for node in group for s in node.tensor_slices))
        edges = tuple(dict.fromkeys(e for node in group for e in node.routing_edges))
        shape = dict(first.local_work.tensor_shape)
        shape.update(tokens=len(slices), elements=sum(s.element_count for s in slices))
        work = replace(
            first.local_work,
            token_count=len(slices),
            tensor_shape=tuple(sorted(shape.items())),
            tensor_slices=slices,
            routing_edges=edges,
            original_token_count=(
                len({(e.request_id, e.token_id) for e in edges})
                if edges
                else len(slices)
            ),
            output_token_count=len(slices),
        )
        node_id = first.node_id + ":batch"
        aliases.update({node.node_id: node_id for node in group})
        replacements[first.node_id] = replace(
            first,
            node_id=node_id,
            local_work=work,
            tensor_slices=slices,
            routing_edges=edges,
            metadata={"conversion_members": tuple(n.node_id for n in group)},
        )
    return rewrite_dependencies(
        tuple(
            replacements.get(node.node_id, node)
            for node in nodes
            if node.node_id not in aliases or node.node_id in replacements
        ),
        aliases,
    )


def batch_conversions(nodes):
    nodes = _batch_stage(nodes, "encode")
    # Transfer slices sharing the same conversion result form one wire message.
    # Other boundaries retain their configured coalescing policy.
    encoded = {node.node_id for node in nodes if node.stage == "encode"}
    converted = tuple(
        node
        for node in nodes
        if node.operation in {"af_a_to_f", "af_f_to_a", "af_router_logits"}
        and set(node.predecessor_ids) & encoded
    )
    if converted:
        combined = _merge(converted, {node.operation for node in converted})
        by_id = {node.node_id: node for node in combined}
        aliases = {
            member: node.node_id
            for node in combined
            for member in node.metadata.get("coalesced_nodes", ())
        }
        converted_ids = {node.node_id for node in converted}
        nodes = rewrite_dependencies(
            tuple(
                by_id.get(aliases.get(node.node_id, node.node_id), node)
                for node in nodes
                if node.node_id not in converted_ids
                or node.node_id not in aliases
                or node.node_id
                == by_id[aliases[node.node_id]].metadata["coalesced_nodes"][0]
            ),
            aliases,
        )
    return _batch_stage(nodes, "decode")
