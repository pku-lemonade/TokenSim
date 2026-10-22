"""Validate endpoint transfer stages and explicit local replica consumption."""

from TokenSim.errors import SimulationStateError


def validate_transfer_stages(events, by_node):
    local = [event for event in events if event.details.get("colocated_dense_reuse")]
    if local:
        validate_local_reuse(events, by_node, local)
        return
    for operation in ("af_a_to_f", "af_f_to_a"):
        stage_ids = {
            event.logical_stage_id or event.node_id
            for event in events
            if event.operation == operation
        }
        if len(stage_ids) != 1:
            event = events[0]
            raise SimulationStateError(
                f"AF trace invariant failed: block={event.block_id}, layer={event.layer_id} "
                f"requires exactly one {operation} logical stage; got {len(stage_ids)}"
            )


def validate_local_reuse(events, by_node, local):
    def require(condition, message):
        if not condition:
            raise SimulationStateError(
                f"AF trace local reuse invariant failed: {message}"
            )

    require(all(event.layer_type == "dense" for event in local), "requires dense work")
    require(
        not any(event.operation in {"af_a_to_f", "af_f_to_a"} for event in events),
        "cannot mix local reuse and endpoint copies",
    )
    attention = [event for event in events if event.stage == "attention"]
    physical = {event.physical_endpoints[0] for event in attention}
    layer_ids = {event.node_id for event in events}
    for stage in ("dense_ffn", "residual"):
        branch = [event for event in local if event.stage == stage]
        require(
            len(branch) == len(attention)
            and {event.physical_endpoints[0] for event in branch} == physical,
            f"{stage} must consume one replica on every attention worker",
        )
    require(len(local) == 2 * len(attention), "unexpected reuse consumer")
    for event in local:
        endpoint = event.physical_endpoints[0]
        width = event.local_work["tensor_shape"]["global_hidden_size"]
        require(
            event.source_layout == "replicated"
            and event.local_work["tensor_shape"]["hidden_size"] == width
            and len(event.tensor_slices) == event.token_count,
            "consumer must read complete replicas",
        )
        pending, ancestors = list(event.predecessor_ids), {}
        while pending:
            node_id = pending.pop()
            if node_id in layer_ids and node_id not in ancestors:
                ancestors[node_id] = by_node[node_id]
                pending.extend(by_node[node_id].predecessor_ids)
        items = set()
        for item in event.tensor_slices:
            require(
                item.interval == (0, width)
                and item.layout == "replicated"
                and item.owner_physical_worker == endpoint
                and item.canonical_producer == endpoint
                and item.tensor_item_id not in items,
                "slice must identify a unique complete local replica",
            )
            items.add(item.tensor_item_id)
            operation = (
                "post_attention_norm"
                if event.stage == "dense_ffn"
                else "all_reduce" if len(attention) > 1 else "dense_ffn"
            )
            producers = [
                source
                for source in ancestors.values()
                if source.operation == operation
                and source.source_domain == item.owner_domain
                and item.owner_domain_rank in source.participant_ranks
                and endpoint in source.physical_endpoints
            ]
            require(bool(producers), "local replica producer is not a dependency")
        if event.stage == "residual":
            require(
                event.target_layout == "replicated", "residual must retain replicas"
            )
