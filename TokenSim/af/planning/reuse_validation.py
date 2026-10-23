"""Check tensor coverage and producer proofs for local dense replica reuse."""

from TokenSim.errors import ConfigurationError


def validate_dense_reuse(nodes):
    def require(condition, message):
        if not condition:
            raise ConfigurationError(f"colocated dense reuse: {message}")

    by_id = {node.node_id: node for node in nodes}
    # One attention node per rank; KV-parallel output nodes share its stage.
    attention = [node for node in nodes if node.operation == "attention"]
    reuse = [node for node in nodes if node.metadata.get("colocated_dense_reuse")]
    require(all(node.layer_type == "dense" for node in nodes), "requires dense work")
    require(
        not any(node.operation in {"af_a_to_f", "af_f_to_a"} for node in nodes),
        "cannot mix local reuse with endpoint transfers",
    )
    require(
        not any(node.operation == "all_gather" for node in nodes),
        "replicated residuals must not be gathered",
    )
    physical = {node.physical_workers[0] for node in attention}
    for stage in ("dense_ffn", "residual"):
        branch = [node for node in reuse if node.stage == stage]
        require(
            len(branch) == len(attention)
            and {node.physical_workers[0] for node in branch} == physical,
            f"{stage} must cover every physical attention rank exactly once",
        )
        require(
            all(node in branch for node in nodes if node.stage == stage),
            f"every {stage} consumer must declare local reuse",
        )
    require(len(reuse) == 2 * len(attention), "unexpected reuse consumer")
    token_sets = []
    for node in reuse:
        work = node.local_work
        require(node.kind == "compute" and work is not None, "requires local compute")
        require(len(node.predecessor_ids) == 1, "requires one complete producer")
        producer = by_id[node.predecessor_ids[0]]
        endpoint = node.physical_workers[0]
        shape = dict(work.tensor_shape)
        width = shape.get("global_hidden_size")
        require(
            width and shape.get("hidden_size") == width,
            "consumer must process the complete hidden vector",
        )
        require(
            node.source_layout == work.source_layout == "replicated",
            "consumer requires replicated input",
        )
        target = (
            "partial"
            if node.stage == "dense_ffn" and len(attention) > 1
            else "replicated"
        )
        require(
            node.target_layout == work.target_layout == target,
            "consumer emits an incompatible layout",
        )
        if node.stage == "dense_ffn":
            require(
                producer.operation == "post_attention_norm"
                and producer.physical_workers == (endpoint,),
                "FFN must consume its local normalized attention replica",
            )
            require(len(producer.predecessor_ids) == 1, "normalization needs one input")
            upstream = by_id[producer.predecessor_ids[0]]
            expected_stage = "attention"
        else:
            upstream = producer
            expected_stage = "dense_ffn"
        if len(attention) > 1:
            full_bytes = sum(item.payload_bytes for item in node.tensor_slices)
            require(
                upstream.operation == "all_reduce"
                and upstream.source_layout == "partial"
                and upstream.target_layout == "replicated"
                and set(upstream.physical_workers) == physical
                and upstream.payload_bytes == full_bytes
                and len(upstream.logical_output_bytes_per_rank) == len(attention)
                and all(
                    size == full_bytes
                    for _, size in upstream.logical_output_bytes_per_rank
                ),
                "replica requires a complete AllReduce",
            )
            inputs = [by_id[item] for item in upstream.predecessor_ids]
            require(
                len(inputs) == len(attention)
                and all(item.stage == expected_stage for item in inputs)
                and {item.physical_workers[0] for item in inputs} == physical
                and dict(upstream.input_producer_ids)
                == {item.physical_workers[0]: item.node_id for item in inputs},
                "AllReduce must depend on every matching partial producer",
            )
        else:
            require(
                upstream.stage == expected_stage
                and upstream.physical_workers == (endpoint,),
                "single-rank replica requires its local producer",
            )
        if producer.kind == "communication":
            owners = [
                item
                for item in producer.participant_endpoints
                if item.physical_worker == endpoint
            ]
            require(len(owners) == 1, "replica has no unique local collective owner")
            owner_domain, owner_rank = owners[0].domain, owners[0].domain_rank
        else:
            owner_domain, owner_rank = producer.source_domain, producer.domain_ranks[0]
        require(
            tuple(node.tensor_slices) == tuple(work.tensor_slices)
            and len(node.tensor_slices) == work.token_count,
            "slice count must match the consumed tokens",
        )
        items = set()
        for item in node.tensor_slices:
            require(
                item.layout == "replicated"
                and item.interval == (0, width)
                and item.element_count == width
                and item.owner_physical_worker == endpoint
                and item.canonical_producer == endpoint
                and item.owner_domain == owner_domain
                and item.owner_domain_rank == owner_rank,
                "slice must name the complete local producer replica",
            )
            require(item.tensor_item_id not in items, "duplicate token slice")
            items.add(item.tensor_item_id)
        token_sets.append({item.rsplit(":", 1)[0] for item in items})
    require(
        all(items == token_sets[0] for items in token_sets), "token coverage differs"
    )
