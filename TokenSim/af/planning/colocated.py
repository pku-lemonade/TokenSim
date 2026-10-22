"""Reuse equivalent replicated dense tensors across colocated logical domains."""

from dataclasses import replace

from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.providers.queries import AFTensorSlice
from TokenSim.dtypes import dtype_size_bytes


def dense_reuse_pairs(planner, link, a_bindings, f_bindings, a_collective):
    """Return physical pairs only when both branches retain complete replicas."""
    af = planner.cluster_config.af
    if not link.colocated_dense_reuse or af.execution_protocol != "partial_return":
        return ()
    dtypes = planner.runtime_plan.dtypes_for_link(link.name)
    if any(
        dtype != dtypes.activation_dtype
        for dtype in (dtypes.a_to_f_dtype, dtypes.f_to_a_dtype)
    ):
        return ()
    by_physical = {b.physical_identity: b for b in f_bindings}
    if (
        len(by_physical) != len(f_bindings)
        or len(a_bindings) != len(f_bindings)
        or len({b.physical_identity for b in a_bindings}) != len(a_bindings)
        or {b.physical_identity for b in a_bindings} != set(by_physical)
    ):
        return ()
    if len(a_bindings) > 1:
        if (
            a_collective is None
            or a_collective.operation != "all_reduce"
            or a_collective.target_layout != "replicated"
            or set(a_collective.domain_ranks) != {b.domain_rank for b in a_bindings}
        ):
            return ()
        binding = f_bindings[0]
        plan = planner.runtime_plan.collective_plan(
            binding.domain, binding.dp_rank, "expert_compute", binding.pp_rank
        )
        if plan is None or plan.operation != "all_reduce":
            return ()
        full_bytes = dict(a_collective.logical_output_bytes_per_rank)
        if (
            not full_bytes
            or any(size != a_collective.payload_bytes for size in full_bytes.values())
            or plan.payload_bytes not in (None, a_collective.payload_bytes)
        ):
            return ()
    return tuple((a, by_physical[a.physical_identity]) for a in a_bindings)


def replica_slices(planner, source, tokens, suffix, dtype):
    width = planner.hidden_size
    return tuple(
        AFTensorSlice(
            tensor_item_id=f"request:{request}:token:{token}:{suffix}",
            interval=(0, width),
            layout="replicated",
            owner_domain=source.domain,
            owner_domain_rank=source.domain_rank,
            owner_dp_rank=source.dp_rank,
            owner_tp_rank=source.tp_rank,
            owner_physical_worker=source.physical_identity,
            canonical_producer=source.physical_identity,
            element_count=width,
            payload_bytes=width * dtype_size_bytes(dtype, role="activation_dtype"),
        )
        for request, token in tokens
    )


def build_dense_reuse(
    planner, *, block, layer_id, link, prefix, nodes, f_domain, pairs, attention_ids
):
    """Keep full residual replicas and bind consumers directly to local producers."""
    tokens = planner._block_tokens(block)
    dtype = planner.runtime_plan.dtypes_for_link(link.name).activation_dtype
    shape = {
        "tokens": len(tokens),
        "hidden_size": planner.hidden_size,
        "elements": len(tokens) * planner.hidden_size,
    }
    dense_ids = {}
    for a, f in pairs:
        node_id = f"{prefix}:dense:r{f.domain_rank}"
        dense_ids[f.domain_rank] = node_id
        node = planner._runtime_compute_node(
            node_id=node_id,
            operation="dense_ffn",
            stage="dense_ffn",
            predecessors=(attention_ids[a.domain_rank],),
            binding=f,
            layer_type="dense",
            layer_id=layer_id,
            token_count=len(tokens),
            original_token_count=len(tokens),
            output_token_count=len(tokens),
            tensor_slices=replica_slices(planner, a, tokens, "activation", dtype),
            source_layout="replicated",
            target_layout="partial" if len(pairs) > 1 else "replicated",
        )
        nodes.append(replace(node, metadata={"colocated_dense_reuse": True}))
    f_bindings = tuple(f for _, f in pairs)
    collective = planner._append_runtime_collective(
        nodes,
        node_id=f"{prefix}:dense:collective",
        domain=f_domain,
        dp_rank=block.placement.f_dp_rank,
        bindings=f_bindings,
        producer_ids=dense_ids,
        after_stage="expert_compute",
        token_count=len(tokens),
        layer_type="dense",
        required=len(pairs) > 1,
        producer_layout="partial",
        required_target_layout="replicated",
    )
    for a, f in pairs:
        node = planner._runtime_compute_node(
            node_id=f"{prefix}:residual:r{a.domain_rank}",
            operation="residual",
            stage="residual",
            predecessors=(
                collective.node_id if collective else dense_ids[f.domain_rank],
            ),
            binding=a,
            layer_type="dense",
            layer_id=layer_id,
            token_count=len(tokens),
            original_token_count=len(tokens),
            output_token_count=len(tokens),
            tensor_slices=replica_slices(planner, f, tokens, "dense", dtype),
            source_layout="replicated",
            target_layout="replicated",
            shape=shape,
        )
        nodes.append(replace(node, metadata={"colocated_dense_reuse": True}))
    return AFExecutionDAG(tuple(nodes))
