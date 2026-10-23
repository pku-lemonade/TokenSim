"""Combine KV-parallel attention partials before the output projection.

Each KVP group of consecutive TP ranks exchanges partial outputs and
log-sum-exp values with one all-to-all, after which every rank merges the
fragments of its ``heads / tp`` retained query heads and projects them.  The
existing attention all-reduce then consumes the projected partial sums.
"""

from __future__ import annotations

from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.af.providers.context_parallel import combine_payload_bytes
from TokenSim.errors import ConfigurationError


def append_kv_parallel_combine(
    planner, nodes, *, block, layer_id, layer_type, prefix, domain, bindings, producers
):
    """Return per-rank output producers, adding combine nodes on KVP layers."""
    partitions = {
        binding.domain_rank: binding.kv_partition.for_layer(layer_id)
        for binding in bindings
    }
    kvp = {partition.kv_parallel_size for partition in partitions.values()}
    if kvp == {1}:
        return producers
    if len(kvp) != 1 or len(bindings) % next(iter(kvp)):
        raise ConfigurationError("KV-parallel ranks must form complete groups")
    kvp = kvp.pop()
    plan = planner.runtime_plan.collective_plan(
        domain.name, block.placement.a_dp_rank, "a_compute", bindings[0].pp_rank
    )
    if plan is None:
        raise ConfigurationError(
            "KV-parallel combine reuses the attention collective's network"
        )
    layer = planner.runtime_plan.layer_plan(layer_id)
    tokens = block.token_count
    heads = layer.num_attention_heads // len(bindings)
    payload = combine_payload_bytes(tokens, heads, layer.head_dim, kvp)
    ordered = sorted(bindings, key=lambda binding: binding.tp_rank)
    outputs = {}
    for group in range(len(ordered) // kvp):
        members = ordered[group * kvp : (group + 1) * kvp]
        if [partitions[m.domain_rank].kv_parallel_rank for m in members] != list(
            range(kvp)
        ):
            raise ConfigurationError("KV-parallel group ranks must be consecutive")
        node_id = f"{prefix}:attention_combine:g{group}"
        ranks = tuple(member.domain_rank for member in members)
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                operation="all_to_all",
                kind="communication",
                predecessor_ids=tuple(producers[rank] for rank in ranks),
                input_producer_ids=tuple(
                    (member.physical_identity, producers[member.domain_rank])
                    for member in members
                ),
                stage=None,
                direction="all_to_all",
                source_domain=domain.name,
                target_domain=domain.name,
                logical_participants=tuple(
                    dict.fromkeys(member.instance for member in members)
                ),
                domain_ranks=ranks,
                physical_workers=tuple(member.physical_identity for member in members),
                dp_rank=block.placement.a_dp_rank,
                tp_group=ranks,
                network=plan.network,
                resource_group=plan.resource_group,
                resource_lane=plan.resource_lane,
                payload_bytes=payload,
                logical_input_bytes_per_rank=tuple((rank, payload) for rank in ranks),
                logical_output_bytes_per_rank=tuple((rank, payload) for rank in ranks),
                layer_type=layer_type,
                logical_stage_id=node_id,
                participant_endpoints=tuple(
                    planner._endpoint(member) for member in members
                ),
                source_layout="partial",
                target_layout="sharded",
                metadata={
                    "after_stage": "attention_core",
                    "collective_input_binding_required": True,
                    "original_token_count": tokens,
                    "kv_parallel_size": kvp,
                },
            )
        )
        for member in members:
            output_id = f"{prefix}:attention_output:r{member.domain_rank}"
            outputs[member.domain_rank] = output_id
            nodes.append(
                planner._runtime_compute_node(
                    node_id=output_id,
                    operation="attention_output",
                    stage="attention",
                    predecessors=(node_id,),
                    binding=member,
                    layer_type=layer_type,
                    layer_id=layer_id,
                    token_count=tokens,
                    original_token_count=tokens,
                    output_token_count=tokens,
                    shape={
                        "tokens": tokens,
                        "hidden_size": planner.hidden_size,
                        "output_heads": heads,
                        "head_dim": layer.head_dim,
                        "kv_parallel_size": kvp,
                    },
                )
            )
    return outputs
