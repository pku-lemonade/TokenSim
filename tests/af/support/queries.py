"""Af queries."""

from __future__ import annotations

from TokenSim.af.providers.queries import AFComputeQuery, AFTransferQuery
from TokenSim.moe.config import MoEModelConfig


HIDDEN = 16


def _moe():
    return MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=2,
        num_moe_layers=1,
        moe_intermediate_size=32,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
    )


def _compute_query(
    stage="a",
    *,
    phase="decode",
    tokens=2,
    context=(2,),
    tp_size=1,
    ep_size=1,
    expert_workload=None,
    dtype="fp16",
    local_shape=None,
):
    if expert_workload is None and stage == "expert":
        expert_workload = ((0, tokens),)
    if local_shape is None:
        local_shape = {
            "tokens": tokens,
            "hidden_size": HIDDEN if stage == "router" else -(-HIDDEN // tp_size),
            "global_hidden_size": HIDDEN,
            "tp_size": tp_size,
        }
    return AFComputeQuery(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase=phase,
        stage=stage,
        token_count=tokens,
        context_lengths=context,
        model_id="TestModel",
        dtype=dtype,
        device_type="a-device" if stage in {"a", "router"} else "f-device",
        worker_id=0 if stage in {"a", "router"} else 1,
        operation={
            "a": "a_rank_compute",
            "router": "router_topk",
            "expert": "expert_rank_compute",
        }[stage],
        source_domain="a-domain" if stage in {"a", "router"} else "f-domain",
        logical_participant="a0" if stage in {"a", "router"} else "f0",
        physical_worker="a-pool:0" if stage in {"a", "router"} else "f-pool:0",
        local_shape=local_shape,
        local_token_count=tokens,
        expert_ids=tuple(expert_id for expert_id, _ in (expert_workload or ())),
        expert_workload=tuple(expert_workload or ()),
        tp_group=tuple(range(tp_size)),
        ep_group=tuple(range(ep_size)) if stage == "expert" else (),
    )


def _transfer_query(
    operation="af_a_to_f",
    *,
    payload_bytes=4096,
    endpoints=("a-pool:0", "f-pool:0"),
    network="af-net",
    group_size=None,
):
    direction = {
        "af_a_to_f": "a_to_f",
        "af_f_to_a": "f_to_a",
        "tp_collective": "tp_collective",
        "ep_dispatch": "a_to_f",
        "ep_combine": "f_to_a",
    }.get(operation, "a_to_f")
    return AFTransferQuery(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase="decode",
        direction=direction,
        token_count=2,
        hidden_size=HIDDEN,
        dtype="fp16",
        source_device="a-device",
        target_device="f-device",
        network=network,
        operation=operation,
        group_size=group_size,
        payload_bytes=payload_bytes,
        physical_endpoints=endpoints,
        participant_ranks=tuple(range(len(endpoints))),
    )


def _component_names(estimate):
    return [item["name"] for item in estimate.details["components"]]
