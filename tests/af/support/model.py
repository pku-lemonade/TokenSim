"""Small model, query, runtime-plan and ownership fixtures."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import simpy

from TokenSim.af.planning.batching import AFBlock, AFBlockPlacement
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.af.providers.queries import AFComputeQuery, AFTransferQuery
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_hardware,
)


HIDDEN = 16


def _moe(*, top_k=2, shared=0):
    return MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=top_k,
        num_shared_experts=shared,
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


def _block(histogram=((0, 10), (2, 2))):
    return AFBlock(
        iteration_id=3,
        block_id=4,
        phase="decode",
        request_ids=(1,),
        request_count=1,
        token_count=12,
        context_lengths=(32,),
        source_a_worker_id=2,
        link_name="af-link",
        placement=AFBlockPlacement(
            a_domain="a-domain",
            a_instance="a1",
            a_dp_rank=1,
            f_domain="f-domain",
            f_instance="f0",
            f_dp_rank=0,
        ),
        expert_histogram=histogram,
    )


def _formal_runtime_plan(cluster, moe_config):
    hardware = af_test_hardware(layers=2, capacity_gib=0.1)
    model = replace(hardware.model("TestModel"), moe=moe_config)
    return compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        hardware=hardware,
        model=model,
        block_size=4,
        dtype="fp16",
    )


def _small_formal_moe_dag(cluster=None):
    cluster = cluster or _cluster(a_tp=1, a_dp=1, f_tp=2, f_dp=1, f_ep=2)
    moe_config = MoEModelConfig(
        is_moe_model=True,
        num_experts=2,
        num_experts_per_tok=1,
        num_moe_layers=1,
        moe_intermediate_size=32,
        num_shared_experts=1,
        intermediate_size=64,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=8,
    )
    runtime_plan = _formal_runtime_plan(cluster, moe_config)
    planner = ParallelExecutionPlanner(cluster, runtime_plan=runtime_plan)
    placement = planner.placement(
        "decode",
        "a-pool:0",
        cluster.execution_domains["f-domain"].logical_instances[0],
    )
    assignment = RoutingAssignment(
        iteration_id=6,
        phase="decode",
        decode_step=1,
        layer_id=0,
        request_id=1,
        token_id=8,
        top_k_index=0,
        expert_id=0,
        routing_weight=1.0,
        a_dp_rank=0,
        a_tp_rank=0,
    )
    snapshot = LayerRoutingSnapshot(
        iteration_id=6,
        phase="decode",
        decode_step=1,
        layer_id=0,
        top_k=1,
        original_tokens=((1, 8),),
        assignments=(assignment,),
    )
    block = AFBlock(
        iteration_id=6,
        block_id=0,
        phase="decode",
        request_ids=(1,),
        request_count=1,
        token_count=1,
        context_lengths=(9,),
        source_a_worker_id=0,
        link_name="af-link",
        placement=placement,
        routing_snapshots=(snapshot,),
    )
    return planner.build_layer(block, 0, cluster.af.links[0])


def _runtime_hardware(*, layers=2, capacity=0.1):
    """Catalog with ``capacity`` GiB per device and a ``layers``-layer TestModel."""
    return af_test_hardware(layers=layers, capacity_gib=capacity)


def _runtime_plan(
    cluster=None,
    *,
    hardware=None,
    moe_config=None,
    dtype="fp16",
):
    cluster = cluster or _cluster()
    hardware = hardware or _runtime_hardware()
    base = hardware.model("TestModel")
    moe = moe_config or _moe(top_k=1, shared=1)
    hidden = moe.hidden_size or base.hidden_size
    heads = moe.num_attention_heads or base.num_attention_heads
    model = replace(
        base,
        moe=moe,
        hidden_size=hidden,
        intermediate_size=moe.intermediate_size or base.intermediate_size,
        num_attention_heads=heads,
        num_key_value_heads=moe.num_key_value_heads or base.num_key_value_heads,
        head_dim=hidden // heads,
    )
    return compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        model=model,
        hardware=hardware,
        block_size=4,
        dtype=dtype,
    )


def _request(request_id, prefill_len=8, decode_len=8):
    request = Request(request_id, prefill_len, decode_len, block_size=4)
    request.arrive(SimpleNamespace(now=float(request_id)))
    return request


def _engine(cluster, *, moe_config=None):
    env = simpy.Environment()
    hardware = _runtime_hardware()
    engine = LLMEngine(
        env=env,
        kv_transfer_config=KVTransferConfig.default(),
        model_spec=(
            replace(hardware.model("TestModel"), moe=moe_config)
            if moe_config
            else hardware.model("TestModel")
        ),
        cluster_config=cluster,
        hardware=hardware,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        af_compute_provider=FixedComputeProvider(0.001, 0.001),
        af_transfer_provider=FixedTransferProvider(
            0.001, 0.001, dispatch=0.001, combine=0.001, collective=0.001
        ),
        config=SimulationConfig(
            block_size=4,
            batching="paged-attn",
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_num_seqs=8,
        ),
    )
    return env, engine


def _split_phase_alias_cluster():
    base = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
    attention = base.execution_domains["a-domain"]
    ffn = base.execution_domains["f-domain"]
    prefill_attention = replace(
        attention,
        name="prefill-a",
        phases=("prefill",),
    )
    decode_attention = replace(
        attention,
        name="decode-a",
        phases=("decode",),
        logical_instances=[
            replace(instance, name=f"{instance.name}-decode")
            for instance in attention.logical_instances
        ],
    )
    link = base.af.links[0]
    af = replace(
        base.af,
        links=[
            replace(
                link,
                name="prefill-af",
                phases=("prefill",),
                attention_domain="prefill-a",
            ),
            replace(
                link,
                name="decode-af",
                phases=("decode",),
                attention_domain="decode-a",
            ),
        ],
        execution_domains=[prefill_attention, decode_attention, ffn],
    )
    return base.__class__(
        num_workers=base.num_workers,
        networks=base.networks,
        worker_groups=base.worker_groups,
        kv_transfer=base.kv_transfer,
        parallel_config=base.parallel_config,
        af=af,
    )
