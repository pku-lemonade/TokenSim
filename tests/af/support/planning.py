"""Af planning."""

from __future__ import annotations

from dataclasses import replace

from TokenSim.af.planning.batches import AFBlock, AFBlockPlacement
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment
from tests.af.support.hardware import af_test_hardware
from tests.af.support.parallel import _cluster


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
