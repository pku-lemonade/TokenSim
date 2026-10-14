"""Af runtime."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import simpy

from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.moe.config import MoEModelConfig
from tests.af.support.hardware import af_test_hardware
from tests.af.support.parallel import _cluster
from tests.af.support.providers import FixedComputeProvider, FixedTransferProvider


def _runtime_hardware(*, layers=2, capacity=0.1):
    """Catalog with ``capacity`` GiB per device and a ``layers``-layer TestModel."""
    return af_test_hardware(layers=layers, capacity_gib=capacity)


def _moe_config():
    return MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=1,
        num_moe_layers=1,
        moe_intermediate_size=32,
        num_shared_experts=1,
        intermediate_size=64,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=8,
    )


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
    moe = moe_config or _moe_config()
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
