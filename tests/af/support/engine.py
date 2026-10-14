"""Af engine."""

from __future__ import annotations

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request
from TokenSim.workload.source import LLMSource
from tests.af.support.config import _cluster_dict
from tests.af.support.hardware import af_test_hardware
from tests.af.support.providers import FixedComputeProvider


class _CountingRequest(Request):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class _DTypeCheckingComputeProvider(FixedComputeProvider):
    def __init__(self):
        super().__init__(0.0002, 0.0003)
        self.validated_contracts = []

    def validate_effective_dtypes(self, model_id, effective_dtypes):
        self.validated_contracts.append((model_id, effective_dtypes))


def _typed_cluster(*, phases=("decode",), trace_enabled=True):
    values = _cluster_dict(phases=phases, trace_enabled=trace_enabled)
    values["af"].update(
        {
            "weight_dtype": "fp8",
            "activation_dtype": "bf16",
            "kv_cache_dtype": "fp32",
        }
    )
    values["af"]["links"][0].update(
        {
            "a_to_f_dtype": "fp8",
            "f_to_a_dtype": "fp16",
        }
    )
    return ClusterConfig(**values)


def _disabled_cluster():
    return ClusterConfig(
        num_workers=1,
        networks={"local": "profile"},
        worker_groups=[
            {
                "name": "full-pool",
                "capabilities": ["prefill", "decode_full"],
                "hardware": "a-device",
                "num_workers": 1,
                "network": "local",
            }
        ],
    )


def _build_engine(cluster, compute=None, transfer=None):
    env = simpy.Environment()
    hardware = af_test_hardware(layers=1, capacity_gib=0.1)
    engine = LLMEngine(
        env=env,
        kv_transfer_config=KVTransferConfig.default(),
        model_spec=hardware.model("TestModel"),
        cluster_config=cluster,
        hardware=hardware,
        af_compute_provider=compute,
        af_transfer_provider=transfer,
        config=SimulationConfig(
            block_size=4,
            batching="paged-attn",
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_num_seqs=8,
        ),
    )
    return env, engine, hardware


def _run_request(env, engine, *, request_id=1, decode_len=3):
    request = _CountingRequest(
        id=request_id,
        prefill_len=4,
        decode_len=decode_len,
        block_size=4,
    )
    env.process(
        LLMSource(
            env=env,
            engine=engine,
            requests=[request],
            qps=1,
            distribution="burst",
        )
    )

    def stop_when_done():
        while env.now < 1.0:
            if request.is_done:
                engine.send_task(engine, Task.STOP)
                yield env.timeout(1e-9)
                return
            yield env.timeout(1e-6)
        raise AssertionError("engine request did not finish")

    monitor = env.process(stop_when_done())
    env.run(until=monitor)
    return request
