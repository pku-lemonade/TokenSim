from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import simpy

try:
    import TransformerRoofline  # noqa: F401
except (ImportError, OSError):
    fake_roofline = types.ModuleType("TransformerRoofline")
    fake_roofline.TransformerRoofline = object
    sys.modules["TransformerRoofline"] = fake_roofline

from tests.test_af_executor import (
    FixedComputeProvider,
    FixedTransferProvider,
    _cluster_dict,
    _psla_config,
    af_test_hardware,
)
from TokenSim.config.config import ClusterConfig, KVTransferConfig
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, reset_g_time
from util.request import LLMSource
from util.results import get_af_stats


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
        block_size=4,
        batching="paged-attn",
        kv_transfer_config=KVTransferConfig.default(),
        psla_config=_psla_config(),
        cluster_config=cluster,
        hardware=hardware,
        prefill_worker_pool_type="round_robin",
        decode_worker_pool_type="round_robin",
        max_parallem_sum=8,
        af_compute_provider=compute,
        af_transfer_provider=transfer,
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


class AFEngineIntegrationTest(unittest.TestCase):
    def setUp(self):
        reset_g_time()

    def test_decode_only_connects_runtime_plan_dtypes_and_providers(self):
        compute = _DTypeCheckingComputeProvider()
        transfer = FixedTransferProvider(0.0001, 0.0001)
        env, engine, roofline = _build_engine(
            _typed_cluster(), compute, transfer
        )

        plan = engine.af_runtime_plan
        self.assertIsNotNone(plan)
        self.assertIs(engine.af_executor.runtime_plan, plan)
        self.assertEqual(
            {
                "weight_dtype": plan.effective_dtypes.weight_dtype,
                "activation_dtype": plan.effective_dtypes.activation_dtype,
                "kv_cache_dtype": plan.effective_dtypes.kv_cache_dtype,
                "a_to_f_dtype": plan.effective_dtypes.a_to_f_dtype,
                "f_to_a_dtype": plan.effective_dtypes.f_to_a_dtype,
            },
            {
                "weight_dtype": "fp8",
                "activation_dtype": "bf16",
                "kv_cache_dtype": "fp32",
                "a_to_f_dtype": "fp8",
                "f_to_a_dtype": "fp16",
            },
        )
        self.assertEqual(
            [(layer.layer_id, layer.layer_type) for layer in plan.layers],
            [(0, "dense")],
        )
        self.assertEqual(len(compute.validated_contracts), 1)
        self.assertEqual(compute.validated_contracts[0][0], "TestModel")

        decode_manager = engine.decode_workers.workers[0].scheduler.block_manager
        with patch.object(
            decode_manager,
            "commit_input_cache",
            wraps=decode_manager.commit_input_cache,
        ) as commit_input_cache:
            request = _run_request(env, engine)

        self.assertTrue(request.is_done)
        self.assertEqual(request.generation_idx, 3)
        self.assertEqual(request.step_calls, 3)
        self.assertEqual(commit_input_cache.call_count, 2)
        self.assertEqual(
            [item["phase"] for item in engine.af_iteration_summaries],
            ["decode", "decode"],
        )
        self.assertEqual({query.phase for query in compute.queries}, {"decode"})
        self.assertEqual({query.phase for query in transfer.queries}, {"decode"})
        self.assertTrue({query.stage for query in compute.queries})
        for query in compute.queries:
            self.assertEqual(query.weight_dtype, "fp8")
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.kv_cache_dtype, "fp32")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp16")
        for query in transfer.queries:
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp16")
            if query.direction == "a_to_f":
                self.assertEqual(query.dtype, "fp8")
            elif query.direction == "f_to_a":
                self.assertEqual(query.dtype, "fp16")
        self.assertEqual(
            get_af_stats(engine)["af_phase_fidelity"],
            {
                "prefill": "prefill_aggregate",
                "decode": "decode_af_rank_local",
            },
        )

    def test_af_prefill_uses_compatibility_path_and_full_decode(self):
        compute = _DTypeCheckingComputeProvider()
        transfer = FixedTransferProvider(0.0001, 0.0001)
        env, engine, roofline = _build_engine(
            _typed_cluster(phases=("prefill",)), compute, transfer
        )

        request = _run_request(env, engine, decode_len=2)

        self.assertTrue(request.is_done)
        self.assertEqual(request.step_calls, 2)
        self.assertEqual(
            [item["phase"] for item in engine.af_iteration_summaries],
            ["prefill"],
        )
        self.assertEqual({query.phase for query in compute.queries}, {"prefill"})
        self.assertEqual({query.phase for query in transfer.queries}, {"prefill"})
        self.assertEqual(
            get_af_stats(engine)["af_phase_fidelity"],
            {
                "prefill": "af_prefill_compatibility",
                "decode": "decode_aggregate",
            },
        )

    def test_trace_toggle_does_not_change_engine_timing(self):
        results = {}
        for trace_enabled in (True, False):
            reset_g_time()
            compute = _DTypeCheckingComputeProvider()
            transfer = FixedTransferProvider(0.0001, 0.0001)
            env, engine, roofline = _build_engine(
                _typed_cluster(trace_enabled=trace_enabled), compute, transfer
            )
            request = _run_request(env, engine)
            results[trace_enabled] = {
                "completion": env.now,
                "prefill_latency": request.prefill_latency,
                "decode_time_sum": request.decode_time_sum,
                "iteration_makespans": tuple(
                    item["makespan"] for item in engine.af_iteration_summaries
                ),
                "block_completion": tuple(
                    tuple(sorted(item["block_completed_at"].items()))
                    for item in engine.af_iteration_summaries
                ),
                "query_stages": tuple(query.stage for query in compute.queries),
                "events": tuple(
                    len(engine.af_executor.trace.events_for(item["iteration_id"]))
                    for item in engine.af_iteration_summaries
                ),
            }

        comparable_fields = (
            "completion",
            "prefill_latency",
            "decode_time_sum",
            "iteration_makespans",
            "block_completion",
            "query_stages",
        )
        for field_name in comparable_fields:
            self.assertEqual(results[True][field_name], results[False][field_name])
        self.assertTrue(all(count > 0 for count in results[True]["events"]))
        self.assertTrue(all(count == 0 for count in results[False]["events"]))

    def test_af_disabled_keeps_complete_backend_and_creates_no_af_state(self):
        env, engine, roofline = _build_engine(_disabled_cluster())

        request = _run_request(env, engine, decode_len=2)

        self.assertTrue(request.is_done)
        self.assertEqual(request.step_calls, 2)
        self.assertIsNone(engine.af_runtime_plan)
        self.assertIsNone(engine.af_executor)
        self.assertEqual(
            get_af_stats(engine),
            {
                "af_enabled": False,
                "af_iteration_count": 0,
                "af_trace": None,
                "af_runtime_state": None,
                "af_phase_fidelity": None,
                "af_latency_providers": None,
                "af_operator_stats": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
