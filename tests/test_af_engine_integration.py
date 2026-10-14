"""Af engine integration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from TokenSim.reporting import get_af_stats
from tests.support.af_engine import (
    _DTypeCheckingComputeProvider,
    _build_engine,
    _disabled_cluster,
    _run_request,
    _typed_cluster,
)
from tests.support.af_providers import FixedTransferProvider


class AFEngineIntegrationTest(unittest.TestCase):
    def test_decode_only_connects_runtime_plan_dtypes_and_providers(self):
        compute = _DTypeCheckingComputeProvider()
        transfer = FixedTransferProvider(0.0001, 0.0001)
        env, engine, roofline = _build_engine(_typed_cluster(), compute, transfer)

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
                "prefill": "prefill_af_rank_local",
                "decode": "decode_aggregate",
            },
        )

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
