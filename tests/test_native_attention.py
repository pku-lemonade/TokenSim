"""Regression tests for measured DSV4 CSA/HCA attention integration."""

from unittest import TestCase
from unittest.mock import patch

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.latency.attention import AttentionEstimate
from TokenSim.llm.llm_request import Request


class NativeAttentionTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = HardwareContext.load("data")
        cls.model = cls.context.model("DeepSeek-V4-Pro")

    def backend(self, sku="b200", *, tp=1, native=True, native_moe=True):
        device = self.context.device(sku)
        parallel = ParallelConfig(
            tensor_parallel_size=tp,
            data_parallel_size=8,
            enable_expert_parallel=True,
        )
        return OperatorTableLatencyBackend(
            device=device,
            model=self.model,
            parallel_config=parallel,
            rank_info=ParallelRankInfo.from_global_rank(0, parallel),
            package=self.context.operator_package(device.device_id, "vllm"),
            attention_backend="module" if native else "generic",
            moe_execution={
                "backend": "deep_gemm_amxf4_mega_moe" if native_moe else "generic"
            },
        )

    def test_layer_geometry_and_complete_module_scope(self):
        backend = self.backend()
        self.assertEqual(backend.native_attention.layer_counts["csa"], 30)
        self.assertEqual(backend.native_attention.layer_counts["hca"], 31)
        self.assertNotIn("swa", backend.native_attention.layer_counts)
        self.assertIn("projections", backend.describe()["native_cost_scope"])

    def test_context_and_generation_use_measured_module_rows(self):
        backend = self.backend()
        context = backend.native_attention.estimate_prefill(1, 1, 0)
        decode = backend.native_attention.estimate_decode(1, 1)
        self.assertGreater(context, 0)
        self.assertGreater(decode, 0)
        self.assertTrue(
            any("context:csa" in key for key in backend.native_attention.counts)
        )
        self.assertTrue(
            any("generation:hca" in key for key in backend.native_attention.counts)
        )

    def test_b300_lower_grid_is_explicitly_clamped(self):
        backend = self.backend("b300")
        backend.native_attention.estimate_prefill(1, 1, 0)
        backend.native_attention.estimate_decode(49, 1)
        self.assertTrue(
            any(":clamped" in key for key in backend.native_attention.counts)
        )

    def test_tp_local_heads_select_the_matching_table(self):
        backend = self.backend(tp=8, native_moe=False)
        value = backend.native_attention.estimate_decode(64, 8192)
        self.assertGreater(value, 0)
        self.assertEqual(
            backend.native_attention._query("hca", "generation", 64, 1, 8192).source_id,
            "aiconfigurator-borrowed:b200_sxm:dsv4_attention:vllm:0.24.0",
        )

    def test_ragged_decode_does_not_repeat_projection_launches(self):
        backend = self.backend()
        requests = []
        for identifier, context in enumerate((8192, 65536)):
            request = Request(identifier, context, 20, 256)
            request.generation_idx = 1
            request.num_computed_tokens = context
            request.scheduled_tokens = 1
            request.scheduled_query_tokens = 1
            requests.append(request)

        def measured_module(_kind, _phase, batch, _query, _context):
            # Common launch/weight work + linear query work. Changing KV
            # lengths does not cause another projection launch in serving.
            return AttentionEstimate(100.0 + 2.0 * batch, "exact", "fixture")

        with patch.object(backend.native_attention, "_query", side_effect=measured_module), patch.object(
            backend, "_compose", side_effect=lambda **step: step["attention_us"]
        ):
            latency_us = backend.estimate_step_latency(requests)
        self.assertEqual(latency_us, 61 * (100.0 + 2.0 * 2))
        self.assertAlmostEqual(backend.native_attention.seconds, latency_us * 1e-6)
        self.assertEqual(backend.native_attention.decode_batch_stats["batches"], 1)

    def test_target_logits_cover_verify_rows_and_partial_prefills(self):
        backend = self.backend(native=False, native_moe=False)
        requests = []
        for i in range(2):
            req = Request(i, 1024, 20, 256)
            req.scheduled_tokens = req.scheduled_query_tokens = 512
            requests.append(req)
        decode = Request(2, 1024, 2, 256)
        decode.generation_idx = 1
        decode.num_computed_tokens = 1024
        decode.scheduled_tokens = 1  # final output token
        decode.scheduled_query_tokens = 4
        requests.append(decode)
        composed = []
        with patch.object(backend, "_attention_prefill", return_value=0), patch.object(
            backend, "_attention_decode", return_value=0
        ), patch.object(backend, "_compose", side_effect=lambda **args: composed.append(args) or 0):
            backend.estimate_step_latency(requests)
        self.assertEqual(composed[0]["requests"], 6)  # two bonus + four verify logits
        self.assertEqual(composed[0]["tokens"], 1028)
        self.assertEqual(backend.target_logits_counts["decode_rows"], 4)
        self.assertEqual(backend.target_logits_counts["prefill_rows"], 2)
        self.assertEqual(backend.target_logits_counts["rows"], 6)
        backend.reset_profile_stats()
        self.assertEqual(backend.target_logits_counts, {})

    def test_homogeneous_batch_preserves_measured_lookup(self):
        for sku in ("b200", "b300"):
            estimator = self.backend(sku).native_attention
            expected = estimator.estimate_decode(64, 8192)
            estimator.reset_stats()
            actual = estimator.estimate_decode_batch({(8192, 1): 64})
            self.assertEqual(actual, expected)
            self.assertAlmostEqual(estimator.seconds, actual * 1e-6)
            self.assertEqual(estimator.decode_batch_stats["query_rows"], 64)

    def test_mtp_reuses_indexer_kv_without_removing_query_work(self):
        estimator = self.backend().native_attention
        independent = estimator.estimate_decode(64, 131072)
        estimator.reset_stats()
        verify = estimator.estimate_decode_batch({(131072, 4): 16})
        self.assertLess(verify, independent)
        stats = estimator.mtp_kv_stats
        self.assertAlmostEqual(stats["source_geometry_indexer_bytes_after"] * 2,
                               stats["source_geometry_indexer_bytes_before"])
        self.assertAlmostEqual(stats["estimated_seconds_saved"], (independent - verify) * 1e-6)
        self.assertEqual(estimator.decode_batch_stats["requests"], 16)
        self.assertEqual(estimator.decode_batch_stats["query_rows"], 64)
        self.assertEqual(estimator.decode_batch_stats["indexer_query_tiles"], 32)
        estimator.reset_stats()
        self.assertEqual(estimator.mtp_kv_stats, {})

    def test_mixed_verify_widths_share_only_request_local_indexer_reads(self):
        estimator = self.backend().native_attention
        estimator.estimate_decode_batch({(131072, 1): 16, (131072, 4): 12})
        stats = estimator.mtp_kv_stats
        self.assertAlmostEqual(stats["source_geometry_indexer_bytes_after"] /
                               stats["source_geometry_indexer_bytes_before"], 40 / 64)
        self.assertEqual(estimator.decode_batch_stats["requests"], 28)

    def test_three_query_tail_counts_two_indexer_tiles(self):
        estimator = self.backend().native_attention
        estimator.estimate_decode_batch({(131072, 3): 16})
        self.assertEqual(estimator.decode_batch_stats["indexer_query_tiles"], 32)
        stats = estimator.mtp_kv_stats
        self.assertAlmostEqual(stats["source_geometry_indexer_bytes_after"] /
                               stats["source_geometry_indexer_bytes_before"], 2 / 3)

    def test_b300_mtp_correction_preserves_the_source_grid_clamp(self):
        estimator = self.backend("b300").native_attention
        shape = estimator._query("csa", "generation", 4, 1, 8192)
        self.assertEqual(shape.source_batch_size, 64)
        independent = estimator.estimate_decode(4, 8192)
        estimator.reset_stats()
        value = estimator.estimate_decode_batch({(8192, 4): 1})
        self.assertGreater(value, 0)
        self.assertLess(value, independent)
        self.assertTrue(any("clamped" in k or "extrapolated" in k for k in estimator.counts))

    def test_ragged_estimate_stays_within_uniform_full_batch_costs(self):
        estimator = self.backend().native_attention
        uniform = [estimator.estimate_decode(64, context) for context in (8192, 65536)]
        estimator.reset_stats()
        actual = estimator.estimate_decode_batch({(8192, 1): 32, (65536, 1): 32})
        self.assertGreaterEqual(actual, min(uniform))
        self.assertLessEqual(actual, max(uniform))
        self.assertAlmostEqual(estimator.seconds, actual * 1e-6)
        estimator.reset_stats()
        self.assertEqual(estimator.decode_batch_stats, {})
        self.assertEqual(estimator.seconds, 0.0)

    def test_long_context_uses_supported_slices_without_changing_source_rows(self):
        for sku in ("b200", "b300"):
            estimator = self.backend(sku).native_attention
            original = estimator.backend.package.tables["dsv4_attention"]
            value = estimator.estimate_decode_batch({(737408, 4): 17})
            self.assertGreater(value, 0)
            self.assertGreater(estimator.decode_batch_stats["ragged_grid_lookup_queries"], 0)
            self.assertIs(estimator.backend.package.tables["dsv4_attention"], original)
            self.assertEqual(estimator.lookup.policy.max_extrapolation_ratio, 64.0)

    def test_ragged_grid_keeps_extrapolation_limit(self):
        estimator = self.backend().native_attention
        with self.assertRaises(ConfigurationError):
            estimator.estimate_decode(64, 65 * 65536)

    def test_native_attention_does_not_add_projection_gemms(self):
        backend = self.backend(native_moe=False)
        # The complete module table owns all four projections.  The backend
        # still charges the outer norm/residual operators and MoE/router work.
        backend.native_attention.estimate_decode(1, 8192)
        self.assertEqual(backend._compose(
            tokens=1,
            requests=1,
            attention_us=backend.native_attention.estimate_decode(1, 8192),
            requests_list=[],
        ) > 0, True)
        self.assertGreater(backend.stats.table_match_counts["dsv4_attention"]["exact"], 0)

    def test_native_mode_requires_the_attention_table(self):
        device = self.context.device("b200")
        package = OperatorDataPackage.from_rows(
            self.context.operator_package(device.device_id, "vllm").meta,
            {"gemm": list(self.context.operator_package(device.device_id, "vllm").tables["gemm"])},
        )
        with self.assertRaises(ConfigurationError):
            OperatorTableLatencyBackend(
                device=device,
                model=self.model,
                parallel_config=ParallelConfig(),
                package=package,
                attention_backend="module",
            )

    def test_native_mhc_uses_pre_and_post_without_site_multiplier(self):
        backend = self.backend(native=False, native_moe=False)
        backend = OperatorTableLatencyBackend(
            device=backend.device,
            model=self.model,
            parallel_config=backend.parallel_config,
            rank_info=backend.rank_info,
            package=backend.package,
            mhc_backend="module",
        )
        value = backend.native_mhc.estimate(64)
        self.assertGreater(value, 0)
        self.assertEqual(len(backend.native_mhc.counts), 2)
        self.assertEqual(backend.native_mhc.describe()["hc_mult"], 4)

    def test_native_mhc_owns_residual_and_retains_unfused_norm_work(self):
        template = self.backend(native=False, native_moe=False)
        backend = OperatorTableLatencyBackend(
            device=template.device, model=self.model, parallel_config=template.parallel_config,
            rank_info=template.rank_info, package=template.package, mhc_backend="module",
        )
        operations = []
        with patch.object(backend, "elementwise", side_effect=lambda name, *args: operations.append(name) or 1), patch.object(
            backend, "gemm", return_value=0
        ), patch.object(backend, "_moe_layers", return_value=0):
            backend._compose(tokens=64, requests=16, attention_us=0, requests_list=[])
        self.assertNotIn("residual_add", operations)
        self.assertEqual(operations.count("rmsnorm"), 3)  # attention, FFN, target head
        self.assertGreater(backend.native_mhc.seconds, 0)

    def test_native_mhc_uses_sequence_parallel_rows_before_attention_gather(self):
        template = self.backend(native=False, native_moe=False)
        parallel = ParallelConfig(tensor_parallel_size=4, data_parallel_size=2,
                                  enable_expert_parallel=True)
        backend = OperatorTableLatencyBackend(
            device=template.device, model=self.model, parallel_config=parallel,
            package=template.package, mhc_backend="module",
        )
        calls = []
        with patch.object(backend.native_mhc, "estimate", return_value=1) as mhc, patch.object(
            backend, "elementwise", side_effect=lambda name, rows, hidden: calls.append((name, rows)) or 1
        ), patch.object(backend, "gemm", return_value=0), patch.object(backend, "_moe_layers", return_value=0):
            backend._compose(tokens=66, requests=8, attention_us=0, requests_list=[])
        mhc.assert_called_once_with(17)
        self.assertEqual([rows for name, rows in calls if name == "rmsnorm"], [17, 17, 66])


if __name__ == "__main__":
    import unittest

    unittest.main()
