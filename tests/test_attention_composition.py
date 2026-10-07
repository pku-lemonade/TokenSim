"""Mixed attention ownership, measured anchors and query-shape regressions."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from TokenSim.config.config import ClusterConfig
from TokenSim.config.psla_config import LLMResult, MetricData
from TokenSim.latency.deepseek_v4_attention import AttentionEstimate
from TokenSim.llm.llm_request import Request
from util.results import get_latency_stats
import tests.test_native_attention as native_attention


class AttentionCompositionTest(TestCase):
    @classmethod
    def setUpClass(cls):
        native_attention.NativeAttentionTest.setUpClass()

    def estimator(self, sku="b200", tp=1):
        return native_attention.NativeAttentionTest().backend(sku, tp=tp, native_moe=False).native_attention

    @staticmethod
    def components(key, **_kwargs):
        rows = key["batch_size"] * (key["input_seq_len"] if key["phase"] == "context" else 1)
        # A common intercept and a separate 10us sparse-kernel component.
        return 100.0 + 2.0 * rows, 10.0

    @staticmethod
    def module(_kind, phase, batch, queries, _context):
        rows = batch * (queries if phase == "context" else 1)
        return AttentionEstimate(110.0 + 2.0 * rows, "exact", "ownership-fixture")

    def test_mixed_batch_owns_projections_once_and_keeps_three_phase_components(self):
        estimator = self.estimator()
        with patch.object(estimator, "_query", side_effect=self.module), patch.object(
            estimator.sol, "shared_phase_us", side_effect=self.components
        ):
            value = estimator.estimate_batch({(512, 65536): 1, (256, 131072): 2}, {(8320, 1): 16})
        self.assertAlmostEqual(value, 61 * (100 + 2 * 1040 + 3 * 10))
        stats = estimator.composition_stats
        self.assertAlmostEqual(stats["avoided_shared_seconds"], 61 * 200 * 1e-6)
        self.assertAlmostEqual(stats["phase_seconds"], 61 * 30 * 1e-6)
        self.assertEqual(stats["shared_layer_invocations"], 61)
        self.assertEqual(stats["query_rows"], 1040)
        self.assertEqual(estimator.decode_batch_stats["query_rows"], 16)
        self.assertAlmostEqual(estimator.seconds, stats["shared_projection_seconds"] + stats["phase_seconds"])
        self.assertAlmostEqual(stats["standalone_module_seconds"] - estimator.seconds, stats["avoided_shared_seconds"])

    def test_context_groups_also_share_projections_without_decode(self):
        estimator = self.estimator()
        with patch.object(estimator, "_query", side_effect=self.module), patch.object(
            estimator.sol, "shared_phase_us", side_effect=self.components
        ):
            value = estimator.estimate_batch({(512, 65536): 1, (512, 131072): 1}, {})
        self.assertAlmostEqual(value, 61 * (100 + 2 * 1024 + 2 * 10))
        self.assertEqual(estimator.composition_stats["mixed_phase_batches"], 0)
        self.assertEqual(estimator.decode_batch_stats, {})

    def test_single_phase_homogeneous_measurements_are_unchanged(self):
        for sku, tp in (("b200", 1), ("b300", 1), ("b200", 8), ("b300", 4)):
            with self.subTest(sku=sku, tp=tp):
                estimator = self.estimator(sku, tp)
                expected = estimator.estimate_prefill(1, 512, 65536)
                estimator.reset_stats()
                self.assertEqual(estimator.estimate_batch({(512, 65536): 1}, {}), expected)
                estimator.reset_stats()
                expected = estimator.estimate_decode_batch({(8192, 4): 16})
                estimator.reset_stats()
                self.assertEqual(estimator.estimate_batch({}, {(8192, 4): 16}), expected)
                self.assertEqual(estimator.composition_stats, {})

    def test_measured_efficiencies_are_row_weighted_without_fitted_scale(self):
        estimator = self.estimator()

        def module(kind, phase, batch, queries, context):
            original = self.module(kind, phase, batch, queries, context)
            return AttentionEstimate(original.latency_us * (2 if phase == "context" else 3), "exact", "fixture")

        with patch.object(estimator, "_query", side_effect=module), patch.object(
            estimator.sol, "shared_phase_us", side_effect=self.components
        ):
            value = estimator.estimate_batch({(512, 65536): 1}, {(8192, 1): 16})
        self.assertAlmostEqual(value, 61 * ((100 + 2 * 528) * (512 * 2 + 16 * 3) / 528 + 50))

    def test_source_query_batch_floors_are_retained_in_the_component_allocation(self):
        estimator = self.estimator()

        def module(_kind, phase, _batch, _queries, context):
            queries = 8 if phase == "context" else 1
            return AttentionEstimate(110 + 2 * 64 * queries, "clamped", "fixture", False, 64, context, queries)

        with patch.object(estimator, "_query", side_effect=module), patch.object(
            estimator.sol, "shared_phase_us", side_effect=self.components
        ):
            value = estimator.estimate_batch({(4, 65536): 1}, {(8192, 1): 1})
        self.assertAlmostEqual(value, 61 * (100 + 2 * 512 + 20))
        self.assertEqual(estimator.composition_stats["query_rows"], 5)
        self.assertEqual(estimator.composition_stats["source_floor_projection_rows"], 512 * 61)

    def test_backend_passes_prefill_prefix_and_full_query_tensor(self):
        estimator = self.estimator()
        backend = estimator.backend
        requests = []
        for identifier, prefix in enumerate((65536, 131072)):
            req = Request(identifier, prefix + 512, 20, 256)
            req.num_computed_tokens = prefix
            req.scheduled_tokens = req.scheduled_query_tokens = 512
            requests.append(req)
        req = Request(2, 8192, 20, 256)
        req.generation_idx = 1
        req.num_computed_tokens = 8192
        req.scheduled_tokens = req.scheduled_query_tokens = 1
        requests.append(req)
        with patch.object(estimator, "_query", side_effect=self.module) as query, patch.object(
            estimator.sol, "shared_phase_us", side_effect=self.components
        ), patch.object(backend, "_compose", side_effect=lambda **step: step["attention_us"]):
            value = backend.estimate_step_latency(requests)
        self.assertAlmostEqual(value, 61 * (100 + 2 * 1025 + 30))
        context_calls = [call.args for call in query.call_args_list if call.args[1] == "context"]
        self.assertEqual({call[-1] for call in context_calls}, {65536, 131072})

    def test_composition_export_and_reset_preserve_the_full_time_ledger(self):
        estimator = self.estimator()
        value = estimator.estimate_batch({(512, 65536): 1}, {(8192, 4): 16})
        exported = get_latency_stats(SimpleNamespace(workers=[SimpleNamespace(
            latency_backend=estimator.backend, device=estimator.backend.device)]))
        composition = exported["native_dsv4_attention_composition"]
        result = LLMResult(
            cluster=ClusterConfig.from_file("data/clusters/8_b200/dep8.json"),
            qps=1, request_count=1, prefill_lens=[8192], decode_lens=[20],
            batching="paged-attn", request_time=MetricData.from_list([1]),
            prefill_time=MetricData.from_list([1]), decode_time=MetricData.from_list([1]),
            duration=1, output_qps=1, output_token_ps=20, notdone=[], **exported,
        )
        self.assertEqual(asdict(result)["native_dsv4_attention_composition"], composition)
        self.assertAlmostEqual(composition["shared_projection_seconds"] + composition["phase_seconds"], value * 1e-6)
        self.assertEqual(estimator.decode_batch_stats["query_rows"], 64)
        self.assertEqual(estimator.decode_batch_stats["requests"], 16)
        self.assertEqual(estimator.decode_batch_stats["indexer_query_tiles"], 32)
        self.assertAlmostEqual(estimator.mtp_kv_stats["source_geometry_indexer_bytes_after"] * 2,
                               estimator.mtp_kv_stats["source_geometry_indexer_bytes_before"])
        estimator.backend.reset_profile_stats()
        self.assertEqual(estimator.composition_stats, {})
        self.assertEqual(estimator.seconds, 0)

    def test_empty_batch_has_no_owner_or_time(self):
        estimator = self.estimator()
        self.assertEqual(estimator.estimate_batch({(512, 65536): 0}, {(8192, 4): 0}), 0)
        self.assertEqual(estimator.composition_stats, {})
        self.assertEqual(estimator.seconds, 0)
