"""Main capture/SP compute ownership without changing sparse query progress."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from TokenSim.config.config import ClusterConfig
from TokenSim.config.psla_config import LLMResult, MetricData
from TokenSim.errors import ConfigurationError
from TokenSim.latency.attention import AttentionEstimate
from TokenSim.llm.forward import ForwardPass, RankShape
from util.results import get_latency_stats
import tests.test_attention_composition as composition
import tests.test_full_mtp as full_mtp


class CaptureWorkTest(TestCase):
    @classmethod
    def setUpClass(cls):
        full_mtp.FullMTPTest.setUpClass()
        composition.AttentionCompositionTest.setUpClass()

    def backend_case(self, tp, dp, *, eager=False, width=4):
        helper = full_mtp.FullMTPTest()
        backend = helper.backend(tp=tp, dp=dp)
        requests = helper.decode_requests(20 // width, width=width)
        backend.set_moe_step(ForwardPass({rank: RankShape(
            20 if rank == 0 else (512 if eager else 64),
            width if rank == 0 else (None if eager else 4),
            eager and rank != 0, len(requests) if rank == 0 else (1 if eager else 16),
        ) for rank in range(dp)}))
        with patch.object(backend.native_mhc, "estimate", wraps=backend.native_mhc.estimate) as mhc, \
                patch.object(backend, "gemm", wraps=backend._gemm) as gemm, \
                patch.object(backend.native_attention, "estimate_batch", wraps=backend.native_attention.estimate_batch) as attention, \
                patch.object(backend.native_mtp, "estimate", return_value={}):
            backend.estimate_step_latency(requests)
        router = [call.args[0] for call in gemm.call_args_list
                  if call.args[1:3] == (384, 7168)]
        self.assertEqual(len(router), 1)
        buckets = attention.call_args.args[1]
        self.assertEqual(sum(query * count for (_context, query), count in buckets.items()), 20)
        return backend, mhc.call_args.args[0], router[0], attention.call_args.kwargs["projection_rows"]

    def test_tp1_capture_main_owners_use64_rows_but_keep20_queries(self):
        backend, mhc, router, projections = self.backend_case(1, 8)
        self.assertEqual((mhc, router, projections), (64, 64, 64))
        self.assertEqual(backend.target_logits_counts["forward_rows"], 20)
        self.assertEqual(backend.target_logits_counts["output_forward_rows"], 64)
        self.assertEqual(backend.target_logits_counts["rows"], 20)
        self.assertEqual(backend.main_forward_counts["capture_padding_rows"], 44)
        self.assertEqual(backend.main_forward_counts["norm_layer_rows"], 64 * 122)

    def test_tp4_main_norm_and_router_own_full_capture_shards(self):
        backend, mhc, router, projections = self.backend_case(4, 2)
        self.assertEqual((mhc, router, projections), (16, 16, 64))
        self.assertEqual(backend.main_forward_counts["sp_rows"], 16)
        self.assertEqual(backend.main_forward_counts["router_layer_rows"], 16 * 61)
        self.assertEqual(backend.target_logits_counts["output_forward_rows"], 64)

    def test_remote_context_forces_eager_and_router_keeps_sp_shard(self):
        backend, mhc, router, projections = self.backend_case(4, 2, eager=True)
        self.assertEqual((mhc, router, projections), (5, 5, 20))
        self.assertEqual(backend.main_forward_counts["capture_padding_rows"], 0)
        self.assertEqual(backend.main_forward_counts["graph_padded_steps"], 0)
        self.assertEqual(backend.native_attention.composition_stats.get("capture_batches", 0), 0)

    def test_tp8_tail_padding_is_owned_before_gather_then_sliced(self):
        backend, mhc, router, projections = self.backend_case(8, 1)
        self.assertEqual((mhc, router, projections), (3, 3, 20))
        self.assertEqual(backend.main_forward_counts["sp_tail_rows"], 4)
        self.assertEqual(backend.target_logits_counts["output_forward_rows"], 20)

    def test_staged_q1_keeps_actual_eager_tensor_and_sp_owners(self):
        backend, mhc, router, projections = self.backend_case(8, 1, width=1)
        self.assertEqual((mhc, router, projections), (3, 3, 20))
        self.assertEqual(backend.main_forward_counts["capture_padding_rows"], 0)

    def test_export_and_reset_preserve_main_row_ledger(self):
        backend, *_ = self.backend_case(1, 8)
        exported = get_latency_stats(SimpleNamespace(workers=[SimpleNamespace(
            latency_backend=backend, device=backend.device)]))
        ledger = exported["main_forward_counts"]
        self.assertEqual(ledger["full_rows"], 64)
        result = LLMResult(cluster=ClusterConfig.from_file("data/clusters/8_b200/dep8.json"),
                           qps=1, request_count=1, prefill_lens=[8192], decode_lens=[20],
                           batching="paged-attn", request_time=MetricData.from_list([1]),
                           prefill_time=MetricData.from_list([1]), decode_time=MetricData.from_list([1]),
                           duration=1, output_qps=1, output_token_ps=20, notdone=[], **exported)
        self.assertEqual(asdict(result)["main_forward_counts"], ledger)
        backend.reset_profile_stats()
        self.assertEqual(backend.main_forward_counts, {})

    def estimator(self):
        return composition.AttentionCompositionTest().estimator()

    def test_capture_adds_only_common_work_and_queries_module_once(self):
        estimator = self.estimator()
        with patch.object(estimator, "_query", side_effect=composition.AttentionCompositionTest.module) as queries, \
                patch.object(estimator.sol, "shared_phase_us", side_effect=composition.AttentionCompositionTest.components):
            value = estimator.estimate_batch({}, {(8192, 1): 20}, projection_rows=64)
        self.assertAlmostEqual(value, 61 * (100 + 2 * 64 + 10))
        self.assertEqual(queries.call_count, 2)
        self.assertEqual(estimator.decode_batch_stats["query_rows"], 20)
        self.assertEqual(estimator.decode_batch_stats["indexer_query_tiles"], 20)
        self.assertEqual(estimator.composition_stats["capture_increment_layer_rows"], 44 * 61)

    def test_source_floor_already_covering_capture_has_no_extra_cost(self):
        estimator = self.estimator()

        def clamped(_kind, _phase, _batch, _queries, context):
            return AttentionEstimate(110 + 2 * 128, "clamped", "fixture", False, 128, context, 1)

        with patch.object(estimator, "_query", side_effect=clamped), \
                patch.object(estimator.sol, "shared_phase_us", side_effect=composition.AttentionCompositionTest.components):
            value = estimator.estimate_batch({}, {(8192, 1): 20}, projection_rows=64)
        self.assertAlmostEqual(value, 61 * (110 + 2 * 128))
        self.assertEqual(estimator.composition_stats["capture_shared_increment_seconds"], 0)
        self.assertEqual(estimator.composition_stats["capture_increment_layer_rows"], 0)

    def test_ragged_source_floors_allocate_positive_increment_per_branch(self):
        estimator = self.estimator()

        def clamped(_kind, _phase, _batch, _queries, context):
            floor = 64 if context == 8192 else 128
            return AttentionEstimate(110 + 2 * floor, "clamped", "fixture", False, floor, context, 1)

        with patch.object(estimator, "_query", side_effect=clamped), \
                patch.object(estimator.sol, "shared_phase_us", side_effect=composition.AttentionCompositionTest.components):
            value = estimator.estimate_batch({}, {(8192, 1): 10, (32768, 1): 10}, projection_rows=96)
        self.assertAlmostEqual(value, 61 * 334)
        self.assertEqual(estimator.composition_stats["capture_source_floor_layer_rows"], 61 * 96)
        self.assertEqual(estimator.composition_stats["capture_projection_bound_layer_rows"], 61 * 112)

    def test_context_and_invalid_capture_geometry_are_explicit(self):
        estimator = self.estimator()
        for context, decode, rows in (({}, {(8192, 1): 20}, 16),
                                      ({(512, 65536): 1}, {}, 1024),
                                      ({(512, 65536): 1}, {(8192, 1): 4}, 1024)):
            with self.subTest(context=context, rows=rows), self.assertRaises(ConfigurationError):
                estimator.estimate_batch(context, decode, projection_rows=rows)
