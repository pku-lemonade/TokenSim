"""Native module scope, rank-local geometry and accounting regressions."""

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_moe import MoEPass, RankShape
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.importers.dsv4_megamoe import normalize
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
    OperatorDataValidationError,
)


class NativeMegaMoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = HardwareContext.load("data")
        cls.model = cls.context.model("DeepSeek-V4-Pro")
        cls.device = cls.context.device("b200")

    def package(self, ep=8):
        rows = []
        for phase, tokens in (
            ("generation", (1, 2, 32, 64, 128, 256, 512)),
            ("context", (1024, 2048, 32768)),
        ):
            for n in tokens:
                rows.append(
                    {
                        "dtype": "mxfp4",
                        "phase": phase,
                        "kernel_source": "deepgemm_megamoe",
                        "kernel_dtype": "fp8_fp4",
                        "distribution": "power_law_1.01",
                        "pre_dispatch": "sglang_jit",
                        "source_policy": "random",
                        "num_tokens": n,
                        "hidden_size": 7168,
                        "inter_size": 3072,
                        "top_k": 6,
                        "num_experts": 384,
                        "num_fused_shared_experts": 0,
                        "tp_size": 1,
                        "ep_size": ep,
                        "latency_us": 100 + n,
                        "source_id": "borrowed:test",
                        "includes_gate_topk": False,
                        "includes_buffer_init": False,
                        "used_cuda_graph": True,
                        "includes_routed_scale": True,
                        "num_max_tokens_per_rank": (
                            512 if phase == "generation" else 32768
                        ),
                        "effective_num_max_tokens_per_rank": (
                            576 if phase == "generation" else 32832
                        ),
                    }
                )
        return OperatorDataPackage.from_rows(
            PackageMeta(
                "test",
                "b200_sxm",
                "vllm",
                sources={
                    "borrowed:test": SourceRecord(
                        "borrowed:test",
                        "B",
                        "imported",
                        device="gb200",
                        backend="sglang",
                    )
                },
            ),
            {"dsv4_megamoe": rows},
        )

    def backend(self, *, tp=1, dp=8, fusion=True, communicator=None, device=None):
        parallel = ParallelConfig(
            tensor_parallel_size=tp, data_parallel_size=dp, enable_expert_parallel=True
        )
        return OperatorTableLatencyBackend(
            device=device or self.device,
            model=self.model,
            parallel_config=parallel,
            rank_info=ParallelRankInfo.from_global_rank(0, parallel),
            package=self.package(tp * dp),
            communicator=communicator,
            moe_execution={
                "backend": "deep_gemm_amxf4_mega_moe",
                "shared_expert_fusion": fusion,
            },
            expected_decode_query_len=4,
            max_num_seqs=49,
        )

    def step(self, counts, *, query=1, context=False):
        return MoEPass({i: RankShape(n, query, context) for i, n in enumerate(counts)})

    def test_rank_local_tokens_and_common_max_are_not_multiplied_by_dp(self):
        backend = self.backend()
        backend.set_moe_step(self.step([64, 32, 16, 2, 1, 4, 8, 3]))
        total, routed, shared = backend.native_moe.estimate(64)
        self.assertEqual(routed, 164)
        self.assertGreater(shared, 0)
        self.assertAlmostEqual(total, 61 * (164 + shared))
        self.assertEqual(backend.native_moe.max_rank_rows, 64)

    def test_tp4_sp_splits_leader_rows_among_all_ep_ranks(self):
        backend = self.backend(tp=4, dp=2)
        backend.set_moe_step(self.step([96, 16]))
        rows, own, graph, _ = backend.native_moe.rows_for_group(96)
        self.assertEqual(sorted(rows.values()), [4] * 4 + [24] * 4)
        self.assertEqual(own, 24)
        self.assertFalse(graph)

    def test_graph_padding_and_context_global_eager_use_different_shapes(self):
        backend = self.backend()
        backend.set_moe_step(self.step([12, 4, 8, 4, 4, 4, 4, 4], query=4))
        rows, own, graph, _ = backend.native_moe.rows_for_group(12)
        self.assertTrue(graph)
        self.assertEqual(set(rows.values()), {12})
        backend.set_moe_step(
            self.step([12, 4, 8, 4, 4, 4, 4, 4], query=4, context=True)
        )
        self.assertFalse(backend.native_moe.rows_for_group(12)[2])
        self.assertEqual(backend.native_moe.rows_for_group(12)[0][1], 4)

    def test_tp8_standalone_uses_local_sp_rows(self):
        backend = self.backend(tp=8, dp=1)
        rows, own, _, _ = backend.native_moe.rows_for_group(24)
        self.assertEqual(set(rows.values()), {3})
        self.assertEqual(own, 3)

    def test_uncoordinated_dp_and_cross_node_shapes_are_rejected(self):
        backend = self.backend()
        with self.assertRaises(ConfigurationError):
            backend.native_moe.estimate(64)
        communicator = Mock()
        communicator.describe.return_value = {
            "ep_group_link": "network",
            "ep_group_size": 8,
        }
        backend = self.backend(communicator=communicator)
        backend.set_moe_step(self.step([64] * 8))
        with self.assertRaises(ConfigurationError):
            backend.native_moe.estimate(64)

    def test_native_fused_cost_does_not_query_shared_gemm_or_add_ep_comm(self):
        communicator = Mock()
        communicator.describe.return_value = {
            "ep_group_link": "node",
            "ep_group_size": 8,
        }
        backend = self.backend(communicator=communicator)
        backend.set_moe_step(self.step([64] * 8))
        r = Request(0, 64, 1, 256)
        with patch.object(backend, "_gemm", return_value=0), patch.object(
            backend, "_elementwise", return_value=0
        ), patch.object(backend, "_query", return_value=Mock(latency_us=0)), patch.object(
            backend.native_mtp, "estimate", return_value={}
        ):
            latency = backend._compose(
                tokens=64, requests=1, attention_us=0, requests_list=[r]
            )
        communicator.estimate_ep_all2all.assert_not_called()
        self.assertGreater(latency, 0)
        self.assertAlmostEqual(backend.stats.component_seconds["moe"], latency)
        self.assertEqual(backend.moe_stats.moe_straggler_latency, 0)

    def test_explicit_single_layer_scope_does_not_price_the_target_stack(self):
        backend = self.backend()
        backend.set_moe_step(self.step([64] * 8))
        total, routed, shared = backend.native_moe.estimate(64, layers=1)
        self.assertAlmostEqual(total, routed + shared)
        self.assertAlmostEqual(backend.native_moe.routed_seconds, routed * 1e-6)

    def test_disabled_shared_fusion_prices_two_serial_fp8_matrices_and_activation(self):
        backend = self.backend(fusion=False)
        backend.set_moe_step(self.step([64] * 8))
        with patch.object(backend, "_gemm", return_value=10) as gemm, patch.object(
            backend, "_elementwise", return_value=2
        ):
            _, routed, shared = backend.native_moe.estimate(64)
        self.assertEqual(gemm.call_count, 2)
        self.assertEqual(
            [call.args for call in gemm.call_args_list],
            [(64, 6144, 7168), (64, 7168, 3072)],
        )
        self.assertTrue(
            all(call.kwargs["dtype"] == "fp8_block" for call in gemm.call_args_list)
        )
        self.assertEqual(shared, 22)

    def test_interpolation_gap_retains_cross_buffer_regime(self):
        native = self.backend().native_moe
        value = native._query(768)
        self.assertEqual(value.regime, "cross_buffer_regime")
        self.assertEqual(value.match, "interpolated")
        self.assertEqual(value.latency_us, 868)
        self.assertEqual(value.source_id, "borrowed:test")

    def test_fused_increment_has_no_standalone_launch_and_reset_preserves_cache(self):
        a = self.backend()
        b = self.backend(
            device=replace(
                self.device,
                analytical={**self.device.analytical, "kernel_launch_us": 1e9},
            )
        )
        self.assertEqual(
            a.native_moe.shared_fused_delta(64), b.native_moe.shared_fused_delta(64)
        )
        a.set_moe_step(self.step([64] * 8))
        a.native_moe.estimate(64)
        cache = a.native_moe.cache.copy()
        a.reset_profile_stats()
        self.assertEqual(a.native_moe.cache, cache)
        self.assertEqual(a.native_moe.shared_seconds, 0)
        self.assertFalse(a.native_moe.counts)

    def test_normalized_data_keeps_timing_scope_and_rejects_incompatible_rows(self):
        row = {
            "hidden_size": 7168,
            "inter_size": 3072,
            "topk": 6,
            "num_experts": 384,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "num_fused_shared_experts": 0,
            "used_cuda_graph": True,
            "includes_gate_topk": False,
            "includes_buffer_init": False,
            "includes_routed_scale": True,
            "kernel_source": "deepgemm_megamoe",
            "kernel_dtype": "fp8_fp4",
            "moe_dtype": "w4a8_mxfp4_mxfp8",
            "pre_dispatch": "sglang_jit",
            "source_policy": "random",
            "phase": "generation",
            "num_tokens": 1,
            "global_num_tokens": 8,
            "latency": 0.1,
            "distribution": "power_law_1.01",
            "num_max_tokens_per_rank": 512,
            "effective_num_max_tokens_per_rank": 576,
            "routed_scaling_factor": 2.5,
            "version": "unknown",
        }
        normalized = normalize(
            [row], "test", source_hardware="gb200", collection_version="0.5.10"
        )[0]
        self.assertEqual(normalized["latency_us"], 100)
        self.assertEqual(normalized["recorded_framework_version"], "unknown")
        self.assertEqual(normalized["num_max_tokens_per_rank"], 512)
        with self.assertRaises(OperatorDataValidationError):
            normalize(
                [{**row, "includes_gate_topk": True}],
                "test",
                source_hardware="gb200",
                collection_version="0.5.10",
            )
        with self.assertRaises(OperatorDataValidationError):
            normalize(
                [{**row, "num_fused_shared_experts": 1}],
                "test",
                source_hardware="gb200",
                collection_version="0.5.10",
            )


if __name__ == "__main__":
    unittest.main()
