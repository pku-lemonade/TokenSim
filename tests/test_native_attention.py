"""Regression tests for measured DSV4 CSA/HCA attention integration."""

from unittest import TestCase

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.operator_data.package import OperatorDataPackage


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
            attention_backend="native_dsv4" if native else "generic",
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
                attention_backend="native_dsv4",
            )

    def test_native_mhc_uses_pre_and_post_without_site_multiplier(self):
        backend = self.backend(native=False, native_moe=False)
        backend = OperatorTableLatencyBackend(
            device=backend.device,
            model=self.model,
            parallel_config=backend.parallel_config,
            rank_info=backend.rank_info,
            package=backend.package,
            mhc_backend="native_dsv4",
        )
        value = backend.native_mhc.estimate(64)
        self.assertGreater(value, 0)
        self.assertEqual(len(backend.native_mhc.counts), 2)
        self.assertEqual(backend.native_mhc.describe()["hc_mult"], 4)


if __name__ == "__main__":
    import unittest

    unittest.main()
