"""Analytic and native-provider oracles for coupled GPU/SRAM decode bounds."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.errors import ConfigurationError
from scripts.bound_lpx_sqd_residency import search
from scripts.lpx.fidelity import load
from scripts.lpx.sqd_bounds import (
    batch_tail_bound,
    coupled_bound,
    finite_inventory_bound,
    gpu_global_cost,
    sliding_kv_bytes,
)
from scripts.run_lpx_article_sweep import article_config
from scripts.run_lpx_context_parallel import environment


class SQDCoupledBoundTest(unittest.TestCase):
    def settings(self):
        return dict(
            useful_flops=12,
            gpu_path_s=2,
            gpu_tp=2,
            batch=4,
            gpu_peak=10,
            lp_peak=5,
            sliding_bytes=20,
            sram_bytes=100,
        )

    def test_littles_law_keeps_kv_during_gpu_execution(self):
        # T=10 requires N>=20, at least four SRAM chips and ten GPUs.
        # Useful rate 120 / allocated peak 120 gives this relaxed 100% bound.
        result = coupled_bound(**self.settings())
        self.assertEqual(result["mfu_bound_percent"], 100)
        self.assertEqual(result["gpu_peak_seconds_per_token"], 10)
        self.assertEqual(result["lp_memory_peak_seconds_per_token"], 2)

    def test_lp_arithmetic_and_memory_are_independent_requirements(self):
        result = coupled_bound(**self.settings(), lp_compute_peak_seconds=5)
        self.assertEqual(result["mfu_bound_percent"], 80)
        # A chip can store cache while doing its matrix work: use max, not sum.
        self.assertEqual(
            coupled_bound(**self.settings(), lp_compute_peak_seconds=1)[
                "mfu_bound_percent"
            ],
            100,
        )

    def test_inventory_optimizer_matches_exhaustive_allocations(self):
        settings = self.settings()
        for limit in (7, 14, 23):
            best = 0
            for lp in range(1, limit):
                for gpu in range(2, limit - lp + 1, 2):
                    resident = (lp * 100 - 150) / 20
                    if resident <= 0:
                        continue
                    rate = min(gpu, resident / 2, lp * 5 / 3)
                    best = max(best, 100 * 12 * rate / (gpu * 10 + lp * 5))
            result = finite_inventory_bound(
                max_cards=limit,
                lp_weight_bytes=150,
                **settings,
                lp_compute_peak_seconds=3,
            )
            self.assertAlmostEqual(result["mfu_bound_percent"], best)
        self.assertIsNone(
            finite_inventory_bound(max_cards=3, lp_weight_bytes=150, **settings)
        )

    def test_finite_weight_bound_converges_below_inventory_free_bound(self):
        settings = self.settings()
        small = finite_inventory_bound(max_cards=14, lp_weight_bytes=150, **settings)
        large = finite_inventory_bound(max_cards=140, lp_weight_bytes=150, **settings)
        self.assertLess(small["mfu_bound_percent"], large["mfu_bound_percent"])
        self.assertLessEqual(
            large["mfu_bound_percent"], coupled_bound(**settings)["mfu_bound_percent"]
        )

    def test_tail_covers_unbounded_integer_batches(self):
        settings = {
            k: v for k, v in self.settings().items() if k not in {"batch", "gpu_path_s"}
        }
        tail = batch_tail_bound(
            minimum_batch=8, core_memory_s_per_request=0.25, **settings
        )
        for batch in (8, 9, 64, 1_000_000):
            # Any additional GPU work can only tighten the bound.
            result = coupled_bound(
                **settings, batch=batch, gpu_path_s=batch * 0.25 + 0.1
            )
            self.assertLessEqual(result["mfu_bound_percent"], tail["mfu_bound_percent"])

    def test_invalid_dimensions_and_nonfinite_values_fail(self):
        for key, value in (
            ("gpu_tp", True),
            ("batch", 0),
            ("gpu_path_s", float("nan")),
            ("sram_bytes", 0),
            ("sliding_bytes", -1),
        ):
            with self.subTest(key=key), self.assertRaises(ConfigurationError):
                coupled_bound(**(self.settings() | {key: value}))
        with self.assertRaises(ConfigurationError):
            batch_tail_bound(
                minimum_batch=8, core_memory_s_per_request=1, **self.settings()
            )


class SQDNativeCostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = article_config(load("experiments/lpx/fidelity-study.json"))
        cls.config = config
        cls.hardware, cls.model, _, _ = environment(config)
        cls.provider = OperatorTableAFComputeProvider(
            cls.hardware,
            cls.model,
            fallback="analytical_only",
            graph_device_kernel_us=0.2,
        )

    def test_sliding_payload_uses_native_unreplicated_heads(self):
        # 50 windowed layers, 1024 tokens, 16 heads, separate FP8 K and V.
        self.assertEqual(sliding_kv_bytes(self.model, 102000), 50 * 1024 * 16 * 256 * 2)
        self.assertEqual(sliding_kv_bytes(self.model, 17), 50 * 17 * 16 * 256 * 2)
        with self.assertRaises(ConfigurationError):
            sliding_kv_bytes(replace(self.model, kv_cache_dtype="bf16"), 102000)

    def test_core_ownership_and_linear_traffic_tail(self):
        full = gpu_global_cost(
            self.provider, "h100_sxm", 4, 1, 101000, "global_attention"
        )
        core = gpu_global_cost(self.provider, "h100_sxm", 4, 1, 101000, "global_core")
        batched = gpu_global_cost(
            self.provider, "h100_sxm", 4, 17, 101000, "global_core"
        )
        self.assertEqual(
            [row["layer_id"] for row in core["layers"]], list(range(5, 60, 6))
        )
        self.assertTrue(
            all(
                [p["name"] for p in row["components"]] == ["attention_core"]
                for row in core["layers"]
            )
        )
        self.assertAlmostEqual(
            core["gpu_path_s"], sum(row["seconds"] for row in core["layers"])
        )
        self.assertGreater(full["gpu_path_s"], core["gpu_path_s"])
        self.assertAlmostEqual(
            core["core_memory_s_per_request"], batched["core_memory_s_per_request"]
        )
        self.assertGreaterEqual(
            batched["gpu_path_s"], 17 * core["core_memory_s_per_request"]
        )

    def test_replication_does_not_halve_core_time_beyond_kv_head_count(self):
        four = gpu_global_cost(self.provider, "h100_sxm", 4, 2, 101000, "global_core")
        eight = gpu_global_cost(self.provider, "h100_sxm", 8, 2, 101000, "global_core")
        # Four global KV heads: additional head-TP ranks replicate KV traffic.
        self.assertGreater(eight["gpu_path_s"], four["gpu_path_s"] / 2)
        self.assertGreater(
            eight["core_memory_s_per_request"], four["core_memory_s_per_request"] / 2
        )

    def test_unsupported_ownership_precision_and_width_fail(self):
        for role, tp in (
            ("all_attention", 4),
            ("global_core", 3),
            ("global_core", True),
        ):
            with self.subTest(role=role, tp=tp), self.assertRaises(ConfigurationError):
                gpu_global_cost(self.provider, "h100_sxm", tp, 1, 101000, role)
        shared = replace(self.model, global_kv_layout="value_rope_fp8")
        provider = OperatorTableAFComputeProvider(
            self.hardware, shared, fallback="analytical_only"
        )
        with self.assertRaises(ConfigurationError):
            gpu_global_cost(provider, "h100_sxm", 4, 1, 101000, "global_core")

    def test_driver_preserves_empty_finite_inventory_frontiers(self):
        with patch("builtins.print"):
            result = search(self.config, maximum_batch=1, card_limits=(2,))
        self.assertEqual(len(result["rows"]), 36)
        self.assertEqual(len(result["batch_tails"]), 36)
        self.assertTrue(
            all(row is None for row in result["best_finite_bounds"].values())
        )
        self.assertTrue(
            all(row["finite_inventory"]["2"] is None for row in result["rows"])
        )


if __name__ == "__main__":
    unittest.main()
