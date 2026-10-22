"""Numerical boundaries and metric controls for LPX context sensitivity."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.explore_lpx_context import constant_gap_fit
from scripts.lpx.aa_model import Capacity, GemmaText
from scripts.lpx.context_sensitivity import (
    TARGET_RATIO,
    clear_caches,
    generation,
    make_costs,
    solve_bounded,
)


ROOT = Path(__file__).resolve().parents[1]


class ContextSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")
        raw = json.loads((ROOT / "experiments/lpx/aa-gemma4.json").read_text())
        cls.model = GemmaText(**raw["model"])
        capacity = Capacity(
            cls.hardware, cls.model, weight_dtype="bf16", kv_dtype="bf16"
        )
        cls.lpx = dict(
            name="lpx",
            mode="lpx",
            decode=(capacity.place("groq_lp30", 8, "full", 102000),),
            prefill=None,
        )
        prefill = capacity.place("rubin", 8, "full", 102000)
        cls.af = dict(
            name="af",
            mode="af",
            decode=(
                capacity.place("rubin", 8, "attention", 102000),
                capacity.place("groq_lp30", 12, "ffn", 102000),
            ),
            prefill=prefill,
        )
        cls.global_af = dict(
            name="af-global",
            mode="af-global",
            decode=(
                capacity.place("rubin", 8, "global_attention", 102000),
                capacity.place("groq_lp30", 32, "local_ffn", 102000),
            ),
            prefill=prefill,
        )
        cls.short = dict(name="10k", input_tokens=10000, output_tokens=1500)
        cls.long = dict(name="100k", input_tokens=100000, output_tokens=2000)

    def tearDown(self):
        clear_caches()

    def test_scaling_changes_both_lpu_bandwidth_fields_without_mutating_catalog(self):
        original = self.hardware.device("groq_lp30")
        costs = make_costs(
            self.hardware, self.model, dict(memory_scale=2, compute_scale=3)
        )
        modified = costs.devices["groq_lp30"]
        self.assertEqual(
            modified.memory_bandwidth_bytes_per_s.value,
            original.memory_bandwidth_bytes_per_s.value * 2,
        )
        self.assertEqual(
            modified.on_chip_bandwidth_bytes_per_s.value,
            original.on_chip_bandwidth_bytes_per_s.value * 2,
        )
        self.assertEqual(
            modified.peak_compute_for("bf16").value,
            original.peak_compute_for("bf16").value * 3,
        )
        self.assertIs(self.hardware.device("groq_lp30"), original)

    def test_constant_parameters_preserve_context_order_for_all_modes(self):
        for settings in (
            {},
            dict(kernel_us=0, link_latency_scale=0, memory_scale=16, compute_scale=16),
            dict(kernel_us=1, memory_scale=0.25, link_latency_scale=5),
            dict(global_attention_efficiency=0.01),
        ):
            costs = make_costs(self.hardware, self.model, settings)
            for case in (self.lpx, self.af, self.global_af):
                rates = [
                    generation(
                        costs, case, dict(input_tokens=context, output_tokens=8)
                    )["tps"]
                    for context in (10000, 20000, 50000, 100000)
                ]
                self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
            clear_caches()

    def test_attention_efficiency_changes_only_global_memory_term(self):
        baseline = make_costs(self.hardware, self.model)
        altered = make_costs(
            self.hardware, self.model, {"global_attention_efficiency": 0.09}
        )
        for full in (False, True):
            a = baseline.attention("groq_lp30", 8, full, context=20000)
            b = altered.attention("groq_lp30", 8, full, context=20000)
            self.assertAlmostEqual(
                b.memory_time_us / a.memory_time_us, 10 if full else 1
            )
            self.assertEqual(a.compute_time_us, b.compute_time_us)
            self.assertEqual(a.overhead_us, b.overhead_us)

    def test_bisection_checks_reachability_and_both_directions(self):
        self.assertAlmostEqual(
            solve_bounded(lambda x: x * 2, 3, 0, 2, increasing=True), 1.5
        )
        self.assertAlmostEqual(
            solve_bounded(lambda x: 4 - x * 2, 3, 0, 2, increasing=False), 0.5
        )
        self.assertIsNone(solve_bounded(lambda x: x, 3, 0, 2, increasing=True))

    def test_length_dependent_fixed_cost_can_reverse_output_speed(self):
        baseline = make_costs(self.hardware, self.model)
        first = generation(baseline, self.lpx, self.short)["tps"]

        def response(value):
            costs = make_costs(self.hardware, self.model, {"kernel_us": value})
            result = generation(costs, self.lpx, self.long)["tps"]
            clear_caches()
            return result

        value = solve_bounded(response, first * TARGET_RATIO, 0, 0.2, increasing=False)
        self.assertIsNotNone(value)
        self.assertLess(value, 0.2)
        self.assertAlmostEqual(response(value) / first, TARGET_RATIO, places=10)
        rates = []
        for context in (10000, 20000, 50000, 100000):
            fraction = (context - 10000) / 90000
            costs = make_costs(
                self.hardware,
                self.model,
                {"kernel_us": 0.2 + fraction * (value - 0.2)},
            )
            rates.append(
                generation(
                    costs, self.lpx, dict(input_tokens=context, output_tokens=2000)
                )["tps"]
            )
            clear_caches()
        self.assertTrue(all(a < b for a, b in zip(rates, rates[1:])))

    def test_fixed_stream_gap_reversal_requires_unequal_output_lengths(self):
        costs = make_costs(self.hardware, self.model)
        short = generation(costs, self.lpx, self.short)
        long = generation(costs, self.lpx, self.long)
        gap = (
            TARGET_RATIO * long["mean_native_step_us"] - short["mean_native_step_us"]
        ) / (1 / short["output_intervals"] - TARGET_RATIO / long["output_intervals"])
        a = generation(costs, self.lpx, self.short, post_first_gap_us=gap)
        b = generation(costs, self.lpx, self.long, post_first_gap_us=gap)
        self.assertAlmostEqual(b["tps"] / a["tps"], TARGET_RATIO, places=12)
        controlled = generation(
            costs,
            self.lpx,
            {**self.short, "output_tokens": 2000},
            post_first_gap_us=gap,
        )
        self.assertGreater(controlled["tps"], b["tps"])
        with patch.object(costs, "prefill", return_value=1e12):
            self.assertEqual(generation(costs, self.lpx, self.short), short)

    def test_invalid_settings_are_rejected(self):
        for settings in (
            {"unknown": 1},
            {"memory_scale": 0},
            {"link_latency_scale": -1},
            {"global_attention_efficiency": 0},
            {"memory_efficiency": 1.01},
            {"kernel_us": -1},
        ):
            with self.assertRaises(ConfigurationError):
                make_costs(self.hardware, self.model, settings)

    def test_shared_kernel_and_stream_gap_fit_matches_both_rates(self):
        costs = make_costs(self.hardware, self.model, {"kernel_us": 0.1})
        targets = tuple(
            generation(costs, self.lpx, workload, post_first_gap_us=200000)["tps"]
            for workload in (self.short, self.long)
        )
        fitted = constant_gap_fit(
            self.hardware, self.model, self.lpx, self.short, self.long, targets=targets
        )
        self.assertTrue(fitted["feasible"])
        self.assertAlmostEqual(fitted["required_kernel_us"], 0.1, places=10)
        self.assertAlmostEqual(fitted["post_first_gap_us"], 200000, places=7)
        self.assertAlmostEqual(fitted["short"]["tps"], targets[0], places=8)
        self.assertAlmostEqual(fitted["long"]["tps"], targets[1], places=8)
        self.assertLess(fitted["equal_output_ratio"], 1)
        impossible = constant_gap_fit(
            self.hardware,
            self.model,
            self.global_af,
            self.short,
            self.long,
            targets=(10000, 10100),
        )
        self.assertFalse(impossible["feasible"])
        self.assertLess(impossible["required_kernel_us"], 0)
        equal_outputs = constant_gap_fit(
            self.hardware,
            self.model,
            self.lpx,
            {**self.short, "output_tokens": 2000},
            self.long,
            targets=targets,
        )
        self.assertFalse(equal_outputs["feasible"])
        self.assertIn("equal answer lengths", equal_outputs["reason"])


if __name__ == "__main__":
    unittest.main()
