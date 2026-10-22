"""Independent interpolation, lifecycle and search-boundary checks."""

from dataclasses import replace
import gc
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

from TokenSim.comm.streaming import StreamingConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.native_decode import DecodeCurve, interval_summary, serial_replay
from scripts.lpx.native_search import (
    WORKLOADS,
    acceptance,
    build_runtime,
    layouts,
    memory_report,
    screen,
    streaming_settings,
    summarize,
    with_startup,
)
from scripts.lpx.replay_profiles import capture_profile
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.streaming import StreamingBuilder
from scripts.search_lpx_native import main, verification_candidates

ROOT = Path(__file__).resolve().parents[1]


class NativeCurveTest(unittest.TestCase):
    def test_adaptive_roofline_and_window_knot_match_integer_oracle(self):
        def oracle(context):
            return 1e-6 * (2 + max(4, context / 5) + min(context, 11) / 7)

        curve = DecodeCurve(lambda c: dict(latency_s=oracle(c)), 1, 40, knots=[11])
        for context in range(1, 41):
            self.assertAlmostEqual(curve(context), oracle(context), delta=1e-10)
        result = interval_summary(curve, 0, 41)
        self.assertAlmostEqual(
            result["generation_seconds"], sum(oracle(c) for c in range(1, 41))
        )
        self.assertGreater(len(curve.samples), 5)
        self.assertTrue(curve.check([3, 8, 17, 33]))

    def test_withheld_probes_detect_unsampled_nonlinearity(self):
        # Endpoints and midpoint agree; the quarter-point exposes the error.
        curve = DecodeCurve(
            lambda c: dict(latency_s=1e-3 + math.sin(math.pi * c / 4) ** 2 * 1e-4),
            4,
            12,
        )
        self.assertEqual(len(curve.samples), 3)
        with self.assertRaises(SimulationStateError):
            curve.check([6])
        for context in (3, 13, True):
            with self.assertRaises(ConfigurationError):
                curve(context)

    def test_guard_refines_hidden_kinks_at_the_original_tolerance(self):
        def oracle(context):
            return 1e-3 + abs((context - 4) % 8 - 4) * 2e-8

        curve = DecodeCurve(lambda c: dict(latency_s=oracle(c)), 4, 20)
        curve.guard()
        self.assertTrue(curve.guarded)
        self.assertTrue(curve.guard_refinements)
        self.assertEqual(curve.tolerance_s, 1e-10)
        for context in range(4, 21):
            self.assertAlmostEqual(curve(context), oracle(context), delta=1e-10)
        curve.check([5, 7, 10, 15, 19])

    def test_single_context_and_invalid_estimates(self):
        curve = DecodeCurve(lambda c: dict(latency_s=2e-6), 5, 5)
        self.assertEqual(curve(5), 2e-6)
        for value in (0, -1, float("nan")):
            with self.assertRaises(ConfigurationError):
                DecodeCurve(lambda c: dict(latency_s=value), 1, 2)

    def test_missing_workloads_do_not_pass_acceptance(self):
        rows = [
            dict(input_tokens=i, output_tokens=o, generation_tps=3405)
            for i, o, _ in WORKLOADS
        ]
        result = acceptance(rows)
        self.assertTrue(result["common_3400"])
        self.assertFalse(result["published_targets"])
        for invalid in ([], rows[:2], list(reversed(rows))):
            with self.assertRaises(ConfigurationError):
                acceptance(invalid)

    def test_search_covers_every_allowed_stage_count(self):
        model = SimpleNamespace(
            num_attention_heads=32, intermediate_size=21504, num_layers=60
        )
        candidates = list(layouts(model, [8, 16, 32], 256))
        self.assertEqual(len(candidates), 32 + 16 + 8)
        self.assertIn((16, 9), [(r.a_tp, r.a_pp) for r in candidates])
        self.assertEqual(len(list(layouts(model, [8, 16], 256, [9, 16]))), 4)
        with self.assertRaises(ConfigurationError):
            list(layouts(model, [3], 256))
        self.assertEqual(main(["--max-chips", "257"]), 1)


class NativeSearchIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")
        cls.model = replace(
            cls.hardware.models.resolve("gemma-4-31b-text"),
            dtype="fp8",
            kv_cache_dtype="fp8",
        )
        cls.layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=16,
            f_tp=16,
            a_pp=9,
            f_pp=9,
            f_ep=1,
        )

    def test_native_memory_rejects_smaller_pp_and_reserves_protocol_buffers(self):
        runtime = build_runtime(self.hardware, self.model, self.layout)
        report = memory_report(runtime, 102000)
        peak = report["peak_rank"]
        self.assertEqual(report["ranks"], 144)
        self.assertEqual(
            peak["total_bytes"],
            peak["weights_bytes"] + peak["buffers_bytes"] + peak["kv_bytes"],
        )
        self.assertGreater(report["minimum_headroom_bytes"], 0)
        self.assertGreater(peak["buffers_bytes"], 0)
        too_small = replace(self.layout, a_pp=8, f_pp=8)
        with self.assertRaises(ConfigurationError):
            memory_report(build_runtime(self.hardware, self.model, too_small), 102000)

    def test_collective_cache_does_not_retain_finished_dag_samples(self):
        workers = [
            SimpleNamespace(
                hardware="groq_lp30",
                physical_worker_ref=SimpleNamespace(identity=f"g:{i}", worker_index=i),
            )
            for i in range(2)
        ]
        runtime = SimpleNamespace(hardware=self.hardware, workers=workers)
        builder = StreamingBuilder(
            runtime, [], NativeProfile((), {}), StreamingConfig()
        )
        estimate = builder.collective(("g:0", "g:1"), "all_reduce", 16, "bf16")
        self.assertIs(
            estimate, builder.collective(("g:0", "g:1"), "all_reduce", 16, "bf16")
        )
        reference = weakref.ref(builder)
        del builder
        gc.collect()
        self.assertIsNone(reference())

    def test_serial_forward_advances_every_token_and_releases_each_request(self):
        runtime = build_runtime(self.hardware, self.model, self.layout)
        curve = DecodeCurve(lambda c: dict(latency_s=c * 1e-6), 1001, 1005)
        result = serial_replay(
            runtime, curve, inputs=1000, outputs=6, requests=3, warmup=2
        )
        self.assertEqual(result["executed_decode_steps"], 25)
        self.assertEqual(result["generation_tps"]["count"], 3)
        self.assertAlmostEqual(
            result["generation_tps"]["mean"],
            5 / sum(c * 1e-6 for c in range(1001, 1006)),
        )
        rows = result["request_timings"]
        self.assertEqual([r["warmup"] for r in rows], [True, True, False, False, False])
        self.assertTrue(all(r["final_context"] == 1005 for r in rows))
        self.assertTrue(
            all(
                b["first_output_at"] == a["last_output_at"]
                for a, b in zip(rows, rows[1:])
            )
        )
        with self.assertRaises(ConfigurationError):
            serial_replay(runtime, curve, inputs=999, outputs=6)

    def test_native_probe_uses_actual_reuse_and_streaming_and_local_overrides(self):
        # Real DAG probes, with tiny token ranges; no historical TPS assertions.
        workloads = ((1000, 2, 3400), (10000, 2, 3382), (100000, 2, 3431))
        with patch("scripts.lpx.native_search.WORKLOADS", workloads):
            result = screen(
                self.hardware,
                self.model,
                self.layout,
                verify=True,
                requests=2,
                warmup=1,
            )
        self.assertEqual(result["status"], "ok")
        for row in result["workloads"]:
            probe = row["interpolation"]["probes"][0]
            self.assertGreater(probe["streaming_collectives"], 0)
            self.assertEqual(row["serial_replay"]["executed_decode_steps"], 3)
        runtime = build_runtime(self.hardware, self.model, self.layout)
        baseline = capture_profile(
            runtime,
            "search",
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=10001,
            dp_rank=0,
            seed=42,
            sample=0,
            streaming=streaming_settings(self.layout),
        )
        self.assertAlmostEqual(
            baseline.metadata["modeled_duration_s"],
            result["workloads"][1]["interpolation"]["probes"][0]["latency_s"],
        )
        altered = with_startup(self.hardware, 0.3)
        self.assertEqual(
            self.hardware.device("groq_lp30").analytical["kernel_launch_us"], 0.2
        )
        self.assertEqual(
            altered.device("groq_lp30").analytical["kernel_launch_us"], 0.3
        )
        self.assertTrue(
            StreamingConfig(
                **streaming_settings(replace(self.layout, a_tp=32, f_tp=32))
            ).overlap
            is False
        )

    def test_minimum_selection_and_verification_include_smaller_failure(self):
        def point(pp, tps):
            rows = [
                dict(input_tokens=i, output_tokens=o, generation_tps=tps)
                for i, o, _ in WORKLOADS
            ]
            return dict(
                status="ok",
                lp30=pp * 16,
                layout=vars(replace(self.layout, a_pp=pp, f_pp=pp)),
                acceptance=acceptance(rows),
            )

        rows = [point(8, 3399), point(9, 3420), point(10, 3450), point(16, 3401)]
        result = summarize(rows)
        self.assertEqual(result["common_3400"]["lp30"], 144)
        self.assertEqual(result["published_targets"]["lp30"], 160)
        choices = verification_candidates(rows, result, 4)
        self.assertEqual([r["lp30"] for r in choices], [144, 160, 128, 256])


if __name__ == "__main__":
    unittest.main()
