"""Serial clients, overload detection and request-speed quantiles."""

from pathlib import Path
import unittest

from TokenSim.config.model_config import ModelSpec
from TokenSim.hardware import HardwareContext
from scripts.lpx.capacity import search, classify
from scripts.lpx.config import Workload, Case, Layout
from scripts.lpx.native_simulation import point
from scripts.lpx.requests import request_metrics


class RequestExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = ModelSpec(
            "tiny",
            32,
            64,
            2,
            4,
            2,
            8,
            vocab_size=64,
            dtype="bf16",
            max_position_embeddings=1024,
        )
        cls.workload = Workload(cached_tokens=16, new_tokens=2, output_tokens=8)
        cls.case = Case("base", "rubin", Layout(a_tp=2, f_ep=2))

    def test_median_generation_speed_is_not_decode_time_weighted_rate(self):
        workload = Workload(output_tokens=11)
        turns = [
            dict(
                cohort=i,
                round=0,
                start=0,
                end=1 + seconds,
                ttft_s=1,
                completed=True,
                warmup=False,
            )
            for i, seconds in enumerate((10, 1, 0.1))
        ]
        stats = request_metrics(turns, workload, 3, 1, {"start": 0, "end": 12})
        self.assertEqual(stats["generation_tps"]["p50"], 10)
        self.assertEqual(stats["completed_requests"], 3)
        self.assertAlmostEqual(stats["little_law_complete_request_users"], 14.1 / 12)

    def test_serial_client_submits_only_after_previous_completion(self):
        result = point(
            self.hardware,
            self.model,
            self.workload,
            self.case,
            1,
            warmup_steps=1,
            measurement_turns=3,
            measurement_seconds=1e-9,
            stream_metrics=True,
        )
        rows = result["request_statistics"]["requests"]
        self.assertEqual(len(rows), 3)
        for previous, following in zip(rows, rows[1:]):
            self.assertEqual(previous["last_output_at"], following["submitted_at"])
        self.assertEqual(result["arrivals"]["pending_at_end"], 0)

    def test_open_overload_remains_visible_at_the_measurement_boundary(self):
        base = point(
            self.hardware,
            self.model,
            self.workload,
            self.case,
            1,
            warmup_steps=1,
            measurement_turns=1,
            measurement_seconds=1e-9,
        )
        latency = base["request_statistics"]["latency_s"]["mean"]
        busy = point(
            self.hardware,
            self.model,
            self.workload,
            self.case,
            1,
            warmup_steps=1,
            measurement_turns=0,
            measurement_seconds=10 * latency,
            qps=4 / latency,
            stream_metrics=True,
        )
        self.assertGreater(busy["arrivals"]["pending_at_end"], 0)
        self.assertGreater(busy["arrivals"]["admission_wait_s"]["p95"], 0.01 * latency)
        self.assertFalse(classify(busy, latency)["passed"])
        self.assertEqual(
            busy["arrivals"]["arrivals"] - busy["arrivals"]["admissions"],
            busy["arrivals"]["pending_at_end"],
        )

    def test_capacity_refines_and_rechecks_the_window(self):
        def evaluate(qps, multiplier):
            passed = qps <= 3
            return dict(
                status="ok",
                arrivals=dict(
                    arrivals=100,
                    admissions=100 if passed else 80,
                    pending_at_end=0 if passed else 20,
                    admission_wait_s={"p95": 0 if passed else 1},
                    backlog_growth_per_s=0 if passed else 1,
                ),
                request_statistics={"completed_requests": 99 if passed else 79},
            )

        result = search(evaluate, 1)
        self.assertLessEqual(result["maximum_passing_qps"], 3)
        self.assertGreater(result["maximum_passing_qps"], 2.94)
        self.assertTrue(result["doubled_window_passed"])
        self.assertEqual(result["probes"][-1]["window_multiplier"], 2)


if __name__ == "__main__":
    unittest.main()
