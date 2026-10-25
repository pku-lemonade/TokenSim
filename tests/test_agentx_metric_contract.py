"""AgentX slow-tail speeds invert latency percentiles at the same rank."""

from pathlib import Path
import unittest

import numpy as np

from TokenSim.llm.llm_request import Request
from TokenSim.reporting.metrics import RunReport, get_agentx_metrics


class AgentXMetricContractTest(unittest.TestCase):
    def report(self, intervals):
        requests = []
        for index, interval in enumerate(intervals):
            req = Request(index, 8, 2, 16)
            req.generation_idx = 2
            req.prefill_latency = 1.0
            req.decode_time_sum = interval
            req.total_time = 1.0 + interval
            req.record_prefill_admission(0, 0)
            requests.append(req)
        run = RunReport(
            Path("."),
            1.0,
            "paged-attn",
            agentx_concurrency=4,
            agentx_metadata={"replay_protocol": "aiperf-49634d1c-native"},
        )
        return get_agentx_metrics(run, requests, 2.0)

    def test_interactivity_uses_matching_slow_latency_percentiles(self):
        intervals = [0.01, 0.02, 0.04, 0.08]
        result = self.report(intervals)["interactivity_tps"]
        for percentile in (50, 75, 90, 95, 99):
            with self.subTest(percentile=percentile):
                self.assertAlmostEqual(
                    result[f"p{percentile}"], 1 / np.percentile(intervals, percentile)
                )
        self.assertAlmostEqual(result["mean"], 1 / np.mean(intervals))
        self.assertAlmostEqual(result["max"], 1 / min(intervals))

    def test_normalized_interactivity_uses_latency_per_output_token(self):
        intervals = [0.01, 0.02, 0.04, 0.08]
        result = self.report(intervals)["e2e_normalized_interactivity_tps"]
        seconds_per_token = [(1.0 + value) / 2 for value in intervals]
        for percentile in (50, 75, 90, 95, 99):
            with self.subTest(percentile=percentile):
                self.assertAlmostEqual(
                    result[f"p{percentile}"],
                    1 / np.percentile(seconds_per_token, percentile),
                )
        self.assertAlmostEqual(result["mean"], 1 / np.mean(seconds_per_token))

    def test_native_empty_speed_statistics_remain_unavailable(self):
        result = self.report([])
        for name in ("interactivity_tps", "e2e_normalized_interactivity_tps"):
            self.assertEqual(result[name]["count"], 0)
            self.assertIsNone(result[name]["mean"])
            self.assertIsNone(result[name]["p90"])

    def test_uniform_latencies_have_identical_speed_statistics(self):
        result = self.report([0.04] * 4)["interactivity_tps"]
        self.assertEqual(result["mean"], 25)
        self.assertEqual(result["p90"], 25)

    def test_zero_itl_is_excluded_from_both_latency_and_speed_samples(self):
        result = self.report([0.0, 0.01, 0.04])
        self.assertEqual(result["tpot_s"]["count"], 2)
        self.assertEqual(result["interactivity_tps"]["count"], 2)
        self.assertEqual(
            result["root_tpot_s"]["count"] + result["subagent_tpot_s"]["count"], 2
        )
        self.assertAlmostEqual(
            result["interactivity_tps"]["p90"], 1 / result["tpot_s"]["p90"]
        )

    def test_nonfinite_intervals_do_not_contaminate_percentiles(self):
        result = self.report([float("nan"), float("inf"), -1.0, 0.04])
        self.assertEqual(result["tpot_s"]["count"], 1)
        self.assertEqual(result["interactivity_tps"]["mean"], 25)
