"""Common Rubin normalization and explicit processor-rack power sensitivity."""

import unittest

from scripts.report_lpx_study import power_rows


class LPXReportTest(unittest.TestCase):
    def test_common_baseline_and_rack_rounding_preserve_measured_throughput(self):
        rows = [
            dict(
                status="ok",
                mode="rubin",
                inventory={"rubin": 8},
                power_watts=14400,
                output_tokens_per_s=1000,
            ),
            dict(
                status="ok",
                mode="af",
                inventory={"rubin": 24, "groq_lp30": 2560},
                power_watts=1195200,
                output_tokens_per_s=2000,
            ),
        ]
        allocated, baseline = power_rows(rows, "allocated_processors")
        self.assertEqual(allocated[0]["normalized_throughput"], 1)
        self.assertEqual(
            allocated[1]["normalized_throughput"],
            allocated[1]["tokens_per_s_per_mw"] / baseline,
        )
        provisioned, _ = power_rows(rows, "processor_racks")
        self.assertEqual(provisioned[0]["power_watts"], 129600)
        self.assertEqual(provisioned[1]["power_watts"], 1281600)
        self.assertEqual(provisioned[1]["minimum_lpx_racks"], 10)
        self.assertEqual(provisioned[1]["output_tokens_per_s"], 2000)
        self.assertEqual(rows[0]["power_watts"], 14400)


if __name__ == "__main__":
    unittest.main()
