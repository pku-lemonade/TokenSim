"""Common Rubin normalization and explicit processor-rack power sensitivity."""

import unittest

from scripts.report_lpx_study import power_rows, combine_indexes
from scripts.lpx_power import processor_power_rows


class LPXReportTest(unittest.TestCase):
    def test_power_scope_rejects_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "power scope"):
            processor_power_rows([], "facility")

    def test_combined_frontiers_keep_serving_variants_distinct(self):
        base = dict(
            model={"model_id": "proxy"},
            draft_model={},
            workload={},
            source_sha256="source",
            state="complete",
            phase="formal",
            cases=[{"name": "ruby"}],
            results=[{"case": "ruby", "batch": 4, "status": "ok"}],
        )
        merged = combine_indexes(
            [dict(base, variant="mb1-adp1"), dict(base, variant="mb4-adp1")]
        )
        self.assertEqual(
            {r["case"] for r in merged["results"]}, {"ruby@mb1-adp1", "ruby@mb4-adp1"}
        )
        self.assertEqual(merged["state"], "complete")
        with self.assertRaises(ValueError):
            combine_indexes([base, dict(base, source_sha256="different")])

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
