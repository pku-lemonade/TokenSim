"""Common Rubin normalization and explicit processor-rack power sensitivity."""

import unittest

from scripts.report_lpx_study import power_rows, combine_indexes, observed_comparisons
from scripts.lpx_power import processor_power_rows


class LPXReportTest(unittest.TestCase):
    def test_gain_comparisons_use_observed_feasible_points(self):
        rows = [
            dict(
                case=case,
                mode=mode,
                batch=users,
                tokens_per_s_per_user=speed,
                tokens_per_s_per_mw=efficiency,
                normalized_throughput=efficiency / 100,
            )
            for case, mode, users, speed, efficiency in (
                ("gpu-fast", "rubin", 1, 1000, 20),
                ("gpu-efficient", "rubin", 8, 100, 100),
                ("af-fast", "af", 4, 2000, 60),
                ("af-efficient", "af", 16, 80, 120),
            )
        ]
        result = observed_comparisons(rows, (500, 1500), (0.5, 1.3))
        at_speed = result["minimum_generation_tps"][0]["modes"]
        self.assertEqual(at_speed["af"]["case"], "af-fast")
        self.assertEqual(at_speed["af"]["relative_to_rubin"], 3)
        self.assertIsNone(at_speed["draft"])
        no_gpu = result["minimum_generation_tps"][1]["modes"]
        self.assertIsNone(no_gpu["rubin"])
        self.assertIsNone(no_gpu["af"]["relative_to_rubin"])
        at_efficiency = result["minimum_normalized_efficiency"][0]["modes"]
        self.assertEqual(at_efficiency["af"]["relative_to_rubin"], 20)
        self.assertTrue(
            all(
                row is None
                for row in result["minimum_normalized_efficiency"][1]["modes"].values()
            )
        )

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
