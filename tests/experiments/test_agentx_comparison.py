"""Accuracy reports keep regressions, incomplete runs, and unsupported metrics visible."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.experiments.agentx_comparison import compare_case, compare_manifest


def prediction(output=100, ttft=1.0, tpot=0.05):
    return dict(
        termination_reason="completed",
        notdone=[],
        agentx_metrics=dict(
            concurrency=4,
            request_count=10,
            issue_window_s=3600,
            output_token_throughput_tps=output,
            input_token_throughput_tps=1000,
            ttft_s=dict(p50=ttft, p90=ttft),
            tpot_s=dict(p50=tpot, p90=tpot),
            gpu_cache_hit_rate=0.9,
            cpu_cache_hit_rate=0.02,
        ),
    )


def reference():
    return dict(
        reference_id=42,
        hardware="test",
        concurrency=4,
        window=dict(issue_s=3600),
        metrics=dict(
            output_tput_tps=100,
            input_tput_tps=1000,
            request_count=10,
            ttft_s=dict(p50=1.0, p90=1.0),
            itl_s=dict(p50=0.05, p90=0.05),
            gpu_cache_hit_rate=0.8,
            cpu_cache_hit_rate=0.1,
        ),
    )


class AgentXComparisonTest(unittest.TestCase):
    def test_improvement_and_regression_remain_separate(self):
        before, after = prediction(output=80, ttft=1.1), prediction(output=90, ttft=1.2)
        original = deepcopy((before, after))
        report = compare_case(before, after, reference())
        rows = {r["metric"]: r for r in report["metrics"]}
        self.assertEqual(report["status"], "valid")
        self.assertEqual(rows["output_tps"]["status"], "improved")
        self.assertAlmostEqual(rows["output_tps"]["baseline_error_percent"], -20)
        self.assertAlmostEqual(rows["output_tps"]["current_error_percent"], -10)
        self.assertEqual(rows["ttft_p90_s"]["status"], "regressed")
        self.assertIn("ttft_p90_s", report["critical_regressions"])
        self.assertEqual((before, after), original)
        self.assertIn("cohorts differ", report["cache_diagnostics"]["comparison_scope"])

    def test_failures_missing_metrics_and_nan_cannot_pass(self):
        mutations = [
            ("termination_reason", "failed"),
            ("notdone", [12]),
            ("notdone", None),
        ]
        for key, value in mutations:
            current = prediction()
            current[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(
                    compare_case(prediction(), current, reference())["status"],
                    "invalid",
                )
        for value in (None, 0, float("nan"), float("inf"), True):
            current = prediction()
            current["agentx_metrics"]["output_token_throughput_tps"] = value
            with self.subTest(value=value):
                report = compare_case(prediction(), current, reference())
                self.assertEqual(report["status"], "invalid")
                row = next(r for r in report["metrics"] if r["metric"] == "output_tps")
                self.assertEqual(row["status"], "unavailable")
                self.assertNotIn("current_error_percent", row)

    def test_reference_window_and_concurrency_are_required(self):
        for key, value in (
            ("concurrency", 8),
            ("issue_window_s", 2),
            ("request_count", 0),
        ):
            current = prediction()
            current["agentx_metrics"][key] = value
            with self.subTest(key=key):
                self.assertEqual(
                    compare_case(prediction(), current, reference())["status"],
                    "invalid",
                )

    def test_manifest_retains_fingerprints_and_rejects_changed_commands(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root / "refs"
            refs.mkdir()
            (refs / "inferencex_key_points.json").write_text(json.dumps([reference()]))
            cases = []
            for label in ("baseline", "current"):
                output = root / label
                output.mkdir()
                (output / "agentx_c4.json").write_text(json.dumps(prediction()))
                cases.append(
                    dict(
                        hardware="test",
                        label=label,
                        reference_id=42,
                        concurrency=4,
                        output=str(output),
                        args=["--max_num_seqs", "4", "--results_path", str(output)],
                    )
                )
            manifest = dict(reference_directory=str(refs), cases=cases)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            report = compare_manifest(path)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(len(report["manifest_sha256"]), 64)
            self.assertEqual(
                set(report["cases"][0]["artifacts"]), {"baseline", "current"}
            )
            wrong_reference = reference()
            wrong_reference["hardware"] = "other"
            (refs / "inferencex_key_points.json").write_text(
                json.dumps([wrong_reference])
            )
            self.assertEqual(compare_manifest(path)["cases"][0]["status"], "invalid")
            (refs / "inferencex_key_points.json").write_text(
                json.dumps([reference(), reference()])
            )
            with self.assertRaises(ConfigurationError):
                compare_manifest(path)
            (refs / "inferencex_key_points.json").write_text(json.dumps([reference()]))
            cases[1]["args"][1] = "8"
            path.write_text(json.dumps(manifest))
            self.assertEqual(compare_manifest(path)["cases"][0]["status"], "invalid")
            cases[1]["args"][1] = "4"
            (root / "current/agentx_c4.json").unlink()
            path.write_text(json.dumps(manifest))
            self.assertEqual(compare_manifest(path)["cases"][0]["status"], "incomplete")
            cases.append(cases[0])
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ConfigurationError):
                compare_manifest(path)
