"""Saved formats preserve runner results, selections and checkpoint reuse."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.formats import TRACE_SUFFIXES
from TokenSim.errors import ConfigurationError
from scripts import run_deepseek_mla as cli
from scripts.deepseek_v3_runtime import configuration

FORMATS = ("json", "archive") + (
    ("msgpack",) if importlib.util.find_spec("msgpack") else ()
)


class ExperimentTraceFormatsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)

    def test_formats_preserve_json_values_and_resume_independently(self):
        for options in (
            AFTraceConfig(),
            AFTraceConfig(
                iterations=[1], event_types=["compute"], fields=["node_id", "details"]
            ),
        ):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                root = Path(directory)
                runner = AFExperimentRunner(
                    self.runtime,
                    root / "points",
                    trace_config=options,
                    fallback="analytical_only",
                    profile_metadata={"description": '解码\n"trace"'},
                )
                points, summaries, traces = [], [], []
                for trace_format in FORMATS:
                    point = runner.run(1, 8, steps=2, trace_format=trace_format)
                    points.append(point)
                    self.assertEqual(point.result["trace_format"], trace_format)
                    profile = point.directory / point.result["summary"]["profile"]
                    self.assertEqual(profile.suffix, TRACE_SUFFIXES[trace_format])
                    converted = runner.export_trace_json(
                        point, root / f"{trace_format}.json"
                    )
                    traces.append(converted.read_bytes())
                    converted.unlink()
                    # A saved path also works after the producing process exits.
                    runner.export_trace_json(profile, converted)
                    self.assertEqual(converted.read_bytes(), traces[-1])
                    summary = dict(point.result["summary"])
                    summary.pop("profile")
                    summaries.append(summary)
                self.assertEqual(len({point.key for point in points}), len(FORMATS))
                self.assertTrue(all(trace == traces[0] for trace in traces))
                self.assertTrue(all(row == summaries[0] for row in summaries))
                self.assertTrue(
                    all(
                        point.result["iterations"] == points[0].result["iterations"]
                        for point in points
                    )
                )
                with patch(
                    "TokenSim.af.factory.simulate_decode",
                    side_effect=AssertionError("reran"),
                ):
                    for trace_format in FORMATS:
                        self.assertTrue(
                            runner.run(1, 8, steps=2, trace_format=trace_format).reused
                        )

    def test_default_and_run_many_overrides(self):
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                self.runtime,
                directory,
                trace_format="archive",
                fallback="analytical_only",
            )
            points = runner.run_many(
                [
                    dict(batch_size=1, context_len=8),
                    dict(batch_size=1, context_len=8, trace_format="json"),
                ]
            )
            self.assertEqual(
                [point.result["trace_format"] for point in points], ["archive", "json"]
            )
            self.assertTrue(runner.run(1, 8).reused)

    def test_invalid_format_and_missing_dependency_fail_before_output_creation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            for value in ("unknown", None, [], 42):
                with self.subTest(value=value), self.assertRaises(ConfigurationError):
                    AFExperimentRunner(self.runtime, output, trace_format=value)
            runner = AFExperimentRunner(self.runtime, output)
            with self.assertRaises(ConfigurationError):
                runner.run(1, 8, trace_format="unknown")
            with patch.dict(sys.modules, {"msgpack": None}):
                with self.assertRaisesRegex(ConfigurationError, r"tokensim\[trace\]"):
                    runner.run(1, 8, trace_format="msgpack")
            self.assertFalse(output.exists())

    def test_disabled_modes_need_no_binary_dependency_and_have_no_saved_trace(self):
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                self.runtime, directory, fallback="analytical_only"
            )
            with patch.dict(sys.modules, {"msgpack": None}):
                for mode in ("off", "validate"):
                    point = runner.run(
                        1, 8, trace_format="msgpack", trace_config={"mode": mode}
                    )
                    self.assertIsNone(point.result["summary"]["profile"])
                    with self.assertRaisesRegex(ConfigurationError, "no saved trace"):
                        runner.export_trace_json(point, Path(directory) / "trace.json")
                    self.assertFalse(list(point.directory.glob("*.trace.*")))

    def test_cli_uses_selected_suffix_and_preserves_default_public_summary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            traces, summaries = [], []
            for trace_format in FORMATS:
                output = root / trace_format
                arguments = [
                    "--batches",
                    "1",
                    "--context",
                    "8",
                    "--steps",
                    "2",
                    "--fallback",
                    "analytical_only",
                    "--output",
                    str(output),
                ]
                if trace_format != "json":
                    arguments.extend(("--trace-format", trace_format))
                cli.main(arguments)
                summary = json.loads((output / "summary.json").read_text())
                if trace_format == "json":
                    self.assertNotIn("trace_format", summary)
                else:
                    self.assertEqual(summary.pop("trace_format"), trace_format)
                profile = summary["results"][0].pop("profile")
                self.assertEqual(Path(profile).suffix, TRACE_SUFFIXES[trace_format])
                converted = AFExperimentRunner.export_trace_json(
                    output / profile, root / f"{trace_format}-converted.json"
                )
                traces.append(converted.read_bytes())
                summaries.append(summary)
                with patch.object(cli, "simulate", side_effect=AssertionError("reran")):
                    cli.main(arguments)
            self.assertTrue(all(trace == traces[0] for trace in traces))
            self.assertTrue(all(summary == summaries[0] for summary in summaries))
