"""The reusable runner keeps fixed-batch results and CLI compatibility intact."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from scripts import run_deepseek_mla as cli
from scripts.deepseek_v3_runtime import configuration
from tests.support.hardware import make_model


class AFExperimentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)

    def test_runner_matches_factory_timings_statistics_and_full_trace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"experiment": "comparison"}
            runner = AFExperimentRunner(
                self.runtime,
                root / "points",
                fallback="analytical_only",
                profile_metadata=metadata,
            )
            point = runner.run(2, 8, steps=3)
            executor, rows, compute, transfer = simulate_decode(
                self.runtime,
                2,
                8,
                steps=3,
                fallback="analytical_only",
            )
            try:
                path = root / "reference.json"
                write_chrome_trace(path, executor.trace.events, metadata)
                profile = point.directory / point.result["summary"]["profile"]
                self.assertEqual(path.read_bytes(), profile.read_bytes())
                self.assertEqual(point.result["iterations"], rows)
                self.assertEqual(
                    point.result["summary"]["operator_stats"], compute.stats_dict()
                )
                self.assertEqual(
                    point.result["summary"]["transfer"],
                    json.loads(json.dumps(transfer.describe())),
                )
            finally:
                executor.trace.close()
            with patch(
                "TokenSim.af.factory.simulate_decode",
                side_effect=AssertionError("reran"),
            ):
                self.assertTrue(runner.run(2, 8, steps=3).reused)
                restarted = AFExperimentRunner(
                    self.runtime,
                    root / "points",
                    fallback="analytical_only",
                    profile_metadata=metadata,
                )
                self.assertTrue(restarted.run(2, 8, steps=3).reused)

    def test_trace_modes_and_per_point_fields_preserve_throughput(self):
        choices = [
            AFTraceConfig(),
            AFTraceConfig(mode="validate"),
            AFTraceConfig(mode="off"),
            AFTraceConfig(
                event_types=["compute"], iterations=[1], fields=["node_id", "details"]
            ),
        ]
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                self.runtime, directory, fallback="analytical_only"
            )
            points = runner.run_many(
                [
                    dict(batch_size=1, context_len=8, steps=2, trace_config=option)
                    for option in choices
                ]
            )
            self.assertEqual(len({point.key for point in points}), len(choices))
            summaries = []
            for point, option in zip(points, choices):
                summary = dict(point.result["summary"])
                profile = summary.pop("profile")
                summaries.append(summary)
                self.assertEqual(
                    point.result["iterations"], points[0].result["iterations"]
                )
                if option.mode != "full":
                    self.assertIsNone(profile)
                    self.assertFalse(list(point.directory.glob("*.trace.json")))
                elif option.fields is not None:
                    events = json.loads((point.directory / profile).read_text())[
                        "traceEvents"
                    ]
                    self.assertTrue(events)
                    self.assertTrue(all(row["cat"] == "compute" for row in events))
                    self.assertTrue(
                        all(
                            set(row["args"]) == {"node_id", "details"} for row in events
                        )
                    )
            self.assertTrue(all(summary == summaries[0] for summary in summaries))

    def test_seed_context_steps_and_trace_settings_are_distinct_cases(self):
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                self.runtime, directory, fallback="analytical_only"
            )
            cases = [
                dict(),
                dict(seed=0),
                dict(context_len=9),
                dict(steps=2),
                dict(trace_config={"mode": "off"}),
            ]
            points = [
                runner.run(**dict(dict(batch_size=1, context_len=8), **case))
                for case in cases
            ]
            self.assertEqual(len({point.key for point in points}), len(cases))
            for case in cases:
                self.assertTrue(
                    runner.run(**dict(dict(batch_size=1, context_len=8), **case)).reused
                )

    def test_dense_model_uses_the_same_runner_without_deepseek_dependencies(self):
        runtime = build_model_runtime(
            self.runtime.doc,
            hardware=self.runtime.hardware,
            model=make_model(num_layers=2),
        )
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                runtime,
                directory,
                fallback="analytical_only",
                trace_config={"mode": "off"},
            )
            point = runner.run(1, 8)
            self.assertGreater(point.result["summary"]["tokens_per_s"], 0)
            self.assertIsNone(point.result["summary"]["profile"])

    def test_invalid_case_does_not_create_a_checkpoint(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            runner = AFExperimentRunner(self.runtime, output)
            for case in (
                dict(batch_size=0),
                dict(seed=-1),
                dict(steps=True),
                dict(fallback="bad"),
                dict(trace_config={"fields": ["bad"]}),
            ):
                values = dict(batch_size=1, context_len=8)
                values.update(case)
                with self.subTest(case=case), self.assertRaises(ConfigurationError):
                    runner.run(**values)
            self.assertFalse(output.exists())

    def test_registered_table_content_invalidates_even_with_the_same_version(self):
        runtime = configuration()
        source = SourceRecord("measured:runner-test", "A", "measured")

        def package(latency):
            return OperatorDataPackage.from_rows(
                PackageMeta(
                    "same-version",
                    "a100_sxm_80g",
                    "vllm",
                    sources={source.source_id: source},
                ),
                {
                    "gemm": [
                        dict(
                            dtype="bf16",
                            m=1,
                            n=1,
                            k=1,
                            latency_us=latency,
                            source_id=source.source_id,
                        )
                    ]
                },
            )

        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                runtime, directory, trace_config={"mode": "off"}
            )
            runtime.hardware.register_package(package(1.0))
            first = runner.run(1, 8)
            self.assertTrue(runner.run(1, 8).reused)
            runtime.hardware.register_package(package(2.0))
            changed = runner.run(1, 8)
            self.assertFalse(changed.reused)
            self.assertNotEqual(first.key, changed.key)


class ExperimentCLITest(unittest.TestCase):
    def arguments(self, output):
        return [
            "--batches",
            "1",
            "2",
            "--context",
            "8",
            "--fallback",
            "analytical_only",
            "--output",
            str(output),
        ]

    def test_restart_skips_completed_batch_and_retries_interrupted_batch(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            simulate = cli.simulate
            calls = []

            def interrupt(config, batch, *values, **options):
                calls.append(batch)
                if batch == 2:
                    raise KeyboardInterrupt()
                return simulate(config, batch, *values, **options)

            with (
                patch.object(cli, "simulate", side_effect=interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                cli.main(self.arguments(output))
            self.assertEqual(calls, [1, 2])
            completed_trace = output / "decode-layers1-b1.trace.json"
            expected = completed_trace.read_bytes()
            calls.clear()

            def observe(config, batch, *values, **options):
                calls.append(batch)
                return simulate(config, batch, *values, **options)

            with patch.object(cli, "simulate", side_effect=observe):
                cli.main(self.arguments(output))
            self.assertEqual(calls, [2])
            self.assertEqual(completed_trace.read_bytes(), expected)
            summary = (output / "summary.json").read_bytes()
            with patch.object(cli, "simulate", side_effect=AssertionError("reran")):
                cli.main(self.arguments(output))
            self.assertEqual((output / "summary.json").read_bytes(), summary)
            cached = list(
                (output / ".experiments").glob("*/decode-layers1-b1.trace.json")
            )
            self.assertEqual(len(cached), 1)
            self.assertEqual(completed_trace.stat().st_ino, cached[0].stat().st_ino)

    def test_json_trace_configuration_can_select_exact_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "trace-config.json"
            config.write_text(
                '{"event_types":["compute"],"fields":["node_id","payload_bytes"]}'
            )
            output = root / "output"
            cli.main([*self.arguments(output), "--trace-config", str(config)])
            for path in output.glob("*.trace.json"):
                rows = json.loads(path.read_text())["traceEvents"]
                self.assertTrue(rows)
                self.assertTrue(
                    all(
                        set(row["args"]) == {"node_id", "payload_bytes"} for row in rows
                    )
                )
