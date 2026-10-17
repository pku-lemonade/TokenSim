"""Trace choices preserve simulation results and expose their output limits."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.af.factory import simulate_decode
from TokenSim.af.trace.config import AFTraceConfig, TIMING_FIELDS
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.af.trace.transient import ValidationTraceRecorder
from TokenSim.errors import ConfigurationError
from scripts import run_deepseek_mla as cli
from scripts.deepseek_v3_runtime import configuration, write_profile
from tests.af.trace.test_export import reference_payload
from tests.af.trace.test_metrics import event


class TraceConfigurationTest(unittest.TestCase):
    def test_rejects_invalid_values_and_conflicting_choices(self):
        invalid = [
            dict(mode="unknown"),
            dict(validate=1),
            dict(detail="small"),
            dict(event_types="compute"),
            dict(event_types=["unknown"]),
            dict(iterations=[True]),
            dict(iterations=[-1]),
            dict(iterations="0"),
            dict(layers=[-2]),
            dict(layers=[1.5]),
            dict(mode="validate", validate=False),
            dict(mode="off", event_types=["compute"]),
            dict(mode="validate", detail="timing"),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                AFTraceConfig(**values)

    def test_copies_mutable_inputs_and_describes_effective_behavior(self):
        layers, kinds = [-1, 0, 0], ["compute"]
        config = AFTraceConfig(layers=layers, event_types=kinds)
        layers.append(1)
        kinds.append("transfer")
        self.assertEqual(config.layers, (-1, 0))
        self.assertEqual(config.event_types, ("compute",))
        self.assertEqual(config.describe()["filter_scope"], "export_only")
        self.assertEqual(
            config.describe()["dependency_scope"], "may_reference_omitted_events"
        )
        disabled = AFTraceConfig(mode="off").describe()
        for name in (
            "records_events",
            "validates_events",
            "retains_events",
            "exports_events",
        ):
            self.assertFalse(disabled[name])

    def test_filters_intersect_and_keep_event_order_and_dependency_ids(self):
        rows = [
            event("omitted", 0, 1),
            replace(event("keep-one", 1, 2, predecessors=("omitted",)), iteration_id=2),
            replace(event("wrong-layer", 2, 3), iteration_id=2, layer_id=1),
            replace(event("wrong-type", 3, 4), iteration_id=2, event_type="transfer"),
            replace(event("keep-two", 4, 5), iteration_id=2),
        ]
        config = AFTraceConfig(event_types=["compute"], iterations=[2], layers=[0])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            write_chrome_trace(
                output, iter(rows), {"experiment": "sample"}, trace_config=config
            )
            result = json.loads(output.read_text())
        self.assertEqual(
            result["traceEvents"],
            reference_payload([rows[1], rows[4]], {})["traceEvents"],
        )
        self.assertEqual(result["metadata"]["experiment"], "sample")
        self.assertEqual(result["metadata"]["trace_config"]["iterations"], [2])
        self.assertEqual(
            result["traceEvents"][0]["args"]["predecessor_ids"], ["omitted"]
        )

    def test_default_export_is_identical_and_empty_selection_is_valid(self):
        rows = [event("one", 0, 1)]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            write_chrome_trace(output, rows)
            original = output.read_bytes()
            write_chrome_trace(output, rows, trace_config=AFTraceConfig())
            self.assertEqual(output.read_bytes(), original)
            for selection in (AFTraceConfig(iterations=[]), AFTraceConfig(layers=[9])):
                write_chrome_trace(output, rows, trace_config=selection)
                self.assertEqual(json.loads(output.read_text())["traceEvents"], [])

    def test_timing_detail_omits_payload_without_touching_original_records(self):
        row = event("one", 0, 1, unsupported=object())
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            write_chrome_trace(
                output, [row], trace_config=AFTraceConfig(detail="timing")
            )
            result = json.loads(output.read_text())["traceEvents"][0]
            self.assertEqual(set(result["args"]), set(TIMING_FIELDS))
            self.assertEqual(result["ts"], row.start_at * 1e6)
            self.assertEqual(result["dur"], row.service_duration * 1e6)
            self.assertIn("unsupported", row.details)
            for mode in ("off", "validate"):
                with self.assertRaisesRegex(ConfigurationError, "requires full mode"):
                    write_chrome_trace(
                        output, [row], trace_config=AFTraceConfig(mode=mode)
                    )
            self.assertEqual(json.loads(output.read_text())["traceEvents"][0], result)


class TraceRuntimeConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)

    def test_all_modes_preserve_rows_and_provider_observations(self):
        results = []
        validate = AFTraceRecorder.validate_invariants
        for options in (
            AFTraceConfig(),
            AFTraceConfig(validate=False),
            AFTraceConfig(mode="validate"),
            AFTraceConfig(mode="off"),
        ):
            checked = []

            def inspect(recorder, iteration_id):
                checked.append((iteration_id, len(recorder.events_for(iteration_id))))
                return validate(recorder, iteration_id)

            with patch.object(AFTraceRecorder, "validate_invariants", inspect):
                executor, rows, compute, transfer = simulate_decode(
                    self.runtime,
                    2,
                    8,
                    steps=4,
                    fallback="analytical_only",
                    trace_config=options,
                )
            try:
                results.append((rows, compute.stats_dict(), transfer.describe()))
                self.assertEqual(
                    len(checked), 4 if options.enabled and options.validate else 0
                )
                self.assertTrue(all(count > 0 for _, count in checked))
                if options.mode == "validate":
                    self.assertIsInstance(executor.trace, ValidationTraceRecorder)
                    self.assertEqual(executor.trace.events.retained_count, 0)
                    self.assertEqual(executor.trace.iteration_event_indices, {})
                elif options.mode == "off":
                    self.assertEqual(len(executor.trace.events), 0)
                    self.assertIsNone(executor.trace.archive.path)
                else:
                    self.assertTrue(executor.trace.events_for(0))
            finally:
                executor.trace.close()
        self.assertTrue(all(value == results[0] for value in results[1:]))

    def test_validation_mode_releases_interleaved_iterations_with_stable_positions(
        self,
    ):
        results = []
        for mode in ("full", "validate"):
            executor, _, compute, transfer = simulate_decode(
                self.runtime,
                1,
                8,
                fallback="analytical_only",
                trace_config=AFTraceConfig(mode=mode),
            )
            completed = []

            def stream(cohort):
                for step in range(4):
                    requests = [
                        SimpleNamespace(
                            id=cohort,
                            is_prefill=False,
                            prefill_len=8,
                            context_len=8 + step,
                            generation_idx=step,
                        )
                    ]
                    iteration = executor.submit(0, requests, "decode")
                    yield executor.env.process(executor.run_iteration(iteration))
                    completed.append((iteration.iteration_id, iteration.makespan))

            try:
                for cohort in range(3):
                    executor.env.process(stream(cohort))
                executor.env.run()
                results.append((completed, compute.stats_dict(), transfer.describe()))
                if mode == "validate":
                    history = executor.trace.events
                    self.assertEqual(history.retained_count, 0)
                    self.assertLess(history.peak_retained, len(history))
                    self.assertFalse(executor.trace.iteration_event_indices)
            finally:
                executor.trace.close()
        self.assertEqual(results[0], results[1])

    def test_filtered_export_uses_complete_patched_validated_events(self):
        options = AFTraceConfig(
            event_types=["transfer"], iterations=[1], detail="timing"
        )
        executor, _, _, _ = simulate_decode(
            self.runtime,
            2,
            8,
            steps=2,
            fallback="analytical_only",
            trace_config=options,
        )
        try:
            events = list(executor.trace.events)
            self.assertTrue(any(row.event_type == "compute" for row in events))
            executor.trace.validate_invariants(1)
            with TemporaryDirectory() as directory:
                output = write_profile(Path(directory) / "profile.json", executor)
                rows = json.loads(output.read_text())["traceEvents"]
            selected = [row for row in events if options.includes(row)]
            self.assertTrue(rows)
            self.assertEqual(
                [row["args"]["node_id"] for row in rows],
                [row.node_id for row in selected],
            )
            self.assertEqual(
                [row["args"]["predecessor_ids"] for row in rows],
                [list(row.predecessor_ids) for row in selected],
            )
        finally:
            executor.trace.close()

    def test_legacy_trace_false_and_invalid_new_configuration(self):
        for options in ("off", AFTraceConfig()):
            with self.assertRaises(ConfigurationError):
                simulate_decode(self.runtime, 1, 8, trace=False, trace_config=options)
        executor, _, _, _ = simulate_decode(
            self.runtime, 1, 8, trace=False, fallback="analytical_only"
        )
        try:
            with TemporaryDirectory() as directory:
                output = write_profile(Path(directory) / "empty.json", executor)
                self.assertEqual(json.loads(output.read_text())["traceEvents"], [])
        finally:
            executor.trace.close()


class TraceCLITest(unittest.TestCase):
    def test_cli_modes_and_filters_preserve_performance_values(self):
        results = []
        variants = [
            [],
            ["--trace", "off"],
            ["--trace", "validate"],
            [
                "--trace-event-types",
                "compute",
                "--trace-iterations",
                "1",
                "--trace-layers",
                "0",
                "--trace-detail",
                "timing",
                "--no-trace-validation",
            ],
        ]
        for extra in variants:
            with self.subTest(extra=extra), TemporaryDirectory() as directory:
                cli.main(
                    [
                        "--batches",
                        "1",
                        "--context",
                        "8",
                        "--steps",
                        "2",
                        "--fallback",
                        "analytical_only",
                        "--output",
                        directory,
                        *extra,
                    ]
                )
                output = Path(directory)
                summary = json.loads((output / "summary.json").read_text())
                row = summary["results"][0]
                profile = row.pop("profile")
                results.append(row)
                if extra and extra[0] == "--trace":
                    self.assertIsNone(profile)
                    self.assertFalse(list(output.glob("*.trace.json")))
                    self.assertEqual(summary["trace_config"]["mode"], extra[1])
                else:
                    payload = json.loads((output / profile).read_text())
                    if extra:
                        self.assertTrue(payload["traceEvents"])
                        for item in payload["traceEvents"]:
                            self.assertEqual(item["cat"], "compute")
                            self.assertEqual(item["args"]["iteration_id"], 1)
                            self.assertEqual(item["args"]["layer_id"], 0)
                            self.assertNotIn("tensor_slices", item["args"])
                    else:
                        self.assertNotIn("trace_config", summary)
                        self.assertNotIn("trace_config", payload["metadata"])
        self.assertTrue(all(row == results[0] for row in results[1:]))

    def test_invalid_cli_selection_fails_before_creating_outputs(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with self.assertRaises(SystemExit) as error:
                cli.main(
                    ["--output", str(output), "--trace", "off", "--trace-layers", "0"]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())
