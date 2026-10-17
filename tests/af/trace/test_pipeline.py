"""Production prefix export preserves complete traces and atomic file behavior."""

from dataclasses import replace
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.af.factory import simulate_decode
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.pipeline import PipelinedTraceRecorder
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.deepseek_v3_runtime import configuration
from tests.af.support.observations import _valid_events


class PreparedTraceTest(unittest.TestCase):
    def test_invalid_configuration_and_archive_failure_cleanup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for options in (
                "full",
                AFTraceConfig(mode="off"),
                AFTraceConfig(mode="validate"),
            ):
                with self.assertRaises(ConfigurationError):
                    PipelinedTraceRecorder(path, trace_config=options)
            recorder = PipelinedTraceRecorder(path)
            self.addCleanup(recorder.close)
            source = _valid_events()[0]
            recorder.record(source)
            with patch.object(
                recorder.archive,
                "append_iteration",
                side_effect=OSError("archive full"),
            ):
                with self.assertRaisesRegex(SimulationStateError, "archive full"):
                    recorder.complete_iteration(source.iteration_id)
            self.assertEqual(list(path.glob("*.tmp")), [])
            self.assertEqual(recorder.events.retained_count, 1)

    def assert_export_matches(self, recorder, output, *, options=None):
        reference = output.with_name("reference.json")
        # Explicit base iteration avoids invalidating the prepared path.
        from TokenSim.af.trace.archive import ArchivedEventHistory

        write_chrome_trace(
            reference,
            ArchivedEventHistory.__iter__(recorder.events),
            {"description": '解码\n"trace"'},
            trace_config=options,
        )
        write_chrome_trace(
            output,
            recorder.events,
            {"description": '解码\n"trace"'},
            trace_config=options,
        )
        self.assertEqual(output.read_bytes(), reference.read_bytes())

    def test_order_snapshots_selection_and_publication_without_readback(self):
        for options in (
            AFTraceConfig(),
            AFTraceConfig(detail="timing"),
            AFTraceConfig(iterations=[8], event_types=["compute"]),
            AFTraceConfig(iterations=[]),
            AFTraceConfig(validate=False),
        ):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                path = Path(directory)
                recorder = PipelinedTraceRecorder(path, trace_config=options)
                self.addCleanup(recorder.close)
                source = _valid_events()[0]
                mutable = {"snapshot": [1]}
                for index, iteration in enumerate((7, 8, 7, 9, 8, 9)):
                    recorder.record(
                        replace(
                            source,
                            iteration_id=iteration,
                            node_id=str(index),
                            details=mutable,
                        )
                    )
                recorder.complete_iteration(8)
                recorder.complete_iteration(9)
                self.assertEqual(recorder.events.encoded_count, 0)
                recorder.complete_iteration(7)
                mutable["snapshot"].append(2)
                self.assertEqual(recorder.events.retained_count, 0)
                # Build the reference before disabling archive reads.
                self.assert_export_matches(
                    recorder, path / "prepared.json", options=options
                )
                self.assertTrue(recorder.published)
                recorder.clear()
                recorder.record(source)
                recorder.complete_iteration(source.iteration_id)
                with patch.object(
                    recorder.archive,
                    "items",
                    side_effect=AssertionError("archive reread"),
                ):
                    write_chrome_trace(
                        path / "direct.json", recorder.events, trace_config=options
                    )
                recorder.close()
                self.assertEqual(list(path.glob("*.tmp")), [])

    def test_replacement_partial_repeat_different_options_and_directory(self):
        source = _valid_events()[0]
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for situation in (
                "replacement",
                "partial",
                "repeat",
                "options",
                "directory",
            ):
                with self.subTest(situation=situation):
                    recorder = PipelinedTraceRecorder(path)
                    self.addCleanup(recorder.close)
                    recorder.record(source)
                    recorder.complete_iteration(source.iteration_id)
                    options, output = None, path / "observed.json"
                    if situation == "replacement":
                        recorder.events[0] = replace(source, source_id="late patch")
                    elif situation == "partial":
                        recorder.record(
                            replace(source, iteration_id=source.iteration_id + 1)
                        )
                    elif situation == "repeat":
                        write_chrome_trace(output, recorder.events)
                    elif situation == "options":
                        options = AFTraceConfig(detail="timing")
                    else:
                        output = path / "elsewhere" / "observed.json"
                        output.parent.mkdir()
                    self.assert_export_matches(recorder, output, options=options)
                    self.assertTrue(recorder.events.invalidated)
                    recorder.close()
                    self.assertEqual(list(path.glob("*.tmp")), [])

    def test_archived_readback_can_expose_mutable_payload(self):
        for access in ("index", "iterate", "iteration"):
            with self.subTest(access=access), TemporaryDirectory() as directory:
                path = Path(directory)
                recorder = PipelinedTraceRecorder(path)
                self.addCleanup(recorder.close)
                source = replace(_valid_events()[0], details={"snapshot": [1]})
                recorder.record(source)
                recorder.complete_iteration(source.iteration_id)
                if access == "index":
                    observed = recorder.events[0]
                elif access == "iterate":
                    observed = next(iter(recorder.events))
                else:
                    observed = recorder.events_for(source.iteration_id)[0]
                observed.details["snapshot"].append(2)
                self.assert_export_matches(recorder, path / "observed.json")
                self.assertTrue(recorder.events.invalidated)

    def test_empty_output_failures_and_abandoned_recorders(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            recorder = PipelinedTraceRecorder(path)
            self.assert_export_matches(recorder, path / "empty.json")
            recorder.close()
            destination = path / "trace.json"
            destination.write_text("previous result")
            for failure in ("encode", "write", "metadata", "replace", "abandon"):
                with self.subTest(failure=failure):
                    recorder = PipelinedTraceRecorder(path)
                    source = _valid_events()[0]
                    if failure == "encode":
                        source = replace(source, details={"unsupported": object()})
                    recorder.record(source)
                    if failure == "encode":
                        with self.assertRaises(TypeError):
                            recorder.complete_iteration(source.iteration_id)
                    elif failure == "write":
                        with patch.object(
                            recorder._writer().stream,
                            "write",
                            side_effect=OSError("disk full"),
                        ):
                            with self.assertRaisesRegex(OSError, "disk full"):
                                recorder.complete_iteration(source.iteration_id)
                    else:
                        recorder.complete_iteration(source.iteration_id)
                        if failure == "metadata":
                            with self.assertRaises(TypeError):
                                write_chrome_trace(
                                    destination, recorder.events, {"bad": object()}
                                )
                        elif failure == "replace":
                            with patch.object(
                                Path, "replace", side_effect=OSError("rename failed")
                            ):
                                with self.assertRaisesRegex(OSError, "rename failed"):
                                    write_chrome_trace(destination, recorder.events)
                        else:
                            del recorder
                            gc.collect()
                    if failure != "abandon":
                        recorder.close()
                    self.assertEqual(destination.read_text(), "previous result")
                    self.assertEqual(list(path.glob("*.tmp")), [])


class PipelineRuntimeTest(unittest.TestCase):
    def test_output_hint_selects_preparation_and_preserves_runtime_results(self):
        runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        with self.assertRaisesRegex(ConfigurationError, "trace_output must be a path"):
            simulate_decode(runtime, 2, 8, trace_output=123)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            results = []
            for prepare in (False, True):
                executor, rows, compute, transfer = simulate_decode(
                    runtime,
                    2,
                    8,
                    steps=3,
                    fallback="analytical_only",
                    trace_output=path / "trace.json" if prepare else None,
                )
                try:
                    self.assertEqual(
                        isinstance(executor.trace, PipelinedTraceRecorder), prepare
                    )
                    write_chrome_trace(path / "trace.json", executor.trace.events)
                    results.append(
                        (
                            rows,
                            compute.stats_dict(),
                            transfer.describe(),
                            (path / "trace.json").read_bytes(),
                        )
                    )
                finally:
                    executor.trace.close()
            self.assertEqual(results[0], results[1])
            for mode in ("validate", "off"):
                executor, rows, compute, transfer = simulate_decode(
                    runtime,
                    2,
                    8,
                    steps=3,
                    fallback="analytical_only",
                    trace_output=path / "trace.json",
                    trace_config=AFTraceConfig(mode=mode),
                )
                try:
                    self.assertNotIsInstance(executor.trace, PipelinedTraceRecorder)
                    self.assertEqual(
                        (rows, compute.stats_dict(), transfer.describe()),
                        results[0][:3],
                    )
                finally:
                    executor.trace.close()
