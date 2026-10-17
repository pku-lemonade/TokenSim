"""Prefix export keeps interleaving, snapshots, late patches and cleanup."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.af_probe_pipeline import PipelinedTraceRecorder, PrefixWriter
from TokenSim.af.trace.export import write_chrome_trace
from tests.af.support.observations import _valid_events


class PipelineExportTest(unittest.TestCase):
    def test_late_patch_before_publication_and_writer_failure_cleanup(self):
        for threaded in (False, True):
            with self.subTest(threaded=threaded), TemporaryDirectory() as directory:
                path = Path(directory)
                recorder = PipelinedTraceRecorder(path, threaded)
                try:
                    source = _valid_events()[0]
                    recorder.record(source)
                    recorder.complete_iteration(source.iteration_id)
                    recorder.events[0] = replace(source, source_id="late patch")
                    expected = write_chrome_trace(
                        path / "reference.json", recorder.events
                    )
                    observed = recorder.finish_profile(path / "observed.json", {})
                    self.assertEqual(expected.read_bytes(), observed.read_bytes())
                finally:
                    recorder.close()
                writer = PrefixWriter(path, threaded)
                with patch.object(
                    writer.stream, "write", side_effect=OSError("disk full")
                ):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        writer.write(b"more data")
                        writer.drain()
                try:
                    writer.close()
                except OSError:
                    pass
                self.assertEqual(list(path.glob("*.tmp")), [])

    def test_interleaving_snapshots_late_replacements_and_repeated_export(self):
        for threaded in (False, True):
            with self.subTest(threaded=threaded), TemporaryDirectory() as directory:
                path = Path(directory)
                recorder = PipelinedTraceRecorder(path, threaded)
                try:
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
                    self.assertEqual(recorder.events.exported_count, 0)
                    recorder.complete_iteration(7)
                    mutable["snapshot"].append(2)
                    for suffix in ("initial", "repeat", "replacement"):
                        if suffix == "replacement":
                            recorder.events[0] = replace(
                                recorder.events[0], source_id="patched"
                            )
                        expected = write_chrome_trace(
                            path / "expected.json", recorder.events, {"test": suffix}
                        )
                        observed = recorder.finish_profile(
                            path / "observed.json", {"test": suffix}
                        )
                        self.assertEqual(expected.read_bytes(), observed.read_bytes())
                    recorder.clear()
                    recorder.record(source)
                    recorder.complete_iteration(source.iteration_id)
                    recorder.finish_profile(path / "after-clear.json", {})
                finally:
                    recorder.close()
                self.assertEqual(list(path.glob("*.tmp")), [])

    def test_partial_iteration_falls_back_and_abandon_keeps_destination(self):
        for threaded in (False, True):
            with self.subTest(threaded=threaded), TemporaryDirectory() as directory:
                path = Path(directory)
                destination = path / "trace.json"
                destination.write_text("previous result")
                recorder = PipelinedTraceRecorder(path, threaded)
                try:
                    recorder.record(_valid_events()[0])
                    expected = write_chrome_trace(
                        path / "reference.json", recorder.events
                    )
                    recorder.finish_profile(destination, {})
                    self.assertEqual(expected.read_bytes(), destination.read_bytes())
                finally:
                    recorder.close()
                recorder = PipelinedTraceRecorder(path, threaded)
                recorder.record(_valid_events()[0])
                recorder.complete_iteration(_valid_events()[0].iteration_id)
                before = destination.read_bytes()
                recorder.close()
                self.assertEqual(before, destination.read_bytes())
                self.assertEqual(list(path.glob("*.tmp")), [])
