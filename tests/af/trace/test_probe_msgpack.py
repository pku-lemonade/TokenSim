"""The diagnostic binary stream preserves event values and fails atomically."""

from contextlib import closing
from dataclasses import dataclass, replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import af_probe_msgpack as probe
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.pipeline import PipelinedTraceRecorder
from tests.af.support.observations import _valid_events


@dataclass(frozen=True)
class Detail:
    name: str
    values: tuple


@unittest.skipUnless(
    importlib.util.find_spec("msgpack"), "optional msgpack unavailable"
)
class MessagePackProbeTest(unittest.TestCase):
    def test_complete_selected_and_empty_roundtrips(self):
        for options in (
            AFTraceConfig(),
            AFTraceConfig(detail="timing"),
            AFTraceConfig(iterations=[7], event_types=["compute"]),
            AFTraceConfig(iterations=[]),
            AFTraceConfig(fields=["details", "start_at"]),
        ):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                root = Path(directory)
                events = [
                    replace(
                        event,
                        details={
                            "record": Detail('解码\n"trace"', (1, 0.25, -0.0)),
                            "optional": None,
                            "mapping": {3: "integer key"},
                        },
                    )
                    for event in _valid_events()
                ]
                metadata = {"description": "完整 trace", "nested": [1, True, None]}
                reference = write_chrome_trace(
                    root / "reference.json", events, metadata, trace_config=options
                )
                with (
                    probe.installed(root),
                    closing(
                        PipelinedTraceRecorder(root, trace_config=options)
                    ) as recorder,
                ):
                    for event in events:
                        recorder.record(event)
                    recorder.complete_iteration(events[0].iteration_id)
                    binary = write_chrome_trace(
                        root / "trace.json",
                        recorder.events,
                        metadata,
                        trace_config=options,
                    )
                self.assertEqual(binary.suffix, ".msgpack")
                converted = root / "converted.json"
                count = probe.convert(binary, converted)
                self.assertEqual(
                    count, len(json.loads(reference.read_text())["traceEvents"])
                )
                self.assertEqual(converted.read_bytes(), reference.read_bytes())
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_rejects_corrupt_incomplete_and_trailing_records(self):
        import msgpack

        header = msgpack.packb(probe.HEADER, use_bin_type=True)
        footer = msgpack.packb({"metadata": {}, "event_count": 0}, use_bin_type=True)
        wrong_count = msgpack.packb({"metadata": {}, "event_count": 1})
        missing_metadata = msgpack.packb({"event_count": 0})
        cases = (
            b"",
            msgpack.packb({"version": 2}) + footer,
            header,
            header + wrong_count,
            header + missing_metadata,
            header + footer[:-1],
            header + footer + footer,
            header + footer + b"\xd9",
            header + msgpack.packb(["unexpected"]),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "trace.msgpack", root / "trace.json"
            for content in cases:
                with self.subTest(content=content):
                    source.write_bytes(content)
                    destination.write_text("previous complete trace")
                    with self.assertRaises(ValueError):
                        probe.convert(source, destination)
                    self.assertEqual(destination.read_text(), "previous complete trace")
                    self.assertEqual(list(root.glob("*.tmp")), [])
            source.write_bytes(header + footer)
            with self.assertRaisesRegex(ValueError, "must differ"):
                probe.convert(source, source)
            self.assertEqual(source.read_bytes(), header + footer)

    def test_failed_publication_and_context_restore(self):
        from TokenSim.af.trace import pipeline

        original = pipeline._PrefixWriter
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "trace.msgpack"
            destination.write_text("previous complete trace")
            with self.assertRaisesRegex(OSError, "publication failed"):
                with (
                    probe.installed(root),
                    closing(PipelinedTraceRecorder(root)) as recorder,
                ):
                    source = _valid_events()[0]
                    recorder.record(source)
                    recorder.complete_iteration(source.iteration_id)
                    with patch.object(
                        Path, "replace", side_effect=OSError("publication failed")
                    ):
                        write_chrome_trace(root / "trace.json", recorder.events)
            self.assertIs(pipeline._PrefixWriter, original)
            self.assertEqual(destination.read_text(), "previous complete trace")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_prototype_reports_integer_range_limit(self):
        with self.assertRaises(TypeError):
            probe._BinaryEncoder().encode({"outside_msgpack_range": 2**100})
