"""Data-only chunks preserve field evolution without loading project classes."""

from contextlib import closing
from dataclasses import dataclass, replace
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

from scripts import af_probe_datapickle as probe
from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.providers.queries import AFEndpoint
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.pipeline import PipelinedTraceRecorder
from tests.af.support.observations import _valid_events


@dataclass(frozen=True)
class FutureDetail:
    new_field: list


class DataOnlyTraceTest(unittest.TestCase):
    def test_dynamic_fields_large_integers_filters_and_empty_roundtrips(self):
        endpoint = AFEndpoint("a", 0, "worker:0", "test")
        for options in (
            AFTraceConfig(),
            AFTraceConfig(detail="timing"),
            AFTraceConfig(iterations=[]),
            AFTraceConfig(fields=["details"]),
        ):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                root = Path(directory)
                events = [
                    replace(
                        event,
                        source_endpoint=endpoint,
                        details=FrozenDict(
                            {
                                "custom": FutureDetail(
                                    [endpoint, '解码\n"trace"', 2**100]
                                )
                            }
                        ),
                    )
                    for event in _valid_events()
                ]
                metadata = {"nested": [True, None, -0.0]}
                reference = write_chrome_trace(
                    root / "reference.json", events, metadata, trace_config=options
                )
                with (
                    probe.installed(),
                    closing(
                        PipelinedTraceRecorder(root, trace_config=options)
                    ) as recorder,
                ):
                    for event in events:
                        recorder.record(event)
                    recorder.complete_iteration(7)
                    binary = write_chrome_trace(
                        root / "trace.json",
                        recorder.events,
                        metadata,
                        trace_config=options,
                    )
                destination = root / "converted.json"
                probe.convert(binary, destination)
                self.assertEqual(destination.read_bytes(), reference.read_bytes())
                self.assertNotIn(b"TokenSim.af", binary.read_bytes())
                self.assertNotIn(b"FutureDetail", binary.read_bytes())
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_chunk_boundary_and_snapshot_mutation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            details = FutureDetail([1])
            event = replace(_valid_events()[0], details={"custom": details})
            with closing(probe.DataWriter(root, AFTraceConfig())) as writer:
                writer.append([event] * 1025)
                details.new_field.append(2)
                writer.append([event])
                binary = writer.finish(root / "trace.json", {})
            destination = root / "converted.json"
            self.assertEqual(probe.convert(binary, destination), 1026)
            events = json.loads(destination.read_text())["traceEvents"]
            self.assertEqual(events[0]["args"]["details"]["custom"]["new_field"], [1])
            self.assertEqual(
                events[1024]["args"]["details"]["custom"]["new_field"], [1]
            )
            self.assertEqual(
                events[1025]["args"]["details"]["custom"]["new_field"], [1, 2]
            )

    def test_corrupt_oversized_and_unapproved_frames_preserve_destination(self):
        footer = json.dumps(dict(metadata={}, event_count=0)).encode()
        complete = probe.MAGIC + probe.FRAME.pack(2, len(footer)) + footer
        cases = (
            b"",
            probe.MAGIC,
            complete[:-1],
            complete + b"\x01",
            complete + complete,
            probe.MAGIC + probe.FRAME.pack(1, probe.MAX_FRAME + 1),
            probe.MAGIC + probe.FRAME.pack(9, 0),
            probe.MAGIC
            + probe.FRAME.pack(1, len(pickle.dumps(eval)))
            + pickle.dumps(eval),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "trace.pickle", root / "trace.json"
            for content in cases:
                with self.subTest(content=content):
                    source.write_bytes(content)
                    destination.write_text("previous complete trace")
                    with self.assertRaises((ValueError, pickle.UnpicklingError)):
                        probe.convert(source, destination)
                    self.assertEqual(destination.read_text(), "previous complete trace")
                    self.assertEqual(list(root.glob("*.tmp")), [])

    def test_unsupported_objects_fail_before_publication(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            event = replace(_valid_events()[0], details={"unsupported": object()})
            with closing(probe.DataWriter(root, AFTraceConfig())) as writer:
                with self.assertRaisesRegex(TypeError, "unsupported trace value"):
                    writer.append([event])
            self.assertEqual(list(root.glob("*.tmp")), [])
