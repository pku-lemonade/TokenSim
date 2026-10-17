"""Binary trace probes preserve snapshots, order and complete JSON values."""

from contextlib import closing
from dataclasses import dataclass, replace
import importlib.util
import json
from pathlib import Path
import pickle
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import af_probe_archiveonly as archive
from scripts import af_probe_sharedpack as shared
from TokenSim.af.providers.queries import AFEndpoint
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import _encode_event, _event_encoder, write_chrome_trace
from tests.af.support.observations import _valid_events


class ArchiveSnapshotTest(unittest.TestCase):
    def test_interleaved_order_filters_and_independent_snapshot(self):
        for options in (
            AFTraceConfig(),
            AFTraceConfig(iterations=[8]),
            AFTraceConfig(iterations=[]),
        ):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                root = Path(directory)
                events = [
                    replace(event, iteration_id=iteration, details={"snapshot": [1]})
                    for event in _valid_events()
                    for iteration in (7, 8)
                ]
                metadata = {"description": '解码\n"trace"'}
                reference = write_chrome_trace(
                    root / "reference.json", events, metadata, trace_config=options
                )
                with closing(archive.ArchiveOnlyRecorder(root)) as recorder:
                    for event in events:
                        recorder.record(event)
                    recorder.complete_iteration(8)
                    recorder.complete_iteration(7)
                    self.assertEqual(recorder.events.retained_count, 0)
                    snapshot = write_chrome_trace(
                        root / "trace.json",
                        recorder.events,
                        metadata,
                        trace_config=options,
                    )
                    self.assertEqual(snapshot.suffix, ".sqlite3")
                    recorder.events[0] = replace(events[0], source_id="late update")
                    self.assertEqual(recorder.events[0].source_id, "late update")
                converted = root / "converted.json"
                self.assertEqual(archive.convert(snapshot, converted), len(events))
                self.assertEqual(converted.read_bytes(), reference.read_bytes())
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_partial_trace_falls_back_and_failed_publication_preserves_destination(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with closing(archive.ArchiveOnlyRecorder(root)) as recorder:
                recorder.record(_valid_events()[0])
                partial = write_chrome_trace(root / "partial.json", recorder.events)
                self.assertEqual(partial.suffix, ".json")
                self.assertEqual(len(json.loads(partial.read_text())["traceEvents"]), 1)
                recorder.complete_iteration(7)
                destination = root / "trace.sqlite3"
                destination.write_text("previous complete trace")
                with patch.object(
                    Path, "replace", side_effect=OSError("publication failed")
                ):
                    with self.assertRaisesRegex(OSError, "publication failed"):
                        write_chrome_trace(root / "trace.json", recorder.events)
                self.assertEqual(destination.read_text(), "previous complete trace")
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_empty_count_mismatch_and_unapproved_pickle_globals(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with closing(archive.ArchiveOnlyRecorder(root)) as recorder:
                empty = write_chrome_trace(root / "empty.json", recorder.events)
                self.assertEqual(
                    archive.convert(empty, root / "empty-converted.json"), 0
                )
                recorder.record(_valid_events()[0])
                recorder.complete_iteration(7)
                snapshot = write_chrome_trace(root / "trace.json", recorder.events)
            destination = root / "converted.json"
            destination.write_text("previous complete trace")
            connection = sqlite3.connect(snapshot)
            try:
                with connection:
                    connection.execute("UPDATE trace_export SET event_count=2")
                with self.assertRaisesRegex(ValueError, "event count"):
                    archive.convert(snapshot, destination)
                with connection:
                    connection.execute("UPDATE trace_export SET event_count=1")
                    connection.execute(
                        "UPDATE chunks SET payload=?", (pickle.dumps(eval),)
                    )
                with self.assertRaisesRegex(
                    pickle.UnpicklingError, "unsupported trace record"
                ):
                    archive.convert(snapshot, destination)
                self.assertEqual(destination.read_text(), "previous complete trace")
                self.assertEqual(list(root.glob("*.tmp")), [])
            finally:
                connection.close()


@dataclass(frozen=True)
class Detail:
    endpoint: AFEndpoint
    items: list


@unittest.skipUnless(
    importlib.util.find_spec("msgpack"), "optional msgpack unavailable"
)
class SharedRecordsTest(unittest.TestCase):
    def test_references_and_mutation_between_prefixes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = AFEndpoint("a", 0, "worker:0", "test")
            detail = Detail(endpoint, [1])
            event = replace(
                _valid_events()[0],
                source_endpoint=endpoint,
                details={"one": detail, "two": detail},
            )
            expected = []
            writer = shared.SharedWriter(root, AFTraceConfig())
            self.addCleanup(writer.close)
            for index in range(2):
                detail.items.append(index)
                expected.extend(
                    _encode_event(event, AFTraceConfig(), _event_encoder())
                    for _ in range(3)
                )
                writer.append([event] * 3)
            binary = writer.finish(root / "trace.json", {})
            converted = root / "converted.json"
            self.assertEqual(shared.convert(binary, converted), 6)
            reference = (
                '{"displayTimeUnit":"us","traceEvents":['
                + ",".join(expected)
                + '],"metadata":{}}'
            )
            self.assertEqual(converted.read_text(), reference)
            self.assertGreater(writer.encoder.hits, 0)
            self.assertLessEqual(len(writer.encoder.memo), 8192)

    def test_invalid_references_and_definition_limits(self):
        import msgpack

        for tag, data in (
            (2, b""),
            (2, (3).to_bytes(4, "big")),
            (9, b""),
            (1, msgpack.packb((8192, {}))),
        ):
            with self.subTest(tag=tag, data=data), self.assertRaises(ValueError):
                shared.SharedDecoder().extension(tag, data)
        decoder = shared.SharedDecoder()
        data = msgpack.packb((0, {"value": 1}))
        self.assertEqual(decoder.extension(1, data), {"value": 1})
        with self.assertRaises(ValueError):
            decoder.extension(1, data)
        self.assertEqual(decoder.extension(3, data), {"value": 1})
