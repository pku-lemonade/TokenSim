"""Production trace publication handles partial histories and edited records."""

from contextlib import closing
from dataclasses import replace
import importlib.util
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.af.trace.binary_archive import LEGACY_FORMAT
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.formats import (
    TRACE_SUFFIXES,
    convert_trace_to_json,
    make_trace_recorder,
    write_trace,
)
from TokenSim.errors import ConfigurationError
from tests.af.support.observations import _valid_events

FORMATS = ("json", "archive") + (
    ("msgpack",) if importlib.util.find_spec("msgpack") else ()
)


class TraceFormatsTest(unittest.TestCase):
    def assert_export_matches(self, root, events, trace_format, options, name):
        reference = write_chrome_trace(
            root / f"{name}-reference.json", iter(events), trace_config=options
        )
        trace = write_trace(
            root / f"{name}{TRACE_SUFFIXES[trace_format]}",
            events,
            trace_format=trace_format,
            trace_config=options,
        )
        self.assertEqual(trace.suffix, TRACE_SUFFIXES[trace_format])
        converted = convert_trace_to_json(trace, root / f"{name}-converted.json")
        self.assertEqual(converted.read_bytes(), reference.read_bytes())
        return trace

    def test_partial_and_edited_histories_keep_selected_format_and_event_order(self):
        for trace_format in FORMATS:
            with self.subTest(format=trace_format), TemporaryDirectory() as directory:
                root, options = Path(directory), AFTraceConfig()
                with closing(
                    make_trace_recorder(root, options, trace_format)
                ) as recorder:
                    for event in _valid_events():
                        for iteration in (7, 8):
                            recorder.record(
                                replace(
                                    event, iteration_id=iteration, details={"v": [1]}
                                )
                            )
                    recorder.complete_iteration(8)
                    self.assertGreater(recorder.events.retained_count, 0)
                    partial = self.assert_export_matches(
                        root, recorder.events, trace_format, options, "partial"
                    )
                    saved = partial.read_bytes()
                    recorder.complete_iteration(7)
                    recorder.events[0].details["v"].append(2)
                    self.assert_export_matches(
                        root, recorder.events, trace_format, options, "edited"
                    )
                    self.assertEqual(partial.read_bytes(), saved)
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_recorders_select_formats_independently_and_allow_changed_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorders = [
                make_trace_recorder(root, AFTraceConfig(), name) for name in FORMATS
            ]
            try:
                for recorder in recorders:
                    for event in _valid_events():
                        recorder.record(event)
                    recorder.complete_iteration(7)
                for recorder, name in zip(recorders, FORMATS):
                    # Publish the prepared path before ordinary readback invalidates it.
                    path = write_trace(
                        root / f"prepared-{name}.json",
                        recorder.events,
                        trace_format=name,
                    )
                    self.assertEqual(path.suffix, TRACE_SUFFIXES[name])
                    converted = convert_trace_to_json(path, root / f"{name}-full.json")
                    expected = write_chrome_trace(
                        root / "reference.json", _valid_events()
                    )
                    self.assertEqual(converted.read_bytes(), expected.read_bytes())
                    options = AFTraceConfig(iterations=[7], fields=["node_id"])
                    self.assert_export_matches(
                        root, recorder.events, name, options, f"selected-{name}"
                    )
                    # An explicit JSON export reads a binary recorder correctly.
                    explicit = write_trace(root / "explicit.json", recorder.events)
                    self.assertEqual(explicit.read_bytes(), expected.read_bytes())
                    direct = write_chrome_trace(root / "direct.json", recorder.events)
                    self.assertEqual(direct.read_bytes(), expected.read_bytes())
            finally:
                for recorder in recorders:
                    recorder.close()

    def test_single_pass_iterators_cross_archive_chunk_boundaries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                replace(_valid_events()[0], iteration_id=i % 3) for i in range(2051)
            ]
            options = AFTraceConfig(iterations=[1], fields=["node_id"])
            reference = write_chrome_trace(
                root / "reference.json", events, trace_config=options
            )
            for trace_format in FORMATS:
                trace = write_trace(
                    root / f"stream-{trace_format}.json",
                    iter(events),
                    trace_format=trace_format,
                    trace_config=options,
                )
                converted = convert_trace_to_json(trace, root / f"{trace_format}.json")
                self.assertEqual(converted.read_bytes(), reference.read_bytes())

    def test_legacy_indexed_snapshot_is_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = write_trace(
                root / "legacy.json", _valid_events(), trace_format="archive"
            )
            with closing(sqlite3.connect(trace)) as connection, connection:
                connection.execute("UPDATE trace_export SET format=?", (LEGACY_FORMAT,))
            converted = convert_trace_to_json(trace, root / "converted.json")
            expected = write_chrome_trace(root / "reference.json", _valid_events())
            self.assertEqual(converted.read_bytes(), expected.read_bytes())

    def test_failed_publication_preserves_destination_and_cleans_temporary_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for trace_format in FORMATS:
                source = write_trace(
                    root / f"source-{trace_format}.json",
                    _valid_events(),
                    trace_format=trace_format,
                )
                destination = root / f"output-{trace_format}.json"
                destination.write_bytes(b"previous complete trace")
                with patch.object(
                    Path, "replace", side_effect=OSError("publication failed")
                ):
                    with self.assertRaisesRegex(OSError, "publication failed"):
                        convert_trace_to_json(source, destination)
                self.assertEqual(destination.read_bytes(), b"previous complete trace")
                self.assertEqual(list(root.glob("*.tmp")), [])
                published = root / f"failed{TRACE_SUFFIXES[trace_format]}"
                published.write_bytes(b"previous complete trace")
                with patch.object(
                    Path, "replace", side_effect=OSError("publication failed")
                ):
                    with self.assertRaisesRegex(OSError, "publication failed"):
                        write_trace(
                            published, _valid_events(), trace_format=trace_format
                        )
                self.assertEqual(published.read_bytes(), b"previous complete trace")
                self.assertEqual(list(root.glob("*.tmp")), [])

    def test_json_copy_same_path_and_invalid_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_chrome_trace(root / "source.json", _valid_events())
            before = source.stat().st_mtime_ns
            self.assertEqual(convert_trace_to_json(source, source), source)
            self.assertEqual(source.stat().st_mtime_ns, before)
            with self.assertRaises(FileNotFoundError):
                convert_trace_to_json(root / "missing.json", root / "missing.json")
            for value in (None, root / "unknown.bin", []):
                with self.subTest(value=value), self.assertRaises(ConfigurationError):
                    convert_trace_to_json(value, root / "output.json")
            for trace_format in (
                "archive",
                *(["msgpack"] if "msgpack" in FORMATS else []),
            ):
                source = write_trace(
                    root / "binary.json", _valid_events(), trace_format=trace_format
                )
                before = source.read_bytes()
                with self.assertRaisesRegex(ValueError, "must differ"):
                    convert_trace_to_json(source, source)
                self.assertEqual(source.read_bytes(), before)
