"""Range indexing preserves interleaving, chunk bounds and atomic rollback."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from TokenSim.af.trace.archive import ArchivedTraceRecorder
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.formats import (
    convert_trace_to_json,
    make_trace_recorder,
    write_trace,
)
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.errors import SimulationStateError
from scripts.af_probe_spanindex import installed
from tests.af.support.observations import _valid_events


class SpanIndexProbeTest(unittest.TestCase):
    def test_interleaved_ranges_and_random_access_cross_chunk_boundaries(self):
        with installed(), closing(ArchivedTraceRecorder()) as recorder:
            recorder.archive.chunk_size = 3
            expected = []
            for offset, event in enumerate(_valid_events()):
                for identity in (7, 7, 8) if offset % 2 else (8, 7, 7):
                    current = replace(event, iteration_id=identity)
                    recorder.record(current)
                    expected.append(current)
            for identity in (8, 7):
                recorder.complete_iteration(identity)
            self.assertEqual(list(recorder.events), expected)
            for position, current in enumerate(expected):
                self.assertEqual(recorder.events[position], current)
                self.assertEqual(
                    list(recorder.archive.items(start=position, end=position + 1)),
                    [(position, current)],
                )
            for identity in (7, 8):
                self.assertEqual(
                    recorder.archive.positions_for(identity),
                    [
                        i
                        for i, event in enumerate(expected)
                        if event.iteration_id == identity
                    ],
                )
            changed = replace(expected[4], source_id="changed")
            recorder.events[4] = changed
            recorder.archive._cache.clear()
            self.assertEqual(recorder.events[4], changed)
            with self.assertRaises(IndexError):
                recorder.archive.event(12)

    def test_sql_failure_rolls_back_chunks_ranges_and_iteration_membership(self):
        with installed(), closing(ArchivedTraceRecorder()) as recorder:
            recorder.archive.chunk_size = 2
            for event in _valid_events():
                recorder.record(event)
            connection = recorder.archive._open()
            connection.execute(
                "CREATE TEMP TRIGGER fail_write BEFORE INSERT ON spans "
                "WHEN NEW.start >= 2 BEGIN SELECT RAISE(ABORT, 'write failed'); END"
            )
            with self.assertRaisesRegex(SimulationStateError, "write failed"):
                recorder.complete_iteration(7)
            for table in ("iterations", "chunks", "spans"):
                self.assertEqual(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0
                )
            self.assertEqual(recorder.events.retained_count, 4)
            connection.execute("DROP TRIGGER fail_write")
            recorder.complete_iteration(7)
            self.assertEqual(list(recorder.events), list(_valid_events()))

    def test_saved_snapshot_view_converts_through_production_reader(self):
        with TemporaryDirectory() as directory, installed():
            root = Path(directory)
            with closing(
                make_trace_recorder(root, AFTraceConfig(), "archive")
            ) as recorder:
                for event in _valid_events():
                    recorder.record(event)
                recorder.complete_iteration(7)
                saved = write_trace(
                    root / "trace.json", recorder.events, trace_format="archive"
                )
                with closing(sqlite3.connect(saved)) as connection:
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM spans").fetchone()[0],
                        1,
                    )
                converted = convert_trace_to_json(saved, root / "converted.json")
                expected = write_chrome_trace(root / "expected.json", _valid_events())
                self.assertEqual(converted.read_bytes(), expected.read_bytes())
