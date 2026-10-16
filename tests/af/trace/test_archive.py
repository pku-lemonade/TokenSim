"""Trace retirement, bounded readback, interleaving and archive failure paths."""

from dataclasses import replace
import gc
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

from TokenSim.af.caching.estimates import freeze
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.trace import archive_store
from TokenSim.af.trace.archive import ArchivedTraceRecorder
from TokenSim.af.trace.archive_store import TraceArchiveStore
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.errors import SimulationStateError
from tests.af.support.observations import _valid_events
from tests.af.support.profiles import NativeFixture
from tests.af.trace.test_storage import iteration
from tests.support.hardware import make_model
from scripts import run_deepseek_mla as cli


class TraceArchiveTest(unittest.TestCase):
    def recorder(self, **kwargs):
        recorder = ArchivedTraceRecorder(**kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def add_iteration(self, recorder, identity):
        for event in _valid_events():
            recorder.record(replace(event, iteration_id=identity))
        recorder.validate_invariants(identity)
        recorder.complete_iteration(identity)

    def test_interleaved_completion_preserves_global_positions_and_active_patches(self):
        recorder = self.recorder()
        reference = AFTraceRecorder()
        for event in _valid_events():
            for identity in (7, 8, 9):
                current = replace(event, iteration_id=identity)
                recorder.record(current)
                reference.record(current)
        recorder.complete_iteration(8)
        self.assertEqual(recorder.events.retained_count, 8)
        for trace in (recorder, reference):
            trace.events[0] = replace(trace.events[0], source_id="patched")
        self.assertEqual(recorder.events_for(7)[0].source_id, "patched")
        self.assertIs(recorder.events_for(7)[0], recorder.events[0])
        recorder.complete_iteration(9)
        self.assertEqual(list(recorder.iteration_event_indices), [7, 8, 9])
        self.assertEqual(
            dict(recorder.iteration_event_indices), reference.iteration_event_indices
        )
        self.assertEqual(list(recorder.events), reference.events)
        recorder.complete_iteration(7)
        recorder.complete_iteration(7)
        self.assertEqual(recorder.events.retained_count, 0)
        self.assertEqual(recorder.iteration_event_indices.active, {})
        self.assertEqual(recorder.events, reference.events)
        self.assertEqual(recorder.events[-1], reference.events[-1])
        self.assertEqual(recorder.events[::-2], reference.events[::-2])
        self.assertEqual(recorder.events[1:8], reference.events[1:8])
        for identity in (7, 8, 9):
            recorder.validate_invariants(identity)
            self.assertEqual(
                recorder.events_for(identity), reference.events_for(identity)
            )
        for index in (12, -13):
            with self.assertRaises(IndexError):
                recorder.events[index]
        replacement = replace(reference.events[0], source_id="archived patch")
        recorder.events[-12] = replacement
        self.assertEqual(recorder.events_for(7)[0], replacement)
        recorder.archive._cache.clear()
        self.assertEqual(recorder.events[0], replacement)
        with self.assertRaisesRegex(ValueError, "iteration identity"):
            recorder.events[0] = replace(replacement, iteration_id=99)
        with self.assertRaisesRegex(SimulationStateError, "completed trace"):
            recorder.record(reference.events[0])

    def test_completed_replacement_is_used_by_validation_and_failed_patch_is_atomic(
        self,
    ):
        recorder = self.recorder()
        self.add_iteration(recorder, 7)
        original = recorder.events[0]
        with patch(
            "TokenSim.af.trace.archive_store._encode", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                recorder.events[0] = replace(original, source_id="failed patch")
        recorder.archive._cache.clear()
        self.assertEqual(recorder.events[0], original)
        recorder.events[2] = replace(recorder.events[2], predecessor_ids=("missing",))
        with self.assertRaisesRegex(SimulationStateError, "missing predecessors"):
            recorder.validate_invariants(7)

    def test_history_and_readback_cache_have_no_per_step_python_growth(self):
        recorder = self.recorder()
        # Small chunks exercise eviction and cross-chunk iteration readback.
        recorder.archive.chunk_size = 2
        for identity in range(300):
            self.add_iteration(recorder, identity)
            self.assertEqual(recorder.events.retained_count, 0)
            self.assertEqual(recorder.iteration_event_indices.active, {})
            if identity % 23 == 0:
                self.assertEqual(len(recorder.events_for(0)), 4)
                self.assertEqual(len(recorder.events_for(identity)), 4)
                self.assertLessEqual(recorder.archive.cached_event_count, 4)
        self.assertEqual(recorder.events.peak_retained, 4)
        self.assertEqual(len(recorder.events), 1200)
        self.assertEqual(len(recorder.iteration_event_indices), 300)
        self.assertEqual(sum(1 for _ in recorder.events), 1200)
        self.assertLessEqual(recorder.archive.cached_event_count, 4)

    def test_interleaved_readback_decodes_each_chunk_once_with_bounded_cache(self):
        recorder = self.recorder()
        recorder.archive.chunk_size = 2
        for event in _valid_events():
            for identity in range(5):
                recorder.record(replace(event, iteration_id=identity))
        for identity in range(5):
            recorder.complete_iteration(identity)
        self.assertEqual(recorder.archive.cache_chunks, 5)
        with patch.object(
            archive_store.pickle, "loads", wraps=archive_store.pickle.loads
        ) as loads:
            self.assertEqual(sum(1 for _ in recorder.events), 20)
            self.assertEqual(loads.call_count, 10)
        self.assertLessEqual(recorder.archive.cached_event_count, 10)
        recorder.clear()
        self.assertEqual(recorder.archive.cache_chunks, 2)

    def test_frozen_provider_records_round_trip_without_eager_json(self):
        recorder = self.recorder()
        current = replace(_valid_events()[0], details=freeze({"nested": [{"x": 1}]}))
        recorder.record(current)
        with patch.object(type(current), "as_dict", side_effect=AssertionError):
            recorder.complete_iteration(7)
            restored = recorder.events[0]
            self.assertEqual(restored, current)
        with self.assertRaises(TypeError):
            restored.details["x"] = 2
        self.assertEqual(restored.as_dict(), current.as_dict())

    def test_write_or_serialization_failure_retains_active_history_and_rolls_back(self):
        for failure in (OSError("disk full"), sqlite3.OperationalError("write failed")):
            recorder = self.recorder()
            recorder.archive.chunk_size = 2
            original = archive_store._encode
            for event in _valid_events():
                recorder.record(event)
            calls = 0

            def fail_later(events):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise failure
                return original(events)

            with patch(
                "TokenSim.af.trace.archive_store._encode", side_effect=fail_later
            ):
                with self.assertRaisesRegex(SimulationStateError, "could not archive"):
                    recorder.complete_iteration(7)
            self.assertEqual(recorder.archive.iteration_count, 0)
            self.assertEqual(list(recorder.archive.items()), [])
            self.assertEqual(recorder.events.retained_count, 4)
            self.assertEqual(recorder.iteration_event_indices[7], [0, 1, 2, 3])
            recorder.complete_iteration(7)
            self.assertEqual(list(recorder.events), list(_valid_events()))

    def test_sql_write_failure_is_atomic_and_can_be_retried(self):
        recorder = self.recorder()
        recorder.archive.chunk_size = 2
        for event in _valid_events():
            recorder.record(event)
        connection = recorder.archive._open()
        connection.execute(
            "CREATE TEMP TRIGGER fail_write BEFORE INSERT ON positions "
            "WHEN NEW.position >= 2 BEGIN SELECT RAISE(ABORT, 'write failed'); END"
        )
        with self.assertRaisesRegex(SimulationStateError, "write failed"):
            recorder.complete_iteration(7)
        self.assertEqual(recorder.events.retained_count, 4)
        for table in ("iterations", "chunks", "positions"):
            self.assertEqual(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0
            )
        connection.execute("DROP TRIGGER fail_write")
        recorder.complete_iteration(7)
        self.assertEqual(recorder.events, list(_valid_events()))

    def test_disabled_empty_clear_and_close_cleanup(self):
        recorder = self.recorder(enabled=False)
        self.add_iteration(recorder, 7)
        self.assertIsNone(recorder.archive.path)
        self.assertEqual(recorder.events, [])
        self.assertEqual(recorder.iteration_event_indices, {})
        recorder.enabled = True
        recorder.complete_iteration(999)
        self.assertEqual(recorder.events_for(999), ())
        self.add_iteration(recorder, 7)
        directory = recorder.archive.path.parent
        self.assertTrue(directory.exists())
        recorder.clear()
        self.assertFalse(directory.exists())
        self.assertEqual(recorder.events, [])
        self.assertEqual(recorder.iteration_event_indices, {})
        self.add_iteration(recorder, 7)
        self.assertEqual(recorder.iteration_event_indices[7], [0, 1, 2, 3])
        directory = recorder.archive.path.parent
        recorder.close()
        recorder.close()
        self.assertFalse(directory.exists())

    def test_unreferenced_store_cleans_up_its_database(self):
        recorder = ArchivedTraceRecorder()
        self.add_iteration(recorder, 7)
        directory = recorder.archive.path.parent
        reference = weakref.ref(recorder.archive)
        recorder = None
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(directory.exists())

    def test_repeated_full_exports_and_summaries_match_memory_history(self):
        recorder = self.recorder()
        reference = AFTraceRecorder()
        for identity in (7, 8, 9):
            for event in _valid_events():
                current = replace(event, iteration_id=identity)
                recorder.record(current)
                reference.record(current)
            recorder.complete_iteration(identity)
        for identity in (7, 8, 9):
            current = iteration(identity)
            self.assertEqual(recorder.summarize(current), reference.summarize(current))
            self.assertEqual(
                recorder.export_summary(recorder.summarize(current)),
                reference.export_summary(reference.summarize(current)),
            )
            self.assertEqual(
                recorder.summarize_timeline(current),
                reference.summarize_timeline(current),
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_chrome_trace(path, reference.events)
            expected = json.loads(path.read_text())
            for _ in range(2):
                write_chrome_trace(path, recorder.events)
                self.assertEqual(json.loads(path.read_text()), expected)

    def test_store_rejects_invalid_bounds(self):
        for name in ("chunk_size", "cache_chunks"):
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    TraceArchiveStore(**{name: value})


class ArchiveRuntimeTest(NativeFixture, unittest.TestCase):
    def test_dense_mha_and_gqa_moe_match_memory_results(self):
        for model in (make_model(num_layers=2), self.model):
            with self.subTest(model=model.model_id), TemporaryDirectory() as directory:
                runtime = build_model_runtime(
                    self.base.doc, hardware=self.base.hardware, model=model
                )
                results = []
                for recorder_type in (AFTraceRecorder, ArchivedTraceRecorder):
                    with patch(
                        "TokenSim.af.factory.ArchivedTraceRecorder", recorder_type
                    ):
                        executor, rows, compute, transfer = simulate_decode(
                            runtime, 2, 16, steps=4, fallback="analytical_only"
                        )
                    self.addCleanup(executor.trace.close)
                    output = Path(directory) / "trace.json"
                    write_chrome_trace(output, executor.trace.events)
                    results.append(
                        (
                            rows,
                            json.loads(output.read_text()),
                            compute.stats_dict(),
                            transfer.describe(),
                        )
                    )
                    if isinstance(executor.trace, ArchivedTraceRecorder):
                        self.assertEqual(executor.trace.events.retained_count, 0)
                        self.assertEqual(
                            executor.trace.iteration_event_indices.active, {}
                        )
                self.assertEqual(results[0], results[1])

    def test_failed_decode_releases_preceding_iterations_archive(self):
        runtime = build_model_runtime(
            self.base.doc, hardware=self.base.hardware, model=self.model
        )
        recorder = ArchivedTraceRecorder()
        self.addCleanup(recorder.close)
        run_iteration = ModelAFExecutor.run_iteration
        paths = []

        def fail_after_one(executor, current):
            if current.iteration_id == 1:
                paths.append(recorder.archive.path.parent)
                raise RuntimeError("simulation failed")
            return (yield from run_iteration(executor, current))

        with patch("TokenSim.af.factory.ArchivedTraceRecorder", return_value=recorder):
            with patch.object(ModelAFExecutor, "run_iteration", fail_after_one):
                with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                    simulate_decode(runtime, 2, 16, steps=2, fallback="analytical_only")
        self.assertTrue(paths)
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertEqual(recorder.events, [])

    def test_failed_cli_export_closes_the_archive(self):
        with TemporaryDirectory() as directory:
            args = cli.build_parser().parse_args(
                [
                    "--batches",
                    "1",
                    "--context",
                    "8",
                    "--fallback",
                    "analytical_only",
                    "--output",
                    directory,
                ]
            )
            simulate = cli.simulate
            paths = []

            def observe(*args, **kwargs):
                result = simulate(*args, **kwargs)
                paths.append(result[0].trace.archive.path.parent)
                return result

            with patch.object(cli, "simulate", side_effect=observe):
                with patch.object(
                    cli, "write_profile", side_effect=OSError("export failed")
                ):
                    with self.assertRaisesRegex(OSError, "export failed"):
                        cli.run(args)
            self.assertTrue(paths)
            self.assertTrue(all(not path.exists() for path in paths))
