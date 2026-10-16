"""Chrome trace values, final dependency patches and export failure handling."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.metrics import AFTraceEvent
from scripts.deepseek_v3_runtime import configuration, simulate, write_profile
from tests.af.support.observations import _valid_events


def reference_payload(events, metadata):
    rows = []
    for event in events:
        row = event.as_dict()
        rows.append(
            dict(
                name=row["stage"] or row["operation"] or row["event_type"],
                cat=row["event_type"],
                ph="X",
                pid=1,
                tid=row["resource_id"],
                ts=row["start_at"] * 1e6,
                dur=row["service_duration"] * 1e6,
                args=row,
            )
        )
    return json.loads(
        json.dumps(
            dict(
                displayTimeUnit="us",
                traceEvents=rows,
                metadata=metadata,
            )
        )
    )


class TraceExportTest(unittest.TestCase):
    def test_runtime_events_match_reference_after_dependency_patches(self):
        executor, _, _, _ = simulate(
            configuration(a_tp=2, f_tp=2, af_batch_size=1),
            2,
            8,
            steps=2,
            fallback="analytical_only",
        )
        events = executor.trace.events
        self.assertTrue(any(e.tensor_slices for e in events))
        self.assertTrue(any(e.routing_edges for e in events))
        self.assertTrue(any(e.predecessor_ids for e in events))
        metadata = {"description": '解码\n"trace"', "counts": [2, 3]}
        expected = reference_payload(events, metadata)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "profile.json"
            with patch.object(AFTraceEvent, "as_dict", side_effect=AssertionError):
                self.assertEqual(
                    write_chrome_trace(output, iter(events), metadata), output
                )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)
        self.assertEqual(reference_payload(events, metadata), expected)

    def test_empty_trace_and_existing_wrapper_metadata_contract(self):
        default = {"timing": "simulated operator-table service time"}
        executor = SimpleNamespace(trace=SimpleNamespace(events=[]))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            write_chrome_trace(str(output), iter(()))
            self.assertEqual(json.loads(output.read_text()), reference_payload([], {}))
            for metadata in (None, {}, {"model": "dense"}):
                with self.subTest(metadata=metadata):
                    self.assertEqual(write_profile(output, executor, metadata), output)
                    self.assertEqual(
                        json.loads(output.read_text()),
                        reference_payload([], metadata or default),
                    )

    def test_failed_export_preserves_destination_and_removes_temporary_file(self):
        events = list(_valid_events())
        events.append(replace(events[0], details={"unsupported": object()}))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            output.write_text("existing result")
            for destination_exists in (True, False):
                with self.subTest(destination_exists=destination_exists):
                    if not destination_exists:
                        output.unlink()
                    with self.assertRaisesRegex(TypeError, "not JSON serializable"):
                        write_chrome_trace(output, iter(events))
                    self.assertEqual(output.exists(), destination_exists)
                    if destination_exists:
                        self.assertEqual(output.read_text(), "existing result")
                    self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_iterator_failure_also_cleans_up(self):
        def events():
            yield _valid_events()[0]
            raise RuntimeError("trace source failed")

        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            with self.assertRaisesRegex(RuntimeError, "trace source failed"):
                write_chrome_trace(output, events())
            self.assertEqual(list(Path(directory).iterdir()), [])
