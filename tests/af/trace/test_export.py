"""Chrome trace values, final dependency patches and export failure handling."""

from dataclasses import dataclass, field, replace
import gc
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

    def test_record_fields_keep_schema_order_and_attribute_semantics(self):
        @dataclass
        class Ordinary:
            value: int
            default: int = field(default=7, init=False)

        @dataclass(slots=True)
        class Slotted:
            value: int

        @dataclass
        class CustomAccess:
            value: int

            def __getattribute__(self, name):
                value = object.__getattribute__(self, name)
                return value * 2 if name == "value" else value

        class Doubled:
            def __set_name__(self, owner, name):
                self.name = name

            def __get__(self, instance, owner=None):
                return self if instance is None else 2 * vars(instance)[self.name]

            def __set__(self, instance, value):
                vars(instance)[self.name] = value

        @dataclass
        class DescriptorAccess:
            value: int = Doubled()

        ordinary = Ordinary(3)
        ordinary.extra = "not a field"
        ordinary.default = 11
        reordered = Ordinary(5)
        del reordered.value
        reordered.value = 9
        values = (ordinary, reordered, Slotted(4), CustomAccess(6), DescriptorAccess(5))
        events = [replace(_valid_events()[0], details={"records": values})]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            write_chrome_trace(output, events)
            self.assertEqual(
                json.loads(output.read_text()), reference_payload(events, {})
            )
            self.assertNotIn('"extra"', output.read_text())

    def test_export_preserves_collection_policy_on_success_and_failure(self):
        previous, enabled = gc.get_threshold(), gc.isenabled()
        self.addCleanup(gc.set_threshold, *previous)
        self.addCleanup(gc.enable if enabled else gc.disable)
        policies = (
            (True, (700, 8, 9), (250_000, 8, 9)),
            (True, (1_000_000, 8, 9), (1_000_000, 8, 9)),
            (False, (700, 8, 9), (700, 8, 9)),
            (True, (0, 8, 9), (0, 8, 9)),
        )
        with TemporaryDirectory() as directory:
            for enabled, policy, expected in policies:
                for fail in (False, True):
                    with self.subTest(enabled=enabled, policy=policy, fail=fail):
                        (gc.enable if enabled else gc.disable)()
                        gc.set_threshold(*policy)

                        def events():
                            self.assertEqual(gc.get_threshold(), expected)
                            self.assertEqual(gc.isenabled(), enabled)
                            yield _valid_events()[0]
                            if fail:
                                raise RuntimeError("export failed")

                        path = Path(directory) / "trace.json"
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "export failed"):
                                write_chrome_trace(path, events())
                        else:
                            write_chrome_trace(path, events())
                        self.assertEqual(gc.get_threshold(), policy)
                        self.assertEqual(gc.isenabled(), enabled)
