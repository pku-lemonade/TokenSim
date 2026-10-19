"""Trace storage, accounting and export contracts, independent of wall-clock timing."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.af.planning.batching import AFIteration
from TokenSim.af.planning.nodes import RankLocalWork
from TokenSim.af.providers.queries import AFTensorSlice
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.errors import SimulationStateError
from TokenSim.reporting import get_af_stats
from scripts.deepseek_v3_runtime import configuration, simulate
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    _build_engine,
    _run_request,
    _typed_cluster,
)
from tests.af.support.observations import (
    AccessList,
    _event,
    _valid_events,
    lookup_visits,
)


def iteration(identity=7, end=4):
    return AFIteration(identity, (), (), 0, end, {0: end}, {0: 0})


def reference_summary(recorder, current):
    """Original full-scan / per-resource sum / eager serialization implementation."""
    events = [
        event for event in recorder.events if event.iteration_id == current.iteration_id
    ]
    names = {r for e in events for r in (e.resource_ids or (e.resource_id,))}
    makespan = current.makespan or 0
    return dict(
        iteration_id=current.iteration_id,
        phase=current.phase,
        start_at=current.start_at,
        end_at=current.end_at,
        makespan=current.makespan,
        block_completed_at=dict(current.block_completed_at),
        barrier_wait=dict(current.barrier_wait),
        resource_utilization={
            r: (
                sum(
                    e.service_duration
                    for e in events
                    if r in (e.resource_ids or (e.resource_id,))
                )
                / (makespan * recorder.resource_capacities.get(r, 1))
                if makespan > 0
                else 0.0
            )
            for r in sorted(names)
        },
        events=[event.as_dict() for event in events],
    )


class TraceStorageTest(unittest.TestCase):
    def test_deferred_work_records_preserve_exported_json(self):
        work = RankLocalWork(
            2,
            tensor_shape=(("hidden_size", 16),),
            expert_workload=((3, 2),),
            tensor_slices=(AFTensorSlice("x", (0, 16), "partial", payload_bytes=32),),
        )
        event = replace(_valid_events()[0], local_work=work.trace_mapping())
        self.assertIs(event.local_work["tensor_slices"], work.tensor_slices)
        self.assertEqual(event.as_dict()["local_work"], work.as_dict())
        json.dumps(event.as_dict())

    def test_interleaved_iterations_read_only_their_positions(self):
        recorder = AFTraceRecorder()
        for event in _valid_events():
            recorder.record(event)
            recorder.record(replace(event, iteration_id=8))
        expected = tuple(e for e in recorder.events if e.iteration_id == 7)
        recorder.events = AccessList(recorder.events)
        self.assertEqual(
            recorder.iteration_event_indices, {7: [0, 2, 4, 6], 8: [1, 3, 5, 7]}
        )
        self.assertEqual(recorder.events_for(7), expected)
        self.assertEqual(recorder.events.visits, 4)
        recorder.validate_invariants(7)
        recorder.validate_invariants(8)
        self.assertEqual(recorder.events.visits, 12)

    def test_replacement_is_seen_by_lookup_validation_and_export(self):
        recorder = AFTraceRecorder()
        for event in _valid_events():
            recorder.record(event)
        summary = recorder.summarize(iteration())
        replacement = replace(recorder.events[2], predecessor_ids=("missing",))
        recorder.events[2] = replacement
        self.assertIs(recorder.events_for(7)[2], replacement)
        self.assertEqual(
            recorder.export_summary(summary)["events"][2]["predecessor_ids"],
            ("missing",),
        )
        with self.assertRaisesRegex(SimulationStateError, "missing predecessors"):
            recorder.validate_invariants(7)

    def test_model_boundary_patches_are_not_stale_in_index(self):
        originals = []
        record = AFTraceRecorder.record

        def observe(recorder, event):
            originals.append(event)
            record(recorder, event)

        with patch.object(AFTraceRecorder, "record", observe):
            executor, _, _, _ = simulate(
                configuration(), 1, 8, steps=2, fallback="analytical_only"
            )
        changed = [
            (old, new)
            for old, new in zip(originals, executor.trace.events)
            if old is not new
        ]
        self.assertTrue(changed)
        for old, new in changed:
            self.assertFalse(old.predecessor_ids)
            self.assertTrue(new.predecessor_ids)
            self.assertIs(
                next(
                    e
                    for e in executor.trace.events_for(new.iteration_id)
                    if e.node_id == new.node_id
                ),
                new,
            )
        for identity in (0, 1):
            executor.trace.validate_invariants(identity)
            summary = executor.trace.summarize(iteration(identity))
            self.assertEqual(
                executor.trace.export_summary(summary),
                reference_summary(executor.trace, iteration(identity)),
            )

    def test_single_pass_resource_accounting_matches_original(self):
        recorder = AFTraceRecorder(resource_capacities={"z": 2, "a": 3})
        for i, duration in enumerate((1e16, 1.0, 1.0, 0.25, 0.1, 0.2)):
            recorder.record(
                replace(
                    _event(str(i), "compute_a", 0, 0, duration),
                    resource_id="z",
                    resource_ids=("z", "a", "z"),
                )
            )
        recorder.record(
            replace(_event("fallback", "compute_a", 0, 0, 2), resource_ids=())
        )
        for end in (0, 1e17):
            current = iteration(end=end)
            expected = reference_summary(recorder, current)
            with patch.object(
                AFTraceEvent,
                "as_dict",
                side_effect=AssertionError("early serialization"),
            ):
                summary = recorder.summarize(current)
            self.assertEqual(
                list(summary["resource_utilization"]), ["a", "worker:0", "z"]
            )
            self.assertEqual(recorder.export_summary(summary), expected)

    def test_summary_traverses_iteration_events_once(self):
        recorder = AFTraceRecorder()
        events = AccessList(_valid_events())
        with patch.object(recorder, "events_for", return_value=events):
            recorder.summarize(iteration())
        self.assertEqual(events.visits, len(events))

    def test_empty_off_and_clear_preserve_storage_contract(self):
        for enabled in (True, False):
            recorder = AFTraceRecorder(enabled)
            self.assertEqual(recorder.events_for(999), ())
            recorder.validate_invariants(999)
            self.assertEqual(
                recorder.export_summary(recorder.summarize(iteration()))["events"], []
            )
            for event in _valid_events():
                recorder.record(event)
            self.assertEqual(bool(recorder.iteration_event_indices), enabled)
            recorder.clear()
            self.assertEqual(recorder.events, [])
            self.assertEqual(recorder.iteration_event_indices, {})
            self.assertEqual(recorder.events_for(7), ())
            recorder.record(_valid_events()[0])
            self.assertEqual(
                recorder.iteration_event_indices, {7: [0]} if enabled else {}
            )

    def test_repeated_exports_are_detached_without_mutating_history(self):
        recorder = AFTraceRecorder()
        for event in _valid_events():
            recorder.record(event)
        summary = recorder.summarize(iteration())
        engine = SimpleNamespace(
            af_executor=SimpleNamespace(trace=recorder),
            af_iteration_summaries=[summary],
        )
        history = tuple(recorder.events)
        indexes = deepcopy(recorder.iteration_event_indices)
        first, second = get_af_stats(engine), get_af_stats(engine)
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(tuple(recorder.events), history)
        self.assertEqual(recorder.iteration_event_indices, indexes)
        self.assertEqual(first["af_trace"][0], reference_summary(recorder, iteration()))
        saved = deepcopy(first)
        recorder.record(replace(_valid_events()[0], iteration_id=8))
        recorder.record(replace(_valid_events()[0], node_id="new-event"))
        recorder.events[2].details["payload_bytes"] = 999
        summary["block_completed_at"][0] = 99
        summary["resource_utilization"].clear()
        self.assertEqual(first, saved)
        first["af_trace"][0]["events"][0]["source_id"] = "mutated result"
        self.assertEqual(second, saved)
        self.assertNotIn("events", summary)

    def test_engine_completion_never_serializes_events(self):
        for enabled in (True, False):
            with patch.object(
                AFTraceEvent,
                "as_dict",
                side_effect=AssertionError("early serialization"),
            ):
                env, engine, _ = _build_engine(
                    _typed_cluster(trace_enabled=enabled),
                    FixedComputeProvider(0.0002, 0.0003),
                    FixedTransferProvider(0.0001, 0.0001),
                )
                request = _run_request(env, engine)
                self.assertTrue(request.is_done)
                self.assertTrue(engine.af_iteration_summaries)
                self.assertTrue(
                    all("events" not in item for item in engine.af_iteration_summaries)
                )
                if not enabled:
                    self.assertIsNone(get_af_stats(engine)["af_trace"])
                    self.assertEqual(
                        engine.af_executor.trace.iteration_event_indices, {}
                    )
            exported = get_af_stats(engine)
            if enabled:
                self.assertTrue(all(item["events"] for item in exported["af_trace"]))

    def test_lookup_validation_and_summary_visits_grow_linearly(self):
        for steps in (8, 16, 32):
            result = lookup_visits(steps)
            self.assertEqual(result, dict(events=4 * steps, visits=12 * steps))


if __name__ == "__main__":
    unittest.main()
