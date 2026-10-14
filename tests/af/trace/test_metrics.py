"""Trace invariant failures, fidelity labels and time accounting."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from TokenSim.af.providers.queries import AFEndpoint, AFTensorSlice
from TokenSim.af.trace.metrics import (
    AFTraceEvent,
    communication_window,
    summarize_message_waits,
    summarize_timeline,
)
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.errors import SimulationStateError
from TokenSim.reporting import get_af_stats
from tests.af.support.observations import _valid_events


def event(name, start, end, *, ready=None, predecessors=(), resource=None, **details):
    ready = start if ready is None else ready
    return AFTraceEvent(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase="decode",
        event_type="compute",
        resource_id=resource or name,
        ready_at=ready,
        start_at=start,
        end_at=end,
        queue_duration=start - ready,
        service_duration=end - start,
        request_count=1,
        token_count=1,
        source_id="oracle",
        node_id=name,
        predecessor_ids=predecessors,
        details=details,
    )


class TimelineTest(unittest.TestCase):
    def test_communication_wait_uses_elapsed_union_not_blocker_sum(self):
        for transport_start, wait in ((120, 100), (100, 80)):
            result = communication_window(
                20,
                transport_start + 5,
                transport_intervals=[(transport_start, transport_start + 5)],
            )
            self.assertEqual(result["waiting_duration"], wait)
            self.assertEqual(result["total_duration"], wait + 5)

    def test_fixed_service_is_not_also_waiting(self):
        result = communication_window(
            0, 15, fixed_intervals=[(0, 3), (0, 3)], transport_intervals=[(10, 15)]
        )
        self.assertEqual(result["waiting_duration"], 7)
        self.assertEqual(result["fixed_service_duration"], 3)
        with self.assertRaises(ValueError):
            communication_window(
                0, 15, fixed_intervals=[(0, 11)], transport_intervals=[(10, 15)]
            )

    def test_runtime_communication_report_reconstructs_each_window(self):
        from tests.af.communication.test_timing import run_case

        executor, iteration = run_case(submit_us=1, wire_us=10, complete_us=2)
        before = iteration.makespan
        rows = summarize_message_waits(
            executor.trace.events_for(iteration.iteration_id)
        )
        self.assertTrue(rows)
        self.assertTrue(any(r["waiting_duration"] > 0 for r in rows))
        for row in rows:
            self.assertAlmostEqual(
                row["total_duration"],
                row["waiting_duration"]
                + row["fixed_service_duration"]
                + row["transport_service_duration"],
            )
        self.assertEqual(iteration.makespan, before)

    def test_parallel_work_is_not_summed_into_makespan(self):
        rows = [
            event("a", 0, 10),
            event("b", 0, 20),
            event("c", 20, 25, predecessors=("a", "b")),
        ]
        summary = summarize_timeline(rows, 0, 25)
        self.assertEqual(summary["service_sum_by_category"]["compute"], 35)
        self.assertEqual(summary["busy_union_by_category"]["compute"], 25)
        self.assertEqual(summary["critical_duration_by_category"]["compute"], 25)
        self.assertEqual([s["node_id"] for s in summary["critical_chain"]], ["b", "c"])

    def test_resource_blocker_is_linked_without_double_counting_queue(self):
        rows = [
            event("a", 0, 5, resource="gpu"),
            event("b", 5, 10, ready=0, resource="gpu"),
        ]
        summary = summarize_timeline(rows, 0, 10, {"gpu": 1})
        self.assertEqual(summary["critical_duration_by_category"], {"compute": 10})
        self.assertEqual(summary["unexplained_duration"], 0)

    def test_external_queue_and_missing_dependencies_are_not_hidden(self):
        summary = summarize_timeline([event("a", 4, 8, ready=2)], 0, 8)
        self.assertEqual(
            summary["critical_duration_by_category"],
            {"unexplained": 2, "resource_queue": 2, "compute": 4},
        )

    def test_explicit_peer_wait_is_a_gap_not_network_service(self):
        rows = [
            event("a", 0, 2),
            event("b", 7, 12, predecessors=("a",), readiness_wait_started_at=2),
        ]
        summary = summarize_timeline(rows, 0, 12)
        self.assertEqual(
            summary["critical_duration_by_category"], {"compute": 7, "peer_wait": 5}
        )
        self.assertEqual(summary["unexplained_duration"], 0)

    def test_absent_trace_stays_unexplained(self):
        self.assertEqual(summarize_timeline([], 5, 9)["unexplained_duration"], 4)


class AFResultFidelityTest(unittest.TestCase):
    def _engine(self, phases, *, enabled=True):
        return SimpleNamespace(
            af_executor=(
                SimpleNamespace(trace=SimpleNamespace(enabled=False))
                if enabled
                else None
            ),
            af_iteration_summaries=[],
            af_runtime_summary=lambda: {},
            cluster_config=SimpleNamespace(af=SimpleNamespace(phases=tuple(phases))),
        )

    def test_decode_only_marks_prefill_aggregate_and_decode_rank_local(self):
        stats = get_af_stats(self._engine(("decode",)))

        self.assertEqual(
            stats["af_phase_fidelity"],
            {
                "prefill": "prefill_aggregate",
                "decode": "decode_af_rank_local",
            },
        )

    def test_disabled_af_has_no_fidelity_claim(self):
        stats = get_af_stats(self._engine((), enabled=False))

        self.assertIsNone(stats["af_phase_fidelity"])


class AFTraceInvariantTest(unittest.TestCase):
    def _recorder(self, events=None, capacities=None):
        recorder = AFTraceRecorder(True, capacities)
        for event in events or _valid_events():
            recorder.record(event)
        return recorder

    def test_valid_trace_passes(self):
        self._recorder().validate_invariants(7)

    def test_dependency_and_join_ready_time_are_checked(self):
        events = list(_valid_events())
        events[2] = replace(
            events[2],
            ready_at=1.5,
            start_at=1.5,
            queue_duration=0.0,
            end_at=2.5,
        )
        with self.assertRaisesRegex(SimulationStateError, "dependency time"):
            self._recorder(events).validate_invariants(7)

        events = list(_valid_events())
        events[2] = replace(events[2], ready_at=2.5, start_at=2.5, end_at=3.5)
        events[3] = replace(events[3], ready_at=3.5, start_at=3.5, end_at=4.5)
        with self.assertRaisesRegex(SimulationStateError, "join ready time"):
            self._recorder(events).validate_invariants(7)

    def test_resource_capacity_is_checked(self):
        events = list(_valid_events())
        events[1] = replace(
            events[1], resource_id="worker:0", resource_ids=("worker:0",)
        )
        with self.assertRaisesRegex(SimulationStateError, "capacity=1 exceeded"):
            self._recorder(events, {"worker:0": 1}).validate_invariants(7)

    def test_cross_domain_count_and_payload_are_checked(self):
        events = list(_valid_events())
        events[3] = replace(events[3], operation="ep_combine")
        with self.assertRaisesRegex(SimulationStateError, "exactly one af_f_to_a"):
            self._recorder(events).validate_invariants(7)

        events = list(_valid_events())
        events[2] = replace(events[2], details={"payload_bytes": 64})
        with self.assertRaisesRegex(SimulationStateError, "payload mismatch"):
            self._recorder(events).validate_invariants(7)

    def test_multiple_endpoint_legs_share_one_logical_stage(self):
        events = list(_valid_events())
        endpoint0 = AFEndpoint("a-domain", 0, "a:0", "a-device")
        endpoint1 = AFEndpoint("f-domain", 0, "f:0", "f-device")
        events[2] = replace(
            events[2],
            logical_stage_id="i7:b0:l0:a_to_f",
            leg_id="leg0",
            source_endpoint=endpoint0,
            target_endpoint=endpoint1,
            tensor_slices=(
                AFTensorSlice("token0", (0, 8), "sharded", payload_bytes=16),
            ),
            payload_bytes=16,
            details={"payload_bytes": 16},
        )
        second_leg = replace(
            events[2],
            node_id="send-1",
            leg_id="leg1",
            target_endpoint=replace(
                endpoint1, domain_rank=1, physical_worker="f:1", tp_rank=1
            ),
            tensor_slices=(
                AFTensorSlice("token0", (8, 16), "sharded", payload_bytes=16),
            ),
        )
        events.insert(3, second_leg)
        events[4] = replace(
            events[4],
            ready_at=3.0,
            start_at=3.0,
            end_at=4.0,
            predecessor_ids=("send", "send-1"),
        )

        self._recorder(events, {"link:a_to_f": 2}).validate_invariants(7)

        events[3] = replace(events[3], logical_stage_id="duplicate-a-to-f")
        with self.assertRaisesRegex(SimulationStateError, "one af_a_to_f logical"):
            self._recorder(events, {"link:a_to_f": 2}).validate_invariants(7)

    def test_duplicate_tensor_copy_and_f_residual_are_rejected(self):
        events = list(_valid_events())
        source = AFEndpoint("a-domain", 0, "a:0", "a-device")
        target = AFEndpoint("f-domain", 0, "f:0", "f-device")
        tensor_slice = AFTensorSlice("token0", (0, 8), "sharded", payload_bytes=16)
        events[2] = replace(
            events[2],
            logical_stage_id="send-stage",
            source_endpoint=source,
            target_endpoint=target,
            tensor_slices=(tensor_slice,),
            payload_bytes=16,
            details={"payload_bytes": 16},
        )
        events.insert(
            3,
            replace(
                events[2],
                node_id="send-copy",
                leg_id="copy",
            ),
        )
        events[4] = replace(events[4], predecessor_ids=("send", "send-copy"))
        with self.assertRaisesRegex(SimulationStateError, "duplicate physical tensor"):
            self._recorder(events).validate_invariants(7)

        residual = replace(
            _valid_events()[0],
            node_id="bad-residual",
            stage="residual",
            source_domain="f-domain",
        )
        with self.assertRaisesRegex(SimulationStateError, "residual cannot execute"):
            self._recorder(_valid_events() + (residual,)).validate_invariants(7)


if __name__ == "__main__":
    unittest.main()
