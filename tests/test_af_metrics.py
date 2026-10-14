"""AF timing breakdowns and result fidelity."""

import sys
from types import SimpleNamespace
import types
import unittest

from TokenSim.af.timeline import (
    communication_window,
    summarize_message_waits,
    summarize_timeline,
)
from TokenSim.af.trace import AFTraceEvent
from TokenSim.reporting import get_af_stats


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
        from tests.test_af_communication_timing import run_case

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


if __name__ == "__main__":
    unittest.main()


try:
    import TransformerRoofline  # noqa: F401
except (ImportError, OSError):
    fake_roofline = types.ModuleType("TransformerRoofline")
    fake_roofline.TransformerRoofline = object
    sys.modules["TransformerRoofline"] = fake_roofline


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

    def test_prefill_af_is_labeled_as_compatibility_path(self):
        stats = get_af_stats(self._engine(("prefill",)))

        self.assertEqual(
            stats["af_phase_fidelity"],
            {
                "prefill": "af_prefill_compatibility",
                "decode": "decode_aggregate",
            },
        )

    def test_disabled_af_has_no_fidelity_claim(self):
        stats = get_af_stats(self._engine((), enabled=False))

        self.assertIsNone(stats["af_phase_fidelity"])


if __name__ == "__main__":
    unittest.main()
