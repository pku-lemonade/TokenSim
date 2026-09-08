import unittest
from dataclasses import replace

from TokenSim.af.trace import AFTraceEvent, AFTraceRecorder
from TokenSim.errors import SimulationStateError


def _event(
    node_id,
    operation,
    ready,
    start,
    end,
    *,
    predecessors=(),
    resource="worker:0",
    payload=None,
    details=None,
):
    return AFTraceEvent(
        iteration_id=7,
        block_id=0,
        layer_id=0,
        phase="decode",
        event_type="compute" if operation.startswith("compute") else "transfer",
        operation=operation,
        resource_id=resource,
        resource_ids=(resource,),
        ready_at=ready,
        start_at=start,
        end_at=end,
        queue_duration=start - ready,
        service_duration=end - start,
        request_count=1,
        token_count=1,
        source_id="test",
        node_id=node_id,
        predecessor_ids=predecessors,
        payload_bytes=payload,
        details=details or {},
    )


def _valid_events():
    return (
        _event("c0", "compute_a", 0.0, 0.0, 1.0, resource="worker:0"),
        _event("c1", "compute_a", 0.0, 0.0, 2.0, resource="worker:1"),
        _event(
            "send",
            "af_a_to_f",
            2.0,
            2.0,
            3.0,
            predecessors=("c0", "c1"),
            resource="link:a_to_f",
            payload=32,
            details={"payload_bytes": 32},
        ),
        _event(
            "return",
            "af_f_to_a",
            3.0,
            3.0,
            4.0,
            predecessors=("send",),
            resource="link:f_to_a",
            payload=16,
            details={"payload_bytes": 16},
        ),
    )


class AFTraceInvariantTest(unittest.TestCase):
    def _recorder(self, events=None, capacities=None):
        recorder = AFTraceRecorder(True, capacities)
        recorder.events.extend(events or _valid_events())
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


if __name__ == "__main__":
    unittest.main()
