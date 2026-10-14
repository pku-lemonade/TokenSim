from dataclasses import replace
import unittest

from TokenSim.af.latency import AFEndpoint, AFTensorSlice
from TokenSim.af.trace import AFTraceRecorder
from TokenSim.errors import SimulationStateError
from tests.support.af_trace import _valid_events


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
