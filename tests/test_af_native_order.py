import unittest
from types import SimpleNamespace
import simpy
from tests.test_af_fixed_host import FixedHostTest
from TokenSim.errors import ConfigurationError, SimulationStateError


class NativeOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FixedHostTest.setUpClass()
        cls.fixture = FixedHostTest()

    def executor(self, **costs):
        return self.fixture.executor(
            model=True,
            native_control_enabled=True,
            native_call_order_enabled=True,
            **costs,
        )

    def test_control_enqueue_and_device_completion_are_separate(self):
        env, ex = self.executor(
            a_step_prepare_us=10,
            f_step_prepare_us=100,
            host_receive_submit_us=20,
            control_length_readback_us=50,
            compute_submit_us=3,
        )
        self.fixture.run_step(env, ex)
        events = ex.trace.events_for(0)
        by_id = {e.node_id: e for e in events}
        payload_send = by_id["i0:control:payload:host_submit"]
        length_wire = by_id["i0:control:length:transport"]
        payload_wire = by_id["i0:control:payload:transport"]
        self.assertLess(payload_send.end_at, length_wire.end_at)
        self.assertGreaterEqual(
            by_id["i0:control:payload:receive_submit"].start_at,
            by_id["i0:control:length_readback"].end_at,
        )
        embedding = next(e for e in events if e.operation == "embedding_lookup")
        cpu = by_id[embedding.node_id + ":compute_submit"]
        self.assertLess(cpu.end_at, payload_wire.end_at)
        self.assertGreaterEqual(embedding.start_at, payload_wire.end_at)
        ex.trace.validate_invariants(0)

    def test_receivers_prepost_without_occupying_transport(self):
        env, ex = self.executor(host_receive_submit_us=3)
        self.fixture.run_step(env, ex)
        by_id = {e.node_id: e for e in ex.trace.events_for(0)}
        wires = [e for e in by_id.values() if e.event_type == "transfer"]
        for wire in wires:
            root = wire.node_id.removesuffix(":transport")
            recv = by_id[root + ":receive_submit"]
            send = by_id[root + ":host_submit"]
            self.assertLess(recv.start_at, send.start_at)
            self.assertGreaterEqual(wire.start_at, max(recv.end_at, send.end_at))
            self.assertEqual(
                wire.details["readiness_mode"], "native_preposted_call_order"
            )
        self.assertFalse(ex._native_controls)

    def test_trace_cache_invariance(self):
        times = []
        for trace, cache in ((True, True), (False, False), (True, False)):
            env, ex = self.executor(
                trace=trace,
                cache=cache,
                host_receive_submit_us=13,
                control_length_readback_us=9,
                compute_submit_us=7,
            )
            times.append(self.fixture.run_step(env, ex).makespan)
        self.assertEqual(times, [times[0]] * 3)

    def test_cancellation_cleans_preposted_calls(self):
        env, ex = self.executor(a_step_prepare_us=1000000, host_receive_submit_us=3)
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = env.process(ex.run_iteration(ex.submit(0, [request], "decode")))
        env.run(until=10e-6)
        process.interrupt("cancel preposted iteration")
        with self.assertRaises(simpy.Interrupt):
            env.run(until=process)
        env.run()
        self.assertFalse(ex._native_controls)
        self.assertTrue(
            all(
                not r.users and not r.queue
                for r in ex.resources.host_submission_slots.values()
            )
        )
        self.fixture.run_step(env, ex)

    def test_requires_native_control(self):
        with self.assertRaises(ConfigurationError):
            self.fixture.executor(native_call_order_enabled=True)

    def test_failed_control_estimate_unblocks_and_can_retry(self):
        for operation in ("af_control_length", "af_control_payload"):
            env, ex = self.executor(host_receive_submit_us=3)
            original = ex.transfer_provider.estimate

            def failure(query):
                if query.operation == operation:
                    raise ValueError("injected transfer failure")
                return original(query)

            ex.transfer_provider.estimate = failure
            with self.assertRaisesRegex(
                SimulationStateError, "injected transfer failure"
            ):
                self.fixture.run_step(env, ex)
            env.run()
            self.assertFalse(ex._native_controls)
            self.assertFalse(ex.active_iterations)
            self.assertTrue(
                all(
                    not r.users and not r.queue
                    for r in ex.resources.host_submission_slots.values()
                )
            )
            ex.transfer_provider.estimate = original
            self.fixture.run_step(env, ex)


if __name__ == "__main__":
    unittest.main()
