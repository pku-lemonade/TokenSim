"""Native control serialization, readiness, ordering and failure cleanup."""

import json
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.control import encode_eager_control
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.support import profiles as fixed
from tests.af.support.profiles import FixedHostFixture


class NativeControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixed.FixedHostFixture.setUpClass()
        cls.fixture = fixed.FixedHostFixture()

    def run_case(self, **costs):
        env, ex = self.fixture.executor(native_control_enabled=True, **costs)
        self.fixture.run_step(env, ex)
        return env, ex

    def test_wire_schema_and_decimal_size(self):
        for count in (1, 8, 16):
            encoded = encode_eager_control(count)
            expected = (
                '{"dp_metadata_list":{"0":{"max_tokens_across_dp_cpu":%d,'
                '"num_tokens_across_dp_cpu":[%d]}},"is_graph_capturing":false,'
                '"is_graph_replaying":false,"is_profile":false,"is_warmup":false}'
                % (count, count)
            ).encode()
            self.assertEqual(encoded, expected)
            self.assertEqual(
                json.loads(encoded)["dp_metadata_list"]["0"][
                    "num_tokens_across_dp_cpu"
                ],
                [count],
            )
        self.assertEqual(
            len(encode_eager_control(16)), len(encode_eager_control(8)) + 2
        )

    def test_two_control_messages_per_step_and_data_unchanged(self):
        env, ex = self.run_case(model=True)
        for _ in range(2):
            iteration_id = max(e.iteration_id for e in ex.trace.events)
            events = ex.trace.events_for(iteration_id)
            control = [e for e in events if e.event_type == "control_transfer"]
            data = [e for e in events if e.event_type == "transfer"]
            self.assertEqual(
                [e.payload_bytes for e in control], [8, len(encode_eager_control(1))]
            )
            self.assertEqual(
                [e.details["wire_dtype"] for e in control], ["int64", "uint8"]
            )
            self.assertTrue(
                all(
                    e.details["transport_backend"] == "torch_distributed_nccl"
                    for e in control
                )
            )
            self.assertTrue(
                all(e.details["transport_backend"] == "pynccl" for e in data)
            )
            self.assertEqual(len(data), 4)
            self.assertTrue(all(e.payload_bytes == 512 for e in data))
            self.fixture.run_step(env, ex)

    def test_length_readback_precedes_payload_and_decode_blocks_f(self):
        _, ex = self.run_case(
            control_length_readback_us=7,
            control_payload_readback_us=11,
            control_decode_us=200,
        )
        events = ex.trace.events
        by_id = {e.node_id: e for e in events}
        control = [e for e in events if e.event_type == "control_transfer"]
        self.assertAlmostEqual(control[0].start_at, 0)
        self.assertAlmostEqual(control[0].end_at, 5e-6)
        self.assertAlmostEqual(control[1].start_at, 12e-6)
        self.assertAlmostEqual(control[1].end_at, 17e-6)
        self.assertAlmostEqual(by_id["i0:control:decode"].end_at, 228e-6)
        attention = next(e for e in events if e.stage == "attention")
        self.assertAlmostEqual(attention.start_at, 17e-6)
        first_data = next(e for e in events if e.event_type == "transfer")
        self.assertAlmostEqual(first_data.start_at, 228e-6)
        self.assertAlmostEqual(first_data.queue_duration, 0)
        self.assertAlmostEqual(first_data.service_duration, 5e-6)
        self.assertAlmostEqual(first_data.details["readiness_wait_duration"], 111e-6)
        from TokenSim.af.trace.metrics import summarize_timeline

        summary = summarize_timeline(
            events, 0, max(e.end_at for e in events), ex.resources.resource_capacities()
        )
        self.assertAlmostEqual(summary["unexplained_duration"], 0)

    def test_split_block_is_rejected_in_single_stage_protocol(self):
        env, ex = self.fixture.executor(native_control_enabled=True)
        with self.assertRaisesRegex(ConfigurationError, "unsplit"):
            self.fixture.run_step(env, ex, count=17)
        self.assertFalse(ex.active_iterations)

    def test_cancel_control_wait_releases_resources(self):
        env, ex = self.fixture.executor(
            native_control_enabled=True, control_decode_us=200
        )
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = env.process(ex.run_iteration(ex.submit(0, [request], "decode")))

        def cancel():
            yield env.timeout(50e-6)
            process.interrupt("cancel control")

        env.process(cancel())
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

    def test_control_failure_propagates_and_cleans_state(self):
        env, ex = self.fixture.executor(native_control_enabled=True)
        provider = ex.transfer_provider

        class Broken:
            def estimate(self, query):
                raise ValueError("control provider failed")

        ex.transfer_provider = Broken()
        with self.assertRaisesRegex(SimulationStateError, "control provider failed"):
            self.fixture.run_step(env, ex)
        env.run()
        self.assertFalse(ex._native_controls)
        ex.transfer_provider = provider
        self.fixture.run_step(env, ex)

    def test_control_parameters_require_valid_enabled_configuration(self):
        for name in (
            "control_encode_us",
            "control_h2d_us",
            "control_length_readback_us",
            "control_payload_readback_us",
            "control_decode_us",
        ):
            for value in (-1, float("nan"), float("inf"), True):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises((ValueError, ConfigurationError)),
                ):
                    self.fixture.executor(native_control_enabled=True, **{name: value})
            with self.assertRaises(ConfigurationError):
                self.fixture.executor(**{name: 1})
        with self.assertRaises(ConfigurationError):
            self.fixture.executor(native_control_enabled=1)

    def test_transfer_cache_separates_backends(self):
        from dataclasses import replace
        from TokenSim.af.caching.estimates import cached_estimate
        from TokenSim.af.caching.estimates import initialize_cache
        from TokenSim.af.providers.queries import AFServiceEstimate
        from TokenSim.af.providers.queries import AFTransferQuery

        class Provider:
            fallback = "analytical_only"

            def __init__(self):
                initialize_cache(self)

            @cached_estimate
            def estimate(self, query):
                return AFServiceEstimate(
                    5 if query.transport_backend == "pynccl" else 7, "oracle"
                )

        query = AFTransferQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            direction="a_to_f",
            token_count=1,
            hidden_size=256,
            dtype="bf16",
            source_device="a",
            target_device="f",
            network="link",
            payload_bytes=512,
            transport_backend="pynccl",
        )
        provider = Provider()
        self.assertEqual(provider.estimate(query).service_duration, 5)
        self.assertEqual(
            provider.estimate(
                replace(query, transport_backend="torch_distributed_nccl")
            ).service_duration,
            7,
        )
        self.assertEqual(
            provider.estimate(replace(query, iteration_id=1)).service_duration, 5
        )
        self.assertEqual(provider.estimate_cache_hits, 1)

    def test_no_interleaved_eager_control_transactions(self):
        env, ex = self.fixture.executor(
            native_control_enabled=True, control_decode_us=200
        )
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        running = env.process(ex.run_iteration(ex.submit(0, [request], "decode")))
        env.run(until=1e-6)
        with self.assertRaisesRegex(SimulationStateError, "in-flight"):
            self.fixture.run_step(env, ex)
        env.run(until=running)
        self.assertFalse(ex.active_iterations)


class NativeOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FixedHostFixture.setUpClass()
        cls.fixture = FixedHostFixture()

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
