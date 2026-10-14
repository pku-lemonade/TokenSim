from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.errors import ConfigurationError
from tests.af.support.host_fixture import FixedHostFixture


"""Af fixed host."""


class FixedHostTest(FixedHostFixture, unittest.TestCase):
    def test_receiver_preparation_overlaps_compute(self):
        env, ex = self.executor(f_step_prepare_us=200)
        self.run_step(env, ex)
        first = next(e for e in ex.trace.events if e.event_type == "transfer")
        self.assertAlmostEqual(first.start_at, 200e-6)
        self.assertAlmostEqual(first.end_at, 205e-6)
        self.assertAlmostEqual(first.queue_duration, 0)
        self.assertAlmostEqual(first.details["readiness_wait_duration"], 100e-6)
        self.assertEqual(
            len([e for e in ex.trace.events if e.event_type == "host_prepare"]), 1
        )

    def test_costs_apply_to_whole_model_and_each_iteration(self):
        env, ex = self.executor(
            model=True,
            a_step_prepare_us=10,
            f_step_prepare_us=200,
            compute_submit_us=3,
            host_receive_submit_us=2,
            host_completion_us=4,
        )
        for _ in range(2):
            iteration = self.run_step(env, ex)
            events = [
                e for e in ex.trace.events if e.iteration_id == iteration.iteration_id
            ]
            self.assertEqual(
                len([e for e in events if e.event_type == "host_prepare"]), 2
            )
            compute = [e for e in events if e.event_type == "compute"]
            submits = [
                e
                for e in events
                if e.details.get("cost_scope") == "per_compute_dag_node"
            ]
            self.assertEqual(len(compute), len(submits))
            by_id = {e.node_id: e for e in events}
            for event in compute:
                submit = by_id[event.node_id + ":compute_submit"]
                self.assertGreaterEqual(event.start_at, submit.end_at)
            self.assertFalse(ex._step_preparation)

    def test_invalid_costs_rejected(self):
        for field in (
            "a_step_prepare_us",
            "f_step_prepare_us",
            "compute_submit_us",
            "host_receive_submit_us",
        ):
            for value in (-1, float("nan"), float("inf"), True):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises((ValueError, ConfigurationError)),
                ):
                    self.executor(**{field: value})

    def test_cancel_preparation_releases_host(self):
        env, ex = self.executor(f_step_prepare_us=200)
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        iteration = ex.submit(0, [request], "decode")
        process = env.process(ex.run_iteration(iteration))

        def cancel():
            yield env.timeout(10e-6)
            process.interrupt("test cancellation")

        env.process(cancel())
        with self.assertRaises(simpy.Interrupt):
            env.run(until=process)
        env.run()
        self.assertFalse(ex._step_preparation)
        self.assertTrue(
            all(
                not r.users and not r.queue
                for r in ex.resources.host_submission_slots.values()
            )
        )
        self.run_step(env, ex)

    def test_many_blocks_charge_preparation_once(self):
        env, ex = self.executor(a_step_prepare_us=10, f_step_prepare_us=200)
        iteration = self.run_step(env, ex, count=17)
        self.assertGreater(len(iteration.blocks), 1)
        self.assertEqual(
            len([e for e in ex.trace.events if e.event_type == "host_prepare"]), 2
        )

    def test_shared_host_serializes_services_but_not_device_compute(self):
        env, ex = self.executor(
            a_step_prepare_us=10, f_step_prepare_us=200, host_resource_scope="shared"
        )
        self.run_step(env, ex)
        prep = [e for e in ex.trace.events if e.event_type == "host_prepare"]
        self.assertAlmostEqual(sum(e.service_duration for e in prep), 210e-6)
        self.assertAlmostEqual(max(e.end_at for e in prep), 210e-6)
        first = next(e for e in ex.trace.events if e.event_type == "transfer")
        self.assertAlmostEqual(first.start_at, 210e-6)
        self.assertAlmostEqual(first.service_duration, 5e-6)
        self.assertAlmostEqual(first.queue_duration, 0)

    def test_cancel_receiver_submission_releases_host(self):
        env, ex = self.executor(host_receive_submit_us=200)
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = env.process(ex.run_iteration(ex.submit(0, [request], "decode")))

        def cancel():
            yield env.timeout(150e-6)
            process.interrupt("cancel receive")

        env.process(cancel())
        with self.assertRaises(simpy.Interrupt):
            env.run(until=process)
        env.run()
        self.assertTrue(
            all(
                not r.users and not r.queue
                for r in ex.resources.host_submission_slots.values()
            )
        )
        self.run_step(env, ex)

    def test_nonzero_costs_require_native_separated_mode(self):
        fixture = self.fixture
        for override in (
            {"execution_protocol": "partial_return"},
            {"communication_timing": {"enabled": False, "a_step_prepare_us": 1}},
        ):
            doc = fixture.document(
                communication_timing=dict(enabled=True, a_step_prepare_us=1)
            )
            doc["af"].update(override)
            with self.assertRaises(ConfigurationError):
                build_model_runtime(
                    doc, hardware=fixture.base.hardware, model=fixture.model
                )
