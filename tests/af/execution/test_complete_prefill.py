"""Complete-model LPX prefill, output-head ownership and cancellation."""

import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from scripts.lpx.config import Case
from scripts.lpx.native_config import configuration
from tests import test_lpx_native as native


class CompletePrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.NativeLPXTest.setUpClass()
        cls.fixture = native.NativeLPXTest

    def service(self):
        runtime = configuration(
            self.fixture.hardware,
            self.fixture.model,
            self.fixture.base,
            namespace="complete",
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        return env, service, session

    def test_chunks_skip_output_head_and_use_no_af_providers(self):
        env, service, session = self.service()
        with patch.object(
            service.executor.compute_provider,
            "estimate",
            side_effect=AssertionError("AF compute used for prefill"),
        ):
            env.run(
                until=env.process(
                    service.forward(
                        [session], phase="prefill", query_tokens=2, emit_logits=False
                    )
                )
            )
            self.assertTrue(service.prefill.backends)
            self.assertTrue(
                all(
                    backend.stats.component_seconds["lm_head"] == 0
                    for backend in service.prefill.backends.values()
                )
            )
            env.run(
                until=env.process(
                    service.forward([session], phase="prefill", query_tokens=2)
                )
            )
        self.assertEqual(session.cached_context, 20)
        self.assertGreater(
            sum(
                backend.stats.component_seconds["lm_head"]
                for backend in service.prefill.backends.values()
            ),
            0,
        )
        self.assertEqual(service.executor._next_iteration_id, 0)

    def test_cancellation_releases_compute_and_queued_collectives(self):
        for delay in (1e-9, 0.001):
            with self.subTest(delay=delay):
                env, service, session = self.service()
                registry = service.executor.resources.endpoint_resources
                link = service.runtime.cluster.af_link
                endpoints = tuple(
                    worker.physical_worker_ref.identity
                    for worker in service.runtime.workers
                )
                keys = tuple(
                    registry.format_key(key)
                    for key, _ in registry.for_step(
                        link.network, "collective", endpoints
                    )
                )

                def block():
                    yield from registry.run_transfer(keys, 1, 0, ("block",))

                blocker = env.process(block())
                forward = env.process(
                    service.forward([session], phase="prefill", query_tokens=2)
                )
                env.run(until=delay)
                if delay > 1e-9:
                    self.assertTrue(registry.scheduler.waiters)
                forward.interrupt()
                with self.assertRaises(simpy.Interrupt):
                    env.run(until=forward)
                self.assertEqual(session.cached_context, 16)
                self.assertEqual(service.memory.leases[0].current_context, 16)
                self.assertFalse(registry.scheduler.waiters)
                for resource in service.executor.resources.compute_slots.values():
                    self.assertEqual(resource.count, 0)
                    self.assertFalse(resource.queue)
                blocker.interrupt()
                with self.assertRaises(simpy.Interrupt):
                    env.run(until=blocker)
                self.assertFalse(registry.scheduler.owners)

    def test_lpx_scenarios_reject_split_prefill_layouts(self):
        for settings in (
            dict(
                name="af", mode="af", decode=self.fixture.base, prefill=self.fixture.lpu
            ),
            dict(
                name="draft",
                mode="draft",
                decode=self.fixture.base,
                draft=self.fixture.lpu,
            ),
        ):
            with (
                self.subTest(settings=settings),
                self.assertRaisesRegex(
                    ConfigurationError, "prefill requires a complete-model"
                ),
            ):
                Case(**settings)
