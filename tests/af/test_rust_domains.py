"""Exact scheduling contracts for the optional native communication queue."""

import unittest

import simpy

from TokenSim.af.communication import resources
from TokenSim.errors import ConfigurationError
from scripts.af_rust import backend
from scripts.af_rust.domains import scheduler_type
from tests.af.communication import test_queue_equivalence as reference


@unittest.skipUnless(backend.BINARY.exists(), "optional Rust probe has not been built")
class RustDomainTest(unittest.TestCase):
    def scheduler(self, env=None):
        native = scheduler_type(backend.library())
        return native(env or simpy.Environment(), ("a", "b"))

    def test_fifo_cancellation_and_arrival_permutations_match_reference(self):
        harness = reference.QueueEquivalenceTest()
        native = scheduler_type(backend.library())
        for seed in range(20):
            expected = harness.simulate(
                reference.ReferenceScheduler, seed, cancellations=True
            )
            for reverse in (False, True):
                with self.subTest(seed=seed, reverse=reverse):
                    self.assertEqual(
                        harness.simulate(
                            native, seed, cancellations=True, reverse=reverse
                        ),
                        expected,
                    )

    def test_mixed_identity_order_and_duplicate_rejection(self):
        scheduler = self.scheduler()
        identities = (("a",), (2,), (1, "a"), (1,), ("0",), (1, 0))
        requests = [scheduler.request(("a",), identity) for identity in identities]
        self.assertEqual(
            [request.priority for request in scheduler.waiters],
            sorted(request.priority for request in requests),
        )
        with self.assertRaisesRegex(ConfigurationError, "unique stable"):
            scheduler.request(("a",), (1,))
        self.assertEqual(len(scheduler.waiters), len(requests))
        for request in requests:
            scheduler.release(request)
        scheduler.env.run()
        self.assertEqual(scheduler._native.storage, (0, 0, 1))

    def test_released_owner_does_not_remove_reused_waiter_priority(self):
        scheduler = self.scheduler()
        owner = scheduler.request(("a",), (1,))
        scheduler.env.run()
        waiter = scheduler.request(("a",), (1,))
        scheduler.release(owner)
        with self.assertRaisesRegex(ConfigurationError, "unique stable"):
            scheduler.request(("b",), (1,))
        scheduler.env.run()
        self.assertTrue(waiter.acquired)
        self.assertIs(scheduler.owners["a"], waiter)
        scheduler.release(waiter)
        scheduler.env.run()
        self.assertFalse(scheduler.owners)
        self.assertFalse(scheduler.waiters)

    def test_typed_boundary_rejects_unsupported_values_without_mutation(self):
        for identity in ((True,), (2**80,), (1.5,), ("\ud800",)):
            with self.subTest(identity=repr(identity)):
                scheduler = self.scheduler()
                with self.assertRaises(ConfigurationError):
                    scheduler.request(("a",), identity)
                self.assertEqual(scheduler._native.storage, (0, 0, 0))
        for timestamp in (2**53 + 1, 2**60):
            scheduler = self.scheduler(simpy.Environment(initial_time=timestamp))
            with self.assertRaisesRegex(ConfigurationError, "timestamp"):
                scheduler.request(("a",), (1,))

    def test_completed_requests_are_not_retained_across_many_services(self):
        scheduler = self.scheduler()

        def work():
            for index in range(3000):
                yield from scheduler.run(("a", "b"), 0.001, (index,))

        scheduler.env.process(work())
        scheduler.env.run()
        self.assertEqual(scheduler._native.storage, (0, 0, 2))

    def test_installation_restores_reference_scheduler_after_error(self):
        original = resources.DomainScheduler
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with backend.installed(("domains",)):
                self.assertIsNot(resources.DomainScheduler, original)
                raise RuntimeError("probe failure")
        self.assertIs(resources.DomainScheduler, original)
