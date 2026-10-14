"""Endpoint budgets permit disjoint traffic and reserve overlaps atomically."""

import unittest

import simpy

from TokenSim.af.communication.endpoints import EndpointResources


class EndpointBudgetTest(unittest.TestCase):
    def test_disjoint_transfers_overlap_and_aliases_share_injection(self):
        env = simpy.Environment()
        resources = EndpointResources(
            env, {"net": "c2c", "alias": "c2c"}, ("a", "b", "c", "d")
        )
        rows = {}

        def run(order, name, network, pair):
            keys = tuple(
                resources.format_key(key)
                for key, _ in resources.for_step(network, "a_to_f", pair)
            )
            rows[name] = yield from resources.scheduler.run(keys, 2, (order, name))

        for order, args in enumerate(
            (
                ("ab", "net", ("a", "b")),
                ("cd", "net", ("c", "d")),
                ("ad", "alias", ("a", "d")),
            )
        ):
            env.process(run(order, *args))
        env.run()
        self.assertEqual(rows["ab"], (0, 0, 2))
        self.assertEqual(rows["cd"], (0, 0, 2))
        self.assertEqual(rows["ad"], (0, 2, 4))
        self.assertFalse(resources.scheduler.owners)

    def test_waiting_collective_holds_no_subset_and_cancellation_releases(self):
        env = simpy.Environment()
        resources = EndpointResources(env, {"net": "c2c"}, ("a", "b"))
        keys = [
            resources.format_key(key)
            for key, _ in resources.for_step("net", "all_reduce", ("a", "b"))
        ]
        active = resources.scheduler.request(keys[:1], (0,))
        waiting = resources.scheduler.request(keys, (1,))
        env.run()
        self.assertTrue(active.acquired)
        self.assertFalse(waiting.acquired)
        self.assertEqual(set(resources.scheduler.owners), set(keys[:1]))
        resources.scheduler.release(waiting)
        resources.scheduler.release(active)
        env.run()
        self.assertFalse(resources.scheduler.owners)
        self.assertFalse(resources.scheduler.waiters)
