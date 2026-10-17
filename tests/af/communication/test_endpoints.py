"""Endpoint budgets permit disjoint traffic and reserve overlaps atomically."""

import unittest

import simpy
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.af.trace.occupancy import validate_resource_capacity
from TokenSim.af.serving.metrics import steady_metrics

from TokenSim.af.communication.endpoints import EndpointResources


class EndpointBudgetTest(unittest.TestCase):
    def test_hop_latency_keeps_bytes_serialized_and_messages_in_flight(self):
        env = simpy.Environment()
        resources = EndpointResources(
            env, {"net": "c2c"}, ("a", "b", "c"), transfer_occupancy="wire"
        )
        rows, events = {}, []

        def move(index, target):
            keys = tuple(
                resources.format_key(key)
                for key, _ in resources.for_step("net", "a_to_f", ("a", target))
            )
            ready, start, end, injected = yield from resources.run_transfer(
                keys, 11, 10, (index,)
            )
            rows[target] = (ready, start, end, injected)
            events.append(
                AFTraceEvent(
                    0,
                    0,
                    0,
                    "decode",
                    "transfer",
                    keys[0],
                    ready,
                    start,
                    end,
                    start - ready,
                    end - start,
                    1,
                    1,
                    "analytical",
                    resource_ids=keys,
                    node_id=target,
                    details={"resource_occupancy_end_at": injected},
                )
            )

        for index, target in enumerate(("b", "c")):
            env.process(move(index, target))
        env.run(until=3)
        self.assertFalse(resources.scheduler.owners)
        self.assertFalse(rows)
        env.run()
        self.assertEqual(rows["b"], (0, 0, 11, 1))
        self.assertEqual(rows["c"], (0, 1, 12, 2))
        validate_resource_capacity(events, resources.capacities())
        metric = steady_metrics([], events, start=0, end=12)
        self.assertAlmostEqual(
            metric["resource_utilization"]["link:endpoint:c2c:a:tx"], 2 / 12
        )

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
