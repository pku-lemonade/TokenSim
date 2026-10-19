"""Endpoint budgets permit disjoint traffic and reserve overlaps atomically."""

import random
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.communication.resources import DomainScheduler
from TokenSim.af.serving.metrics import steady_metrics
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.af.trace.occupancy import validate_resource_capacity


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


class ReferenceScheduler(DomainScheduler):
    def _arbitrate(self):
        self._scheduled = False
        blocked = set()
        for request in sorted(self._waiters.values(), key=lambda item: item.priority):
            if request.cancelled:
                continue
            if blocked.intersection(request.keys) or any(
                key in self.owners for key in request.keys
            ):
                blocked.update(request.keys)
                continue
            request.acquired = True
            self._waiters.pop(id(request))
            self._priorities.remove(request.priority)
            for key in request.keys:
                self.owners[key] = request
            request.event.succeed()


class QueueEquivalenceTest(unittest.TestCase):
    def simulate(self, scheduler_type, seed, *, cancellations=False, reverse=False):
        env = simpy.Environment()
        keys = tuple(f"endpoint:{i}" for i in range(8))
        scheduler = scheduler_type(env, keys)
        rng = random.Random(seed)
        work = [
            (
                tuple(rng.sample(keys, rng.randint(1, 3))),
                rng.choice((0, 0.01, 0.02)),
                rng.choice((0, 0.005)),
            )
            for _ in range(80)
        ]
        rows = {}

        def run(index, item):
            resource_keys, duration, arrival = item
            yield env.timeout(arrival)
            try:
                rows[index] = yield from scheduler.run(
                    resource_keys, duration, (index,)
                )
            except simpy.Interrupt:
                rows[index] = ("cancelled", env.now)

        indices = list(range(len(work)))
        if reverse:
            indices.reverse()
        processes = {index: env.process(run(index, work[index])) for index in indices}
        if cancellations:

            def cancel():
                yield env.timeout(0.015)
                for index in indices:
                    if index % 7 == 0 and processes[index].is_alive:
                        processes[index].interrupt("test cancellation")

            env.process(cancel())
        env.run()
        self.assertFalse(scheduler.owners)
        self.assertFalse(scheduler.waiters)
        return rows

    def test_ready_queue_order_and_intervals_are_unchanged(self):
        for seed in (0, 1, 42):
            with self.subTest(seed=seed):
                self.assertEqual(
                    self.simulate(DomainScheduler, seed),
                    self.simulate(ReferenceScheduler, seed),
                )

    def test_cancellation_and_simultaneous_arrival_order_preserve_fifo(self):
        for seed in range(20):
            expected = self.simulate(ReferenceScheduler, seed, cancellations=True)
            for reverse in (False, True):
                with self.subTest(seed=seed, reverse=reverse):
                    self.assertEqual(
                        self.simulate(
                            DomainScheduler, seed, cancellations=True, reverse=reverse
                        ),
                        expected,
                    )

    def test_disjoint_activity_does_not_rescan_blocked_endpoints(self):
        env = simpy.Environment()
        keys = tuple(f"blocked:{index}" for index in range(256))
        scheduler = DomainScheduler(env, (*keys, "free"))
        holders = [
            scheduler.request((key,), ("hold", index)) for index, key in enumerate(keys)
        ]
        env.run()
        waiting = [
            scheduler.request((key,), ("wait", index)) for index, key in enumerate(keys)
        ]
        env.run()
        with patch.object(scheduler, "_front", wraps=scheduler._front) as front:
            env.process(scheduler.run(("free",), 1, ("free",)))
            env.run()
            self.assertLessEqual(front.call_count, 4)
        for request in (*waiting, *holders):
            scheduler.release(request)
        env.run()
        self.assertFalse(scheduler.waiters)
        self.assertFalse(scheduler.owners)
