"""Resource-front arbitration matches the prior global FIFO semantics."""

import random
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.communication.resources import DomainScheduler


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
