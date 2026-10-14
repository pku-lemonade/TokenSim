"""Resource-front arbitration matches the prior global FIFO semantics."""

import random
import unittest

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
    def simulate(self, scheduler_type, seed):
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
            rows[index] = yield from scheduler.run(resource_keys, duration, (index,))

        for index, item in enumerate(work):
            env.process(run(index, item))
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
