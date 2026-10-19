"""Physical F queue accounting is independent of the retained member order."""

import unittest

from TokenSim.af.serving.metrics import ResourceMetrics, steady_metrics
from TokenSim.af.serving.timing import CompletedTiming
from tests.af.execution.test_batching import batch_runtime, submit_owner


class BatchTimingTest(unittest.TestCase):
    def test_collection_wait_survives_shared_host_submission_and_compaction(self):
        env, executor, _ = batch_runtime(owners=2, submit_us=2)
        for owner in range(2):
            env.process(executor.run_iteration(submit_owner(executor, owner)[0]))
        env.run()
        members = [event for event in executor.trace.events if event.execution_group_id]
        self.assertEqual(len(members), 2)
        self.assertEqual(members[0].input_ready_at, members[1].input_ready_at)
        self.assertEqual(
            members[0].resource_queue_intervals, members[1].resource_queue_intervals
        )
        first, last = members[0].resource_queue_intervals[0]
        self.assertAlmostEqual(last - first, 5e-6)
        resource = members[0].resource_id
        for events in (
            executor.trace.events,
            [CompletedTiming.from_event(event) for event in executor.trace.events],
        ):
            for start, end in (
                (0, env.now),
                (first + 1e-6, last - 1e-6),
                (last, members[0].start_at),
            ):
                results = [
                    steady_metrics([], order, start=start, end=end)
                    for order in (events, list(reversed(events)))
                ]
                self.assertEqual(results[0], results[1])
                expected = max(0, min(end, last) - max(start, first))
                self.assertAlmostEqual(
                    results[0]["resource_queue_s"][resource], expected
                )
                online = ResourceMetrics(dict(start=start, end=end))
                for event in events:
                    online.observe(event)
                self.assertEqual(online.summary([]), results[0])
        for iteration_id in (0, 1):
            executor.trace.validate_invariants(iteration_id)

    def test_unequal_arrivals_use_one_shared_collection_interval(self):
        env, executor, _ = batch_runtime(owners=2)
        left, _ = submit_owner(executor, 0)
        right, _ = submit_owner(executor, 1)
        env.process(executor.run_iteration(left))

        def later():
            yield env.timeout(2e-6)
            yield from executor.run_iteration(right)

        env.process(later())
        env.run()
        members = [event for event in executor.trace.events if event.execution_group_id]
        self.assertNotEqual(members[0].input_ready_at, members[1].input_ready_at)
        forward = steady_metrics([], executor.trace.events, start=0, end=env.now)
        reverse = steady_metrics(
            [], reversed(executor.trace.events), start=0, end=env.now
        )
        self.assertEqual(forward, reverse)
        self.assertAlmostEqual(
            forward["resource_queue_s"][members[0].resource_id], 5e-6
        )
