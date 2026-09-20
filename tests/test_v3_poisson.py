import unittest
import simpy
from scripts.run_v3_poisson import arrival_times, schedule


class PoissonSchedulerTest(unittest.TestCase):
    def test_arrival_rate_scaling_and_mean(self):
        slow = arrival_times(2, 10000, 42)
        fast = arrival_times(8, 10000, 42)
        self.assertEqual(slow, [4*x for x in fast])
        self.assertTrue(all(a < b for a, b in zip([0]+slow, slow)))
        self.assertAlmostEqual(10000/slow[-1], 2, delta=.05)

    def test_arrivals_continue_during_service_and_fifo_batches_drain(self):
        env = simpy.Environment()
        requests, batches, queues = [], [], []
        def serve(ids):
            yield env.timeout(1)
            return {i:env.now for i in ids}
        schedule(env, [.1,.2,.3,.4], 2, serve, requests, batches, queues)
        env.run()
        self.assertEqual([b['batch_size'] for b in batches], [1,2,1])
        self.assertEqual([r['request_id'] for r in requests], [0,1,2,3])
        self.assertAlmostEqual(requests[-1]['complete_s'], 3.1)
        self.assertAlmostEqual(requests[1]['queue_ms'], 900)
        self.assertEqual(max(q['pending'] for q in queues), 3)


if __name__ == '__main__':
    unittest.main()
