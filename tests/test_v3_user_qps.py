import unittest
from types import SimpleNamespace

import simpy

from scripts.run_v3_user_qps import schedule, validate_and_summarize, workload


class UserQPSTest(unittest.TestCase):
    def run_schedule(self, arrivals, *, users=2, batch=1, length=2, policy='queue'):
        env = simpy.Environment()
        requests, tokens, batches, queues = [], [], [], []

        def serve(reqs, iteration):
            yield env.timeout(1)
            return {r.id:env.now for r in reqs}

        schedule(env, arrivals, users=users, max_batch=batch, output_tokens=length,
            context=32, serve=serve, request_rows=requests, token_rows=tokens,
            batch_rows=batches, queue_rows=queues, busy_policy=policy)
        env.run()
        args = SimpleNamespace(users=users,max_batch=batch,output_tokens=length,
            context=32,a_tp=4,f_ep=32,af_batch_size=8)
        summary=validate_and_summarize(args,1,arrivals,requests,tokens,batches,queues)
        return requests,tokens,batches,queues,summary

    def test_arrivals_during_decode_queue_per_user_and_round_robin(self):
        arrivals=[dict(request_id=0,user=0,arrival_s=.1),
                  dict(request_id=1,user=0,arrival_s=.2),
                  dict(request_id=2,user=1,arrival_s=.3)]
        requests,tokens,batches,queues,summary=self.run_schedule(arrivals)
        self.assertEqual([t['request_id'] for t in tokens],[0,2,0,2,1,1])
        self.assertEqual([t['context_len'] for t in tokens],[32,32,33,33,32,33])
        self.assertAlmostEqual(requests[0]['completion_s'],3.1)
        self.assertAlmostEqual(requests[1]['eligible_s'],3.1)
        self.assertAlmostEqual(requests[1]['first_admitted_s'],4.1)
        self.assertAlmostEqual(summary['total_tps'],6/6.1)
        self.assertEqual(summary['max_pending'],1)
        self.assertEqual(summary['max_active_users'],2)
        self.assertEqual(summary['completed_requests'],3)
        self.assertEqual(summary['unfinished_at_last_arrival'],3)
        self.assertEqual(summary['tps_during_arrivals'],0)

    def test_dynamic_batch_never_duplicates_user_and_drains(self):
        arrivals=[dict(request_id=i,user=i%3,arrival_s=.1+i*.01) for i in range(6)]
        requests,tokens,batches,queues,summary=self.run_schedule(arrivals,users=3,batch=2)
        self.assertEqual(summary['total_output_tokens'],12)
        self.assertEqual(summary['max_actual_batch'],2)
        self.assertEqual(batches[0]['batch_size'],1)
        self.assertTrue(all(len(set(b['users']))==b['batch_size'] for b in batches))
        self.assertEqual(queues[-1]['active_users'],0)
        self.assertEqual(queues[-1]['pending'],0)

    def test_drop_policy_preserves_offered_arrivals_but_counts_delivered_tokens(self):
        arrivals=[dict(request_id=0,user=0,arrival_s=.1),
                  dict(request_id=1,user=0,arrival_s=.2)]
        requests,tokens,batches,queues,summary=self.run_schedule(arrivals,policy='drop')
        self.assertEqual(requests[1]['status'],'dropped')
        self.assertEqual(summary['dropped_requests'],1)
        self.assertEqual(summary['completed_requests'],1)
        self.assertAlmostEqual(summary['total_tps'],2/2.1)

    def test_poisson_qps_scaling_keeps_request_users_paired(self):
        slow=workload(.02,10000,32,42)
        fast=workload(.08,10000,32,42)
        self.assertEqual([r['user'] for r in slow],[r['user'] for r in fast])
        self.assertEqual([r['arrival_s'] for r in slow],[4*r['arrival_s'] for r in fast])
        self.assertEqual(len({r['user'] for r in slow}),32)
        self.assertAlmostEqual(len(slow)/slow[-1]['arrival_s'],.02,delta=.0005)

    def test_unique_users_assigns_each_request_to_a_distinct_user(self):
        rows=workload(.3,32,32,42,unique_users=True)
        self.assertEqual(len({r['user'] for r in rows}),32)
        self.assertEqual(sorted(r['user'] for r in rows),list(range(32)))
        with self.assertRaisesRegex(ValueError,'requests <= users'):
            workload(.3,3,2,42,unique_users=True)


if __name__=='__main__':
    unittest.main()
