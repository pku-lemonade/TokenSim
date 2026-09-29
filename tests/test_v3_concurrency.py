import unittest
import simpy
from scripts.run_v3_concurrency import closed_loop


class ClosedLoopTest(unittest.TestCase):
    def test_round_robin_wait_replacement_and_common_window(self):
        env=simpy.Environment()
        tokens,batches,requests,window=[],[],[],{}
        def serve(reqs,round_id):
            yield env.timeout(1)
            return {r.id:env.now for r in reqs}
        env.process(closed_loop(env,users=4,max_batch=2,output_tokens=2,
            warmup_rounds=1,measure_rounds=3,context=2048,serve=serve,
            tokens=tokens,batches=batches,lifecycles=requests,windows=window))
        env.run()
        self.assertEqual(window,{'start_s':2,'end_s':8})
        selected=[t for t in tokens if t['measured']]
        self.assertEqual(len(selected),12)
        self.assertEqual([b['users'] for b in batches],[[0,1],[2,3]]*4)
        self.assertEqual([r['arrival_s'] for r in requests[4:8]],[3,3,4,4])
        self.assertTrue(all(t['itl_ms']==2000 for t in selected if t['itl_ms'] is not None))
        self.assertTrue(all(t['scheduler_wait_ms']==1000 for t in selected))
        self.assertEqual([t['context_len'] for t in tokens[:6]],[2048]*4+[2049]*2)
        self.assertEqual(sum('completion_s' in r for r in requests),8)
        duration=window['end_s']-window['start_s']
        self.assertEqual(len(selected)/duration,4*(3/duration))


if __name__=='__main__':
    unittest.main()
