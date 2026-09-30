"""Step expectations from vLLM scheduler.py 30f5c01d4ebb schedule()."""

from __future__ import annotations

import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMWorker
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from tests.test_chunked_prefill import _request, _scheduler, _run_step, _env
from tests.test_agentx_routing_dp import engine
import simpy


def apply(scheduler, output, now):
    for request in output.running:
        request.advance(_env(now), 0.01, len(output.running))
    scheduler.update(output.running)


class PrefillCadenceTest(unittest.TestCase):
    def test_two_requests_each_receive_512_and_single_request_uses_whole_budget(self):
        scheduler = _scheduler(8192, long_prefill_token_threshold=512)
        a, b = _request(0, 8192), _request(1, 8192)
        scheduler.add_requests([a, b])
        output = scheduler.schedule_output()
        self.assertEqual([r.scheduled_tokens for r in output.running], [512, 512])
        single = _scheduler(8192, long_prefill_token_threshold=512)
        c = _request(2, 4096)
        single.add_requests([c])
        self.assertEqual(single.schedule_output().running, [c])
        self.assertEqual(c.scheduled_tokens, 4096)

    def test_adaptive_fair_share_is_explicit_and_disabled_by_default(self):
        for adaptive, expected in ((False, 512), (True, 2048)):
            scheduler = _scheduler(
                4096,
                long_prefill_token_threshold=512,
                adaptive_long_prefill_threshold=adaptive,
            )
            scheduler.add_requests([_request(0, 8192), _request(1, 8192)])
            self.assertEqual(
                [r.scheduled_tokens for r in scheduler.schedule_output().running],
                [expected, expected],
            )

    def decoding_and_partial(self, *, max_seqs=None):
        scheduler = _scheduler(
            1024, long_prefill_token_threshold=512, max_parallem_sum=max_seqs
        )
        decode = _request(0, 16, decode_len=20)
        partial = _request(1, 2048, decode_len=20)
        scheduler.add_requests([decode, partial])
        _run_step(scheduler, 1)
        self.assertEqual(decode.generation_idx, 1)
        self.assertEqual(partial.num_computed_tokens, 512)
        return scheduler, decode, partial

    def test_throttled_step_defers_running_and_waiting_prefills_but_serves_decode(self):
        scheduler, decode, partial = self.decoding_and_partial()
        waiting = _request(2, 4096)
        scheduler.add_requests([waiting])
        output = scheduler.schedule_output(throttle_prefills=True)
        self.assertTrue(scheduler.defer_prefills)
        self.assertEqual(output.running, [decode])
        self.assertEqual(decode.scheduled_tokens, 1)
        self.assertEqual(partial.num_computed_tokens, 512)
        self.assertEqual(scheduler.waiting, [waiting])
        self.assertEqual(scheduler.stats["deferred_running_prefills"], 1)
        self.assertEqual(scheduler.stats["deferred_waiting_prefills"], 1)

    def test_no_decode_means_no_deferral_even_on_throttled_step(self):
        scheduler = _scheduler(1024, long_prefill_token_threshold=512)
        a, b = _request(0, 4096), _request(1, 4096)
        scheduler.add_requests([a, b])
        _run_step(scheduler, 1)
        output = scheduler.schedule_output(throttle_prefills=True)
        self.assertFalse(scheduler.defer_prefills)
        self.assertEqual([r.scheduled_tokens for r in output.running], [512, 512])

    def test_capacity_bound_release_disables_next_throttle(self):
        scheduler, decode, partial = self.decoding_and_partial(max_seqs=2)
        waiting = _request(2, 4096)
        scheduler.add_requests([waiting])
        released = scheduler.schedule_output(throttle_prefills=False)
        self.assertEqual([r.scheduled_tokens for r in released.running], [1, 512])
        self.assertTrue(scheduler.prefill_capacity_bound)
        apply(scheduler, released, 2)
        throttled = scheduler.schedule_output(throttle_prefills=True)
        self.assertFalse(scheduler.defer_prefills)
        self.assertEqual([r.scheduled_tokens for r in throttled.running], [1, 512])

    def test_budget_exhaustion_and_max_sequences_keep_remaining_requests_waiting(self):
        for budget, limit, expected in (
            (1000, None, [512, 488]),
            (4096, 2, [512, 512]),
        ):
            scheduler = _scheduler(
                budget, long_prefill_token_threshold=512, max_parallem_sum=limit
            )
            requests = [_request(i, 4096) for i in range(3)]
            scheduler.add_requests(requests)
            output = scheduler.schedule_output()
            self.assertEqual([r.scheduled_tokens for r in output.running], expected)
            self.assertEqual(scheduler.waiting, [requests[2]])
            self.assertTrue(scheduler.prefill_capacity_bound)

    def test_threshold_must_be_nonnegative(self):
        with self.assertRaises(ConfigurationError):
            _scheduler(1024, long_prefill_token_threshold=-1)

    def test_running_insertion_order_survives_resuming_an_older_request(self):
        scheduler = _scheduler(4096)
        younger = _request(1, 32, decode_len=10, arrived_at=1)
        resumed_older = _request(0, 32, decode_len=10, arrived_at=0)
        # A resumed request is appended behind an already running request.
        scheduler.add_requests([younger, resumed_older])
        _run_step(scheduler, 2)
        self.assertEqual(scheduler.schedule_output().running, [younger, resumed_older])

    def test_dp_worker_uses_shared_counter_but_non_dp_does_not_throttle(self):
        env = simpy.Environment()
        e = engine(env)
        w = e.workers[0]
        w.prefill_schedule_interval = 8
        w.scheduler.long_prefill_token_threshold = 4
        decode = _request(0, 1, decode_len=10)
        partial = _request(1, 64, decode_len=10)
        w.scheduler.max_num_batched_tokens = 16
        w.scheduler.add_requests([decode, partial])
        _run_step(w.scheduler, 1)
        for counter, expected in ((1, True), (8, False), (0, False)):
            w.dp_step_counter = counter
            w.scheduler.prefill_capacity_bound = False
            w.prepare_step()
            self.assertEqual(w.scheduler.defer_prefills, expected)
        w.parallel_config.data_parallel_size = 1
        w.dp_step_counter = 1
        w.scheduler.prefill_capacity_bound = False
        w.prepare_step()
        self.assertFalse(w.scheduler.defer_prefills)


if __name__ == "__main__":
    unittest.main()
