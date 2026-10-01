"""MTP acceptance, target query width and request progress regressions."""

from types import SimpleNamespace
from unittest import TestCase

import simpy

from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler


class MTPProgressTest(TestCase):
    def scheduler(self, acceptance=2.49):
        scheduler = object.__new__(LLMPagedAttnScheduler)
        scheduler.mtp_enabled = True
        scheduler.mtp_acceptance_length = acceptance
        scheduler.mtp_seed = 0
        scheduler.cache_config = SimpleNamespace(
            hybrid_layout=SimpleNamespace(speculative_tokens=3)
        )
        return scheduler

    def test_decode_progress_is_two_or_three_but_query_is_four(self):
        scheduler = self.scheduler()
        values = []
        for request_id in range(256):
            req = Request(request_id, 8, 20, 16)
            req.generation_idx = 1
            values.append(scheduler._decode_output_tokens(req, 19))
            self.assertEqual(scheduler._query_tokens(req, values[-1]), 4)
        self.assertEqual(set(values), {2, 3})
        self.assertGreater(values.count(2), values.count(3))

    def test_final_step_truncates_query_width(self):
        scheduler = self.scheduler()
        req = Request(1, 8, 2, 16)
        req.generation_idx = 1
        accepted = scheduler._decode_output_tokens(req, 1)
        self.assertEqual(accepted, 1)
        self.assertEqual(scheduler._query_tokens(req, accepted), 1)

    def test_request_advance_counts_accepted_tokens_once(self):
        env = simpy.Environment()
        req = Request(1, 8, 8, 16)
        req.arrive(env)
        req.generation_idx = 1
        req.num_computed_tokens = req.context_len
        req.scheduled_tokens = 3
        req.scheduled_query_tokens = 4
        req.advance(env, 0.1, 4)
        self.assertEqual(req.generation_idx, 4)
        self.assertEqual(req.query_tokens, req.step_tokens)


if __name__ == "__main__":
    import unittest

    unittest.main()
