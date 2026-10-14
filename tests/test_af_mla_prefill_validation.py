"""MLA rejects unsupported partial forward passes before submitting work."""

import unittest

from TokenSim.af.factory import simulate_decode
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import Request
from scripts.deepseek_v3_runtime import configuration


class MLAPrefillValidationTest(unittest.TestCase):
    def test_partial_prefill_does_not_submit_boundaries_or_change_ownership(self):
        runtime = configuration(af_batch_size=2)
        executor, _, _, _ = simulate_decode(runtime, 1, 16)
        for computed, scheduled in ((0, 8), (8, 56)):
            with self.subTest(computed=computed, scheduled=scheduled):
                req = Request(7, 64, 3, 16)
                req.num_computed_tokens = computed
                req.scheduled_tokens = scheduled
                before = len(executor.trace.events)
                with self.assertRaisesRegex(ConfigurationError, "unchunked prefill"):
                    executor.submit(0, [req], "prefill")
                self.assertEqual(len(executor.trace.events), before)
                self.assertEqual(req.af_a_owners, {})

    def test_whole_equal_length_prefill_is_accepted(self):
        runtime = configuration(af_batch_size=2)
        executor, _, _, _ = simulate_decode(runtime, 1, 16)
        requests = [Request(i, 16, 3, 16) for i in (1, 2)]
        for req in requests:
            req.scheduled_tokens = 16
        iteration = executor.submit(0, requests, "prefill")
        self.assertEqual(iteration.blocks[0].token_count, 32)
