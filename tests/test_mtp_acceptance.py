"""Acceptance distributions are reproducible across retries and committed rounds."""

from collections import Counter
import unittest

import simpy

from TokenSim.cli.arguments import build_parser
from TokenSim.cli.configuration import simulation_config
from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.speculative_execution import MTPAcceptanceConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import Request
from TokenSim.llm.paged_tokens import PagedTokenBudgetMixin
from TokenSim.llm.speculative_acceptance import sample_acceptance
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from tests.test_kv_cache import layout
from types import SimpleNamespace
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.reporting.stats import get_latency_stats
from tests.support.hardware import make_device, make_model


class MTPAcceptanceTest(unittest.TestCase):
    def test_results_retain_acceptance_identity_and_execution_counts(self):
        scheduler = self.budget()
        scheduler.stats = {"mtp_verification_rounds": 3, "mtp_emitted_tokens": 7}
        device = make_device()
        worker = SimpleNamespace(
            id=0,
            device=device,
            scheduler=scheduler,
            latency_backend=OperatorTableLatencyBackend(
                device=device, model=make_model(), fallback="analytical_only"
            ),
        )
        result = get_latency_stats(SimpleNamespace(workers=[worker]))
        recorded = result["module_stats"]["speculative"]["acceptance"]["0"]
        self.assertEqual(recorded["sampling_version"], "blake2b_round_v1")
        self.assertEqual(recorded["expected_accepted_tokens"], 2.49)
        self.assertEqual(recorded["mtp_verification_rounds"], 3)
        self.assertEqual(recorded["seed"], 42)

    def test_paged_execution_preserves_rounds_and_token_accounting(self):
        def run():
            env = simpy.Environment()
            cache = SimpleNamespace(
                hybrid_layout=layout(eagle=True),
                num_gpu_blocks=40,
                model="test",
                block_size=16,
            )
            scheduler = LLMPagedAttnScheduler(
                0,
                cache,
                max_num_seqs=1,
                max_num_batched_tokens=4,
                mtp_enabled=True,
                mtp_acceptance={"kind": "conditional", "rates": [0.6]},
            )
            req = Request(0, 1, 101, 16)
            req.arrive(env)
            scheduler.add_requests([req])
            accepted = []
            for _ in range(101):
                running, _ = scheduler.schedule()
                self.assertEqual(running, [req])
                if not req.is_context_build:
                    scheduler.record_mtp_execution(req)
                    accepted.append(req.scheduled_accepted_tokens)
                env.run(until=env.now + 0.001)
                req.advance(env, 0.001, 1)
                scheduler.on_tokens_computed(req)
                scheduler.update(running)
                if req.is_done:
                    break
            self.assertTrue(req.is_done)
            self.assertEqual(req.generation_idx, 101)
            self.assertEqual(req.decode_round, len(accepted))
            self.assertEqual(scheduler.stats["mtp_emitted_tokens"], 100)
            self.assertEqual(scheduler.stats["mtp_accepted_tokens"], sum(accepted))
            self.assertEqual(scheduler.stats["mtp_verified_tokens"], 2 * len(accepted))
            return accepted

        self.assertEqual(run(), run())

    def draws(self, config, mean=1, count=20000, seed=42, request=0):
        return [sample_acceptance(config, mean, seed, request, i) for i in range(count)]

    def test_mean_and_adjacent_rounds(self):
        values = self.draws(MTPAcceptanceConfig(), mean=2.49)
        self.assertEqual(set(values), {2, 3})
        self.assertAlmostEqual(sum(values) / len(values), 2.49, delta=0.015)
        changed = sum(a != b for a, b in zip(values, values[1:]))
        self.assertGreater(changed, len(values) * 0.45)
        self.assertLess(changed, len(values) * 0.55)

    def test_conditional_distribution_stops_at_rejection(self):
        config = MTPAcceptanceConfig("conditional", rates=(0.8, 0.6, 0.5))
        counts = Counter(self.draws(config))
        for accepted, probability in enumerate((0.2, 0.32, 0.24, 0.24), 1):
            self.assertAlmostEqual(counts[accepted] / 20000, probability, delta=0.015)
        self.assertAlmostEqual(config.describe(3, 1)["expected_accepted_tokens"], 2.52)
        for rates, expected in (((0, 1, 1), 1), ((1, 1, 1), 4), ((1, 0, 1), 2)):
            self.assertEqual(
                set(
                    self.draws(
                        MTPAcceptanceConfig("conditional", rates=rates), count=100
                    )
                ),
                {expected},
            )

    def test_empirical_distribution(self):
        config = MTPAcceptanceConfig("empirical", distribution=(0.1, 0.2, 0.3, 0.4))
        counts = Counter(self.draws(config))
        for length, p in enumerate(config.distribution, 1):
            self.assertAlmostEqual(counts[length] / 20000, p, delta=0.015)

    def test_seeds_requests_and_call_order(self):
        config = MTPAcceptanceConfig()
        left = self.draws(config, 2.49, count=100)
        self.assertEqual(left, self.draws(config, 2.49, count=100))
        self.assertNotEqual(left, self.draws(config, 2.49, count=100, seed=43))
        self.assertNotEqual(left, self.draws(config, 2.49, count=100, request=1))
        self.assertEqual(
            left[::-1],
            [sample_acceptance(config, 2.49, 42, 0, i) for i in reversed(range(100))],
        )

    def budget(self, config=None):
        budget = PagedTokenBudgetMixin()
        budget.mtp_acceptance = config or MTPAcceptanceConfig()
        budget.mtp_acceptance_length = 2.49 if config is None else 1
        budget.mtp_enabled, budget.speculative_tokens = True, 3
        budget.mtp_seed, budget.max_model_len = 42, None
        return budget

    def request(self):
        req = Request(0, 16, 100, 16)
        req.generation_idx, req.num_computed_tokens = 1, 16
        return req

    def test_retry_budget_and_clipping_do_not_change_the_draw(self):
        budget, req = self.budget(), self.request()
        expected = budget._decode_output_tokens(req, 100)
        self.assertEqual(budget._step_tokens(req, 3), 0)
        self.assertEqual(req.decode_round, 0)
        self.assertEqual(budget._step_tokens(req, 4), expected)
        self.assertEqual(budget._decode_output_tokens(req, 1), 1)
        self.assertEqual(req.scheduled_accepted_tokens, expected)
        self.assertEqual(budget._decode_output_tokens(req, 100), expected)

    def test_only_executed_decode_advances_round(self):
        env, req = simpy.Environment(), self.request()
        req.arrive(env)
        req.scheduled_tokens, req.scheduled_query_tokens = 2, 4
        req.advance(env, 0, 1)
        self.assertEqual(req.decode_round, 1)
        req.prepare_recompute()
        req.scheduled_tokens = 1
        req.advance(env, 0, 1)
        self.assertEqual(req.decode_round, 1)
        req.cancel_requested = True
        req.advance(env, 0, 1)
        self.assertEqual(req.decode_round, 1)

    def test_legacy_is_available_for_reproduction(self):
        budget = self.budget(MTPAcceptanceConfig("legacy"))
        budget.mtp_acceptance_length, budget.mtp_seed = 2.49, 0
        for request_id, expected in ((0, 3), (2, 2), (17, 3)):
            req = self.request()
            req.id = request_id
            self.assertEqual(budget._decode_output_tokens(req, 100), expected)

    def test_committed_execution_counts_unclipped_and_emitted_tokens(self):
        budget = self.budget(MTPAcceptanceConfig("conditional", rates=(1, 1, 1)))
        req = self.request()
        req.scheduled_tokens = budget._decode_output_tokens(req, 1)
        req.scheduled_query_tokens = 4
        budget.stats = {
            name: 0
            for name in (
                "mtp_verification_rounds",
                "mtp_verified_tokens",
                "mtp_accepted_tokens",
                "mtp_emitted_tokens",
            )
        }
        budget.record_mtp_execution(req)
        self.assertEqual(list(budget.stats.values()), [1, 4, 4, 1])

    def test_configuration_and_cli(self):
        args = build_parser().parse_args(
            [
                "--mtp_num_speculative_tokens",
                "3",
                "--mtp_acceptance",
                '{"kind":"conditional","rates":[0.8,0.6,0.5]}',
            ]
        )
        config = simulation_config(args)
        self.assertEqual(config.mtp_acceptance.rates, (0.8, 0.6, 0.5))
        self.assertIsInstance(config.mtp_acceptance.rates, tuple)
        for value in (
            {"kind": "unknown"},
            {"kind": "conditional", "rates": [1, 1]},
            {"kind": "conditional", "rates": [1, True, 0]},
            {"kind": "empirical", "distribution": [0, 0, 0, 0]},
            {"kind": "empirical", "distribution": [0, 0, 0, float("nan")]},
            {"kind": "mean", "rates": [1, 1, 1]},
            {"unknown": True},
            False,
        ):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                SimulationConfig(mtp_num_speculative_tokens=3, mtp_acceptance=value)
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                mtp_num_speculative_tokens=3,
                mtp_acceptance_length=2,
                mtp_acceptance={"kind": "conditional", "rates": [1, 1, 1]},
            )
