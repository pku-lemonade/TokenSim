"""Load conservation when repeated routing histograms are accumulated lazily."""

import unittest

from TokenSim.moe.stats import MoEStats


class MoEStatsTest(unittest.TestCase):
    @staticmethod
    def record(stats, histogram, ranks):
        stats.record_step(
            histogram=histogram,
            per_rank_load=ranks,
            compute_latency=0.25,
            straggler_latency=0,
            all2all_event_count=1,
        )

    def test_repeated_and_distinct_histograms_conserve_loads(self):
        stats = MoEStats(effective_moe_config={"num_experts": 4})
        histogram, ranks = {0: 2, 3: 6}, {0: 2, 1: 6}
        for _ in range(100):
            self.record(stats, histogram, ranks)
        self.record(stats, {0: 1, 2: 3}, {0: 4})
        self.assertEqual(stats.per_expert_load, {0: 201, 2: 3, 3: 600})
        self.assertEqual(stats.per_rank_expert_load, {0: 204, 1: 600})
        self.assertEqual(stats.routed_token_count, 804)
        self.assertEqual(stats.mean_expert_load, 201)
        self.assertEqual(stats.moe_all2all_event_count, 101)
        self.assertEqual(histogram, {0: 2, 3: 6})
        self.assertEqual(ranks, {0: 2, 1: 6})

    def test_reads_and_aggregation_include_pending_steps_once(self):
        left = MoEStats(per_expert_load={0: 5}, per_rank_expert_load={0: 5})
        right = MoEStats()
        histogram, ranks = {1: 3}, {1: 3}
        self.record(left, histogram, ranks)
        self.assertEqual(left.per_expert_load, {0: 5, 1: 3})
        self.record(left, histogram, ranks)
        self.record(right, histogram, ranks)
        combined = left.aggregate(right)
        self.assertEqual(combined.per_expert_load, {0: 5, 1: 9})
        self.assertEqual(combined.per_rank_expert_load, {0: 5, 1: 9})
        self.assertEqual(combined.routed_token_count, 9)
        first = combined.as_dict()
        self.assertEqual(combined.as_dict(), first)
        self.record(left, histogram, ranks)
        self.assertEqual(left.per_expert_load, {0: 5, 1: 9})

    def test_bounded_pending_histograms_preserve_all_steps(self):
        stats = MoEStats()
        for expert in range(4100):
            self.record(stats, {expert: 1}, {0: 1})
        self.assertEqual(stats.routed_token_count, 4100)
        self.assertEqual(stats.per_rank_expert_load, {0: 4100})
        self.assertEqual(stats.per_expert_load, dict.fromkeys(range(4100), 1))
