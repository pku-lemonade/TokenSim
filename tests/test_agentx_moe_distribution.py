"""Regression cases for whole-step routing and fixed measured MoE workload."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

from TokenSim.config.config import ParallelConfig
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request
from TokenSim.moe.routing import ExpertRouting
from TokenSim.moe.stats import MoEStats
from TokenSim.operator_data.manifest import WorkloadGrid, build_manifest
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)


def decode(identifier, tokens=1, **kwargs):
    r = Request(identifier, 256, 1024, 256, **kwargs)
    r.generation_idx = 1
    r.num_computed_tokens = 256
    r.scheduled_tokens = tokens
    return r


class AgentXMoEDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")
        cls.device = cls.hardware.device("b200")
        cls.parallel = ParallelConfig(data_parallel_size=8, enable_expert_parallel=True)
        cls.placement = {
            rank: list(range(rank * 48, (rank + 1) * 48)) for rank in range(8)
        }

    def routing(self, seed=0):
        return ExpertRouting(self.model.moe, seed=seed, rank_to_experts=self.placement)

    def backend(self, model=None, *, distributions=("balanced", "power_law_1.01")):
        model = model or self.model
        rows = []
        for distribution in distributions:
            for dtype in ("fp8_block", "mxfp4"):
                for tokens in (8, 16):
                    rows.append(
                        {
                            "dtype": dtype,
                            "distribution": distribution,
                            "num_tokens": tokens,
                            "hidden_size": 7168,
                            "inter_size": 3072,
                            "top_k": 6,
                            "num_experts": 384,
                            "tp_size": 1,
                            "ep_size": 8,
                            "latency_us": (
                                10
                                if dtype == "mxfp4" and distribution == "power_law_1.01"
                                else 100
                            )
                            * tokens,
                            "source_id": "measured:test-moe",
                        }
                    )
        package = OperatorDataPackage.from_rows(
            PackageMeta(
                "test",
                "b200_sxm",
                "vllm",
                sources={
                    "measured:test-moe": SourceRecord(
                        "measured:test-moe", "A", "measured"
                    )
                },
            ),
            {"moe": rows},
        )
        return OperatorTableLatencyBackend(
            device=self.device,
            model=model,
            parallel_config=self.parallel,
            package=package,
            fallback="table_first",
        )

    def test_49_independent_decode_requests_spread_remainder_across_ep_ranks(self):
        histogram = self.routing().histogram_for_step([decode(i) for i in range(49)])
        self.assertEqual(sum(histogram.values()), 49 * 6)
        loads = [
            sum(histogram.get(e, 0) for e in experts)
            for experts in self.placement.values()
        ]
        self.assertEqual(sorted(loads), [36, 36, 37, 37, 37, 37, 37, 37])
        self.assertEqual(max(histogram.values()), 1)

    def test_request_partition_does_not_change_synthetic_step_histogram(self):
        a = self.routing(seed=13).histogram_for_step([decode(i) for i in range(49)])
        b = self.routing(seed=13).histogram_for_step([decode(0, tokens=49)])
        self.assertEqual(a, b)

    def test_remainder_rotates_and_long_prefill_is_route_conserving(self):
        routing = self.routing(seed=7)
        total = Counter()
        for i in range(64):
            total.update(routing.histogram_for_step([decode(i)]))
        self.assertEqual(set(total.values()), {1})
        self.assertEqual(len(total), 384)
        r = Request(100, 1_000_000, 1, 256)
        r.scheduled_tokens = 8192
        histogram = routing.histogram_for_step([r])
        self.assertEqual(sum(histogram.values()), 8192 * 6)
        self.assertLessEqual(max(histogram.values()) - min(histogram.values()), 1)

    def test_explicit_histograms_survive_aggregation_and_prefill_decode_transition(
        self,
    ):
        routing = self.routing()
        explicit = Request(
            0,
            256,
            2,
            256,
            prefill_expert_histogram={383: 9},
            decode_expert_histogram={382: 6},
        )
        histogram = routing.histogram_for_step([explicit, decode(1)])
        self.assertEqual(histogram[383], 9)
        self.assertEqual(sum(histogram.values()), 15)
        explicit.generation_idx = 1
        explicit.num_computed_tokens = 256
        explicit.scheduled_tokens = 1
        self.assertEqual(routing.histogram_for_step([explicit]), {382: 6})

    def test_native_cost_uses_mxfp4_power_law_once_independent_of_histogram(self):
        balanced, hot = self.backend(), self.backend()
        a = decode(0)
        b = decode(0, decode_expert_histogram={0: 6})
        with patch.object(hot.lookup, "lookup", wraps=hot.lookup.lookup) as queries:
            hot_time = hot._moe_layers(1, [b])
        balanced_time = balanced._moe_layers(1, [a])
        self.assertEqual(queries.call_count, 1)
        self.assertEqual(queries.call_args.args[1]["dtype"], "mxfp4")
        self.assertEqual(queries.call_args.args[1]["distribution"], "power_law_1.01")
        self.assertEqual(hot_time, balanced_time)
        self.assertEqual(hot_time, 80 * 61)
        self.assertEqual(hot.moe_stats.moe_straggler_latency, 0)
        self.assertEqual(hot.moe_stats.per_rank_expert_load, {0: 6})
        # A cached power-law estimate must retain its distribution metadata.
        self.assertEqual(hot._moe_layers(1, [b]), hot_time)
        self.assertEqual(hot.moe_stats.moe_straggler_latency, 0)

    def test_generic_power_law_fallback_does_not_add_distribution_twice(self):
        backend = self.backend(
            replace(self.model, architecture="transformer", components=None),
            distributions=("power_law_1.01",),
        )
        r = decode(0, decode_expert_histogram={0: 6})
        self.assertEqual(backend._moe_layers(1, [r]),
                         backend._moe_layers(1, [decode(1)]))
        self.assertEqual(backend.moe_stats.moe_straggler_latency, 0)
        self.assertEqual(backend._moe_layers(1, [r]),
                         backend._moe_layers(1, [decode(1)]))

    def test_unloaded_experts_are_in_mean_load_denominator(self):
        stats = MoEStats(
            effective_moe_config={"num_experts": 384}, per_expert_load={0: 6}
        )
        self.assertEqual(stats.mean_expert_load, 6 / 384)
        self.assertEqual(stats.expert_load_imbalance_ratio, 384)

    def test_manifest_requests_native_expert_dtype_and_fixed_distribution(self):
        manifest = build_manifest(
            "test",
            [self.model],
            grid=WorkloadGrid(batch_sizes=(1,), prefill_tokens=(512,), tp_sizes=(1,)),
        )
        self.assertTrue(manifest.keys["moe"])
        self.assertTrue(
            all(
                k["dtype"] == "mxfp4" and k["distribution"] == "power_law_1.01"
                for k in manifest.keys["moe"]
            )
        )


if __name__ == "__main__":
    unittest.main()
