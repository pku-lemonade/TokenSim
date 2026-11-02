"""Independent small partition oracles and analytic replay limits."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.mfu_capacity import ResidentCapacity
from scripts.lpx.mfu_metrics import (
    critical_path_seconds,
    throughput_bounds,
    useful_matrix_flops,
)
from scripts.lpx.mfu_replay import replay_cohorts
from scripts.lpx.mfu_probe import MFUCase
from scripts.lpx.native_config import configuration
from scripts.lpx.native_search import memory_report
from scripts.lpx.partition import PartitionFootprint
from scripts.lpx.replay_engine import ReplayNode
from scripts.run_lpx_context_parallel import layout_for


class ResidentCapacityTest(unittest.TestCase):
    def test_prefix_loads_and_all_stage_counts_match_exhaustive_partitions(self):
        rng = random.Random(42)
        for _ in range(12):
            blocks = tuple((rng.randint(1, 8), rng.randint(1, 8)) for _ in range(5))
            footprint = PartitionFootprint(
                weights=tuple((rng.randint(0, 12), rng.randint(0, 12)) for _ in blocks),
                kv=tuple(
                    tuple(block * rng.randint(1, 4) for block in row) for row in blocks
                ),
                blocks=blocks,
                embedding=(6, 8),
                buffers=(2, 3),
                capacities=(70, 75),
                tied_embeddings=rng.choice((True, False)),
                kv_watermark=rng.choice((0, 0.01, 0.2)),
            )
            screen = ResidentCapacity(footprint)
            for resident in (1, 2, 4):
                scaled = replace(
                    footprint,
                    kv=tuple(
                        tuple(value * resident for value in row) for row in footprint.kv
                    ),
                )
                loads = screen.segment_loads(resident)
                for index, (start, end) in enumerate(zip(screen.starts, screen.ends)):
                    self.assertEqual(
                        tuple(loads[index]), scaled.stage_bytes(int(start), int(end))
                    )
                feasible = []
                for stages in range(1, 6):
                    for cuts in combinations(range(1, 5), stages - 1):
                        bounds = (0, *cuts, 5)
                        if all(
                            all(
                                load <= cap
                                for load, cap in zip(
                                    scaled.stage_bytes(a, b), footprint.capacities
                                )
                            )
                            for a, b in zip(bounds, bounds[1:])
                        ):
                            feasible.append(stages)
                            break
                self.assertEqual(screen.feasible_stages(resident), tuple(feasible))

    def test_frontier_retains_a_request_limit_and_rejects_invalid_counts(self):
        footprint = PartitionFootprint(((2,),), ((4,),), ((2,),), (0,), (0,), (14,))
        screen = ResidentCapacity(footprint)
        self.assertEqual(screen.frontier(2)["maximum_resident_by_pp"], {1: 2})
        self.assertTrue(screen.frontier(2)["request_limit_reached"])
        self.assertEqual(screen.frontier(5)["maximum_resident_by_pp"], {1: 3})
        self.assertFalse(screen.frontier(5)["request_limit_reached"])
        for count in (0, True, -1):
            with self.assertRaises(ConfigurationError):
                screen.feasible_stages(count)


class MFUMetricsTest(unittest.TestCase):
    def test_resident_memory_report_extends_the_one_request_ledger(self):
        hardware = HardwareContext.load(Path(__file__).resolve().parents[1] / "data")
        model = replace(
            hardware.model("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
        )
        runtime = configuration(
            hardware,
            model,
            layout_for(8, 8, 32),
            namespace="mfu-memory",
            colocated_dense_reuse=True,
        )
        one = memory_report(runtime, 102000)
        two = memory_report(runtime, 102000, 2)
        self.assertNotIn("resident_requests", one)
        self.assertEqual(two["resident_requests"], 2)
        self.assertGreater(
            two["peak_rank"]["total_bytes"], one["peak_rank"]["total_bytes"]
        )
        self.assertLess(two["minimum_headroom_bytes"], one["minimum_headroom_bytes"])
        with self.assertRaises(ConfigurationError):
            memory_report(runtime, 102000, 0)
        with self.assertRaises(ConfigurationError):
            MFUCase(16, 3, 8)
        with self.assertRaises(ConfigurationError):
            MFUCase(16, 4, 8, layer_counts=(1,) * 8)

    def test_useful_flops_respect_per_layer_geometry_and_bounded_context(self):
        model = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        ).model("gemma-4-31b-text")
        short = useful_matrix_flops(model, 100)
        self.assertEqual(
            short["linear"],
            useful_matrix_flops(replace(model, qk_norm=False), 100)["linear"],
        )
        long = useful_matrix_flops(model, 200)
        self.assertEqual(short["linear"], long["linear"])
        self.assertEqual(long["attention"], 2 * short["attention"])
        before = useful_matrix_flops(model, 2000)
        after = useful_matrix_flops(model, 2001)
        globals_only = sum(
            4 * model.num_attention_heads * model.attention_geometry(i).head_dim
            for i in range(model.num_layers)
            if model.attention_window(i) == 0
        )
        self.assertEqual(after["attention"] - before["attention"], globals_only)

    def test_bounds_relax_contention_and_accept_forward_index_references(self):
        nodes = (
            ReplayNode((2,), ("unit",), "unit", 0, 2, 2),
            ReplayNode((), ("unit",), "unit", 0, 3, 3),
            ReplayNode((1,), (), None, 1, 0, 0),
        )
        self.assertEqual(critical_path_seconds(nodes), 6)
        bound = throughput_bounds(nodes, resident_requests=10, batch=1)
        self.assertEqual(bound["aggregate_tps"], 1 / 5)
        self.assertEqual(bound["busiest_resource"], "unit")
        with self.assertRaises(ConfigurationError):
            critical_path_seconds((replace(nodes[0], predecessors=(0,)),))
        with self.assertRaises(ConfigurationError):
            critical_path_seconds((replace(nodes[0], predecessors=(-1,)),))
        with self.assertRaises(ConfigurationError):
            throughput_bounds(nodes, resident_requests=3, batch=2)


class MFUReplayTest(unittest.TestCase):
    def test_client_feedback_retains_its_assigned_program(self):
        fast = SimpleNamespace(nodes=(ReplayNode((), ("fast",), "fast", 0, 1, 1),))
        slow = SimpleNamespace(nodes=(ReplayNode((), ("slow",), "slow", 0, 2, 2),))
        result = replay_cohorts(
            fast,
            additional_profiles=(slow,),
            resident_requests=2,
            warmup_rounds=1,
            measured_rounds=3,
        )
        self.assertAlmostEqual(result["aggregate_tps"], 1.5)
        self.assertEqual(result["program_assignment"], {0: 0, 1: 1})
        unequal = replay_cohorts(
            fast,
            additional_profiles=(slow,),
            resident_requests=3,
            warmup_rounds=1,
            measured_rounds=2,
            initial_spacing_s=1,
            issue_interval_s=1,
        )
        self.assertEqual(unequal["program_assignment"], {0: 0, 1: 1, 2: 0})
        self.assertEqual(unequal["per_cohort_completions"], {0: 2, 1: 2, 2: 2})
        self.assertEqual(unequal["aggregate_tps"], 1)

    def test_two_stage_pipeline_and_serial_feedback_match_analytic_rates(self):
        profile = SimpleNamespace(
            nodes=(
                ReplayNode((), ("stage0",), "stage0", 0, 1, 1),
                ReplayNode((0,), ("stage1",), "stage1", 0, 1, 1),
            )
        )
        serial = replay_cohorts(profile, resident_requests=1)
        pipeline = replay_cohorts(profile, resident_requests=2)
        self.assertAlmostEqual(serial["aggregate_tps"], 0.5)
        self.assertAlmostEqual(pipeline["aggregate_tps"], 1)
        self.assertEqual(
            sum(pipeline["per_cohort_completions"].values()),
            pipeline["measured_completions"],
        )

    def test_batch_token_credit_and_issue_pacing_preserve_dependencies(self):
        profile = SimpleNamespace(nodes=(ReplayNode((), ("unit",), "unit", 0, 1, 1),))
        result = replay_cohorts(
            profile,
            resident_requests=4,
            batch=2,
            initial_spacing_s=2,
            issue_interval_s=2,
        )
        self.assertAlmostEqual(result["aggregate_tps"], 1)
        self.assertAlmostEqual(result["mean_per_request_tps"], 0.25)
        self.assertEqual(result["measured_tokens"], 2 * result["measured_completions"])
        with self.assertRaises(ConfigurationError):
            replay_cohorts(profile, resident_requests=3, batch=2)


if __name__ == "__main__":
    unittest.main()
