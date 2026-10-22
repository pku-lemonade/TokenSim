"""Independent exhaustive partition oracle and native placement invariants."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
import unittest

import simpy

from TokenSim.af.runtime.admission import ResidentKV, rank_kv_bytes
from TokenSim.af.runtime.topology import AFCacheConfig
from TokenSim.af.serving.handoff import plan_handoff
from TokenSim.af.serving.service import AFService
from TokenSim.config.af.domains import DomainParallelConfig
from TokenSim.config.cache_config import stage_layer_range
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration
from scripts.lpx.native_search import build_runtime, memory_report, streaming_settings
from scripts.lpx.partition import (
    PartitionFootprint,
    capacity_partition,
    native_footprint,
    partition_solutions,
)
from scripts.lpx.replay_profiles import capture_profile


class PartitionOracleTest(unittest.TestCase):
    def test_dp_matches_exhaustive_rank_and_block_aware_partitions(self):
        weights = ((10, 12), (14, 8), (7, 18), (16, 10), (8, 9), (11, 7))
        kv = ((3, 2), (15, 10), (2, 2), (4, 3), (13, 16), (4, 2))
        blocks = ((4, 2),) * 6
        footprint = PartitionFootprint(weights, kv, blocks, (15, 12), (2, 3), (65, 60))
        actual = partition_solutions(footprint, 6)
        for stages in range(1, 7):
            expected = []
            for cuts in combinations(range(1, 6), stages - 1):
                bounds, peak, valid = (0, *cuts, 6), 0, True
                for start, end in zip(bounds, bounds[1:]):
                    for rank, capacity in enumerate((65, 60)):
                        resident = sum(kv[i][rank] for i in range(start, end))
                        block = (end - start) * (4, 2)[rank]
                        reserved = -(-resident // block) * block
                        weight_bytes = sum(weights[i][rank] for i in range(start, end))
                        boundary = (15, 12)[rank] if start == 0 or end == 6 else 0
                        load = weight_bytes + reserved + (2, 3)[rank] + boundary
                        valid &= load <= capacity
                        peak = max(peak, load / capacity)
                if valid:
                    expected.append(peak)
            self.assertEqual(stages in actual, bool(expected))
            if expected:
                self.assertAlmostEqual(
                    actual[stages]["peak_capacity_fraction"], min(expected)
                )
                self.assertEqual(sum(actual[stages]["layer_counts"]), 6)

    def test_explicit_partitions_are_positive_complete_and_strict(self):
        counts = (1, 3, 2)
        self.assertEqual(
            [list(stage_layer_range(6, 3, i, counts)) for i in range(3)],
            [[0], [1, 2, 3], [4, 5]],
        )
        for bad in ((2, 2), (1, 0, 5), (1, True, 4), (1, 2, 2)):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                stage_layer_range(6, 3, 0, bad)
        for bad in ((2,), (0, 2), (True, 1)):
            with (
                self.subTest(bad=bad),
                self.assertRaises((ValueError, ConfigurationError)),
            ):
                DomainParallelConfig(
                    pipeline_parallel_size=2, pipeline_layer_counts=bad
                )
        with self.assertRaises(ConfigurationError):
            PartitionFootprint(((1,),), ((1, 1),), ((1,),), (0,), (0,), (10,))
        tied = PartitionFootprint(((1,),), ((0,),), ((0,),), (4,), (0,), (10,))
        self.assertEqual(tied.stage_bytes(0, 1), (5,))
        self.assertEqual(replace(tied, tied_embeddings=False).stage_bytes(0, 1), (9,))

    def test_watermark_reserves_compiled_free_blocks_beyond_resident_bytes(self):
        footprint = PartitionFootprint(
            ((10,),), ((19,),), ((4,),), (0,), (0,), (34,), kv_watermark=0.2
        )
        self.assertEqual(footprint.stage_bytes(0, 1), (34,))
        self.assertIn(1, partition_solutions(footprint, 1))
        self.assertFalse(partition_solutions(replace(footprint, capacities=(33,)), 1))
        for watermark in (True, -0.1, 1, float("nan")):
            with (
                self.subTest(watermark=watermark),
                self.assertRaises(ConfigurationError),
            ):
                replace(footprint, kv_watermark=watermark)
        for watermark in (0, 0.01, 0.2, 0.9):
            for required in (1, 2, 5, 99, 100, 101):
                item = replace(footprint, kv=((4 * required,),), kv_watermark=watermark)
                expected_slots = next(
                    blocks
                    for blocks in range(required, 2000)
                    if blocks - int(watermark * blocks) >= required
                )
                self.assertEqual(item.stage_bytes(0, 1), (10 + expected_slots * 4,))


class NativePartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = replace(
            cls.hardware.model("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
        )
        cls.layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=8,
            f_tp=8,
            a_pp=11,
            f_pp=11,
            f_ep=1,
        )
        cls.partition = capacity_partition(
            cls.hardware, cls.model, cls.layout, 102000, {}
        )

    def runtime(self):
        return build_runtime(
            self.hardware,
            self.model,
            self.layout,
            layer_counts=self.partition["layer_counts"],
        )

    def test_native_memory_matches_dp_and_every_layer_has_exact_physical_owners(self):
        runtime = self.runtime()
        report = memory_report(runtime, 102000)
        self.assertGreater(report["minimum_headroom_bytes"], 0)
        memory = ResidentKV(runtime.plan)
        memory.admit(
            1, "search-a-domain", "a0", maximum_context=102000, current_context=0
        )
        footprint = native_footprint(runtime, 102000)
        boundaries = self.partition["layer_counts"]
        for layer in range(60):
            expected_stage = next(
                i
                for i in range(11)
                if layer in stage_layer_range(60, 11, i, boundaries)
            )
            self.assertEqual(
                runtime.plan.pipeline_stage("search-a-domain", layer), expected_stage
            )
            a = runtime.plan.bindings_for_layer("search-a-domain", "a0", layer)
            f = runtime.plan.component_execution(
                "search-f-domain", layer, "dense_ffn", 0
            )
            self.assertEqual(
                {b.physical_identity for b in a}, {b.physical_identity for b in f}
            )
            self.assertEqual(len(a), 8)
        for binding in runtime.plan.rank_bindings:
            if binding.capability != "af_attention":
                continue
            stage = binding.pp_rank
            rank = binding.tp_rank
            state = runtime.plan.physical_state(binding.physical_identity)
            raw = (
                state.model_bytes
                + state.protocol_buffer_bytes
                + memory.reserved[binding.physical_identity]
            )
            reserved = self.partition["stage_rank_bytes"][stage][rank]
            self.assertGreaterEqual(reserved, raw)
            block = binding.kv_partition.bytes_per_block
            slots = (
                reserved - state.model_bytes - state.protocol_buffer_bytes
            ) // block
            required = -(-memory.reserved[binding.physical_identity] // block)
            watermark = footprint.kv_watermark
            self.assertGreaterEqual(slots - int(watermark * slots), required)
            self.assertLess(slots - 1 - int(watermark * (slots - 1)), required)
        self.assertEqual(
            runtime.plan.summary()["domains"][0]["pipeline_layer_counts"],
            list(boundaries),
        )
        replica = runtime.plan.replica("search-a-domain", 0)
        cache_config = AFCacheConfig.from_plan(runtime.plan, replica)
        self.assertEqual(cache_config.num_layers_per_rank, boundaries[0])
        self.assertEqual(sum(map(sum, footprint.kv)), sum(memory.reserved.values()))

    def test_explicit_balanced_counts_preserve_native_graph_and_memory(self):
        layout = replace(self.layout, a_pp=16, f_pp=16)
        counts = tuple(len(stage_layer_range(60, 16, i)) for i in range(16))
        implicit = build_runtime(self.hardware, self.model, layout)
        explicit = build_runtime(self.hardware, self.model, layout, layer_counts=counts)
        self.assertEqual(
            memory_report(implicit, 102000), memory_report(explicit, 102000)
        )
        settings = dict(
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=10001,
            dp_rank=0,
            seed=42,
            sample=0,
            streaming=streaming_settings(layout),
        )
        left = capture_profile(implicit, "partition-test", **settings)
        right = capture_profile(explicit, "partition-test", **settings)
        self.assertEqual(left.nodes, right.nodes)
        self.assertEqual(
            left.metadata["modeled_duration_s"], right.metadata["modeled_duration_s"]
        )

    def test_kv_handoff_preserves_every_layer_across_different_partitions(self):
        source_layout = replace(self.layout, a_pp=16, f_pp=16)
        source = configuration(
            self.hardware,
            self.model,
            source_layout,
            namespace="source",
            colocated_dense_reuse=True,
        )
        target = configuration(
            self.hardware,
            self.model,
            self.layout,
            namespace="target",
            colocated_dense_reuse=True,
            pipeline_layer_counts=self.partition["layer_counts"],
        )
        env = simpy.Environment()
        a, b = AFService(env, source, trace=False), AFService(env, target, trace=False)
        before = a.admit(1, 0, maximum_context=102000, current_context=100000)
        after = b.admit(1, 0, maximum_context=102000, current_context=0)
        copies = plan_handoff(
            a,
            before,
            b,
            after,
            context=100000,
            mode="full",
            cross_network="cross",
            replication_network="replication",
        )
        expected = sum(
            rank_kv_bytes(target.plan, binding, 100000)
            for binding in target.plan.rank_bindings
            if binding.kv_partition
        )
        self.assertEqual(sum(copy.bytes for copy in copies), expected)
        self.assertEqual({copy.layer_id for copy in copies}, set(range(60)))
        a.release(before)
        b.release(after)

    def test_partition_cache_is_scoped_by_model_context_and_reuse(self):
        cache = {}
        capacity_partition(self.hardware, self.model, self.layout, 102000, cache)
        capacity_partition(self.hardware, self.model, self.layout, 1000, cache)
        self.assertEqual(len(cache), 2)
        with self.assertRaises(ConfigurationError):
            build_runtime(
                self.hardware, self.model, self.layout, layer_counts=(1,) * 11
            )

    def test_large_prefill_buffers_respect_native_allocator_watermark(self):
        layout = replace(self.layout, a_tp=16, f_tp=16, a_pp=16, f_pp=16)
        seed = configuration(
            self.hardware,
            self.model,
            layout,
            namespace="workspace",
            micro_batch=64,
            colocated_dense_reuse=True,
        )
        solutions = partition_solutions(native_footprint(seed, 102000), 16)
        self.assertTrue(solutions)
        for stages in sorted(solutions)[:3]:
            runtime = configuration(
                self.hardware,
                self.model,
                replace(layout, a_pp=stages, f_pp=stages),
                namespace="workspace",
                micro_batch=64,
                colocated_dense_reuse=True,
                pipeline_layer_counts=solutions[stages]["layer_counts"],
            )
            self.assertGreaterEqual(
                memory_report(runtime, 102000)["minimum_headroom_bytes"], 0
            )


if __name__ == "__main__":
    unittest.main()
