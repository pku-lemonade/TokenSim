"""Ownership, traffic, capacity and operator-lifetime checks for mixed decode."""

from dataclasses import replace
import unittest

from TokenSim.errors import ConfigurationError
from scripts.lpx.fidelity import load
from scripts.lpx.mfu_metrics import resource_demands
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.sqd_bounds import sliding_kv_bytes
from scripts.lpx.sqd_memory import CoreMemory, gpu_memory
from scripts.lpx.sqd_probe import pool_bound
from scripts.lpx.sqd_program import GlobalCoreProgram, core_messages, cross_transfer
from scripts.lpx.streaming import StreamingBuilder, duration
from scripts.run_lpx_article_sweep import article_config


class TinyProgram:
    append = StreamingBuilder.append

    def __init__(self):
        self.nodes = []


class SQDCrossTrafficTest(unittest.TestCase):
    def test_query_output_and_canonical_kv_byte_conservation(self):
        incoming, outgoing = core_messages(
            (0, 1, 2, 3), (0, 1), heads=8, kv_heads=2, dim=16, batch=2
        )
        self.assertEqual(incoming, ((0, 0, 256), (1, 0, 128), (2, 1, 256), (3, 1, 128)))
        self.assertEqual(sum(n for _, _, n in outgoing), 8 * 16 * 2 * 2)

    def test_gpu_head_replication_appends_each_physical_kv_copy(self):
        incoming, outgoing = core_messages(
            (0, 1, 2, 3), (0, 1, 2, 3), heads=8, kv_heads=2, dim=16, batch=2
        )
        self.assertEqual(
            sum(n for _, _, n in incoming), 8 * 16 * 2 * 2 + 4 * 2 * 16 * 2 * 2
        )
        self.assertEqual(sum(n for _, _, n in outgoing), 8 * 16 * 2 * 2)

    def test_shared_uplink_serializes_distinct_endpoint_messages(self):
        p = TinyProgram()
        for _ in range(2):
            sent, arrived = cross_transfer(
                p,
                (),
                ((0, 0, 80), (1, 1, 80)),
                direction="to_gpu",
                bandwidth=80,
                latency_s=3,
            )
            self.assertEqual(p.nodes[sent].duration, 2)
            self.assertEqual(p.nodes[arrived].delay, 3)
        self.assertEqual(duration(p.nodes), 7)
        self.assertIn("sqd/lp-tray:0:tx", p.nodes[0].resources)
        self.assertIn("sqd/gpu-node:0:rx", p.nodes[0].resources)

    def test_invalid_head_partitions_and_duplicate_endpoints_fail(self):
        for lp, gpu in (((0, 0), (0, 1)), ((0, 1, 2), (0, 1)), ((True, 1), (0, 1))):
            with self.subTest(lp=lp), self.assertRaises(ConfigurationError):
                core_messages(lp, gpu, heads=8, kv_heads=2, dim=16)


class SQDMemoryProgramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = article_config(load("experiments/lpx/fidelity-study.json"))
        config["compute"]["attention_mfu"] = 0.5
        cls.memory = CoreMemory(config, 4)
        cls.builder = GlobalCoreProgram(
            cls.memory.hardware,
            cls.memory.model,
            lp_tp=4,
            counts=(1,) * 60,
            gpu_device="b200_sxm",
            gpu_tp=4,
            replicas=2,
        )
        cls.profiles = [cls.builder.build(i) for i in range(2)]

    def test_native_cache_projection_keeps_every_window_and_global_weight(self):
        m = self.memory
        self.assertEqual(
            sum(sum(row) for row in m.footprint.kv), sliding_kv_bytes(m.model, 102000)
        )
        for layer in range(60):
            self.assertTrue(all(n > 0 for n in m.footprint.weights[layer]))
            if not m.model.attention_window(layer):
                self.assertEqual(m.footprint.kv[layer], (0,) * 4)
                self.assertEqual(m.footprint.blocks[layer], (0,) * 4)
        self.assertEqual(m.footprint.kv_watermark, 0.01)

    def test_projected_slots_and_capacity_increase_with_residency(self):
        one, two = self.memory.select(60, 1), self.memory.select(60, 2)
        self.assertIsNotNone(one)
        self.assertEqual(one["layer_counts"], (1,) * 60)
        self.assertLess(
            two["minimum_lp30_headroom_bytes"], one["minimum_lp30_headroom_bytes"]
        )
        self.assertGreater(
            two["boundary_pool"][0]["request_slots_bytes"],
            one["boundary_pool"][0]["request_slots_bytes"],
        )
        self.assertIsNone(self.memory.select(60, 1_000_000))

    def test_cached_capacity_does_not_skip_type_validation(self):
        self.assertIsNotNone(self.memory.select(60, 1))
        for pp, resident in ((60, True), (60, 1.0), (60.0, 1)):
            with (
                self.subTest(pp=pp, resident=resident),
                self.assertRaises(ConfigurationError),
            ):
                self.memory.select(pp, resident)

    def test_gpu_capacity_uses_cohort_affinity_not_even_request_division(self):
        m = self.memory
        ledger = gpu_memory(
            m.hardware,
            m.model,
            device="b200_sxm",
            tp=4,
            replicas=2,
            resident=12,
            batch=4,
        )
        # Three cohorts map 2:1 to replicas, so the busiest replica owns 8 requests.
        self.assertTrue(all(row["requests_per_busiest_replica"] == 8 for row in ledger))
        self.assertEqual(ledger[0]["kv_bytes_per_rank"], 8 * 102000 * 512 * 2)
        with self.assertRaises(ConfigurationError):
            gpu_memory(
                m.hardware,
                replace(m.model, kv_cache_dtype="bf16"),
                device="b200_sxm",
                tp=4,
                replicas=2,
                resident=12,
            )
        with self.assertRaises(ConfigurationError):
            gpu_memory(
                m.hardware,
                m.model,
                device="b200_sxm",
                tp=4,
                replicas=2,
                resident=12,
                batch=5,
            )
        with self.assertRaises(ConfigurationError):
            gpu_memory(
                m.hardware,
                m.model,
                device="b200_sxm",
                tp=4,
                replicas=1,
                resident=1_000_000,
            )

    def test_native_operator_ownership_preserves_all_projections_and_ffns(self):
        p = self.profiles[0]
        operators = p.metadata["operators"]
        self.assertEqual(operators["qkv_projection"], 60 * 4)
        self.assertEqual(operators["output_projection"], 60 * 4)
        self.assertEqual(operators["attention_core"], 50 * 4)
        self.assertEqual(operators["down"], 60 * 4)
        self.assertEqual(operators["post_ffn_norm"], 60 * 4)
        gpu_nodes = [
            node
            for node in p.nodes
            if any(key.startswith("sqd/gpu-worker:") for key in node.resources)
        ]
        self.assertEqual(len(gpu_nodes), 10)
        self.assertAlmostEqual(
            sum(n.duration for n in gpu_nodes), self.builder.gpu_cost["gpu_path_s"]
        )
        other_gpu = {
            key
            for node in self.profiles[1].nodes
            for key in node.resources
            if key.startswith("sqd/gpu-worker:")
        }
        self.assertTrue(all(not set(node.resources) & other_gpu for node in gpu_nodes))

    def test_lp30_global_layer_service_matches_native_cost_minus_core(self):
        chip, layer = 5 * 4, 5
        expected = sum(
            part["latency_us"]
            for part in self.builder.parts(layer, "attention")
            if part["name"] != "attention_core"
        )
        for stage, operation in (
            ("layout", "post_attention_norm"),
            ("dense_ffn", ""),
            ("residual", ""),
        ):
            expected += sum(
                part["latency_us"]
                for part in self.builder.parts(layer, stage, operation)
            )
        actual = resource_demands(self.profiles[0].nodes)[f"mfu-search/worker:{chip}"]
        self.assertAlmostEqual(actual, expected * 1e-6)

    def test_lifetime_bound_combines_disjoint_gpu_pools_with_shared_lp30(self):
        result = pool_bound(
            self.profiles, self.builder.gpu_cost, resident=16, batch=1, replicas=2
        )
        self.assertGreater(result["aggregate_tps"], 0)
        self.assertLessEqual(result["aggregate_tps"], result["gpu_pool_limited_tps"])
        self.assertLessEqual(
            result["aggregate_tps"], result["lp30_resource_limited_tps"]
        )
        # Reject accidental changes in a supposedly shared LP30 program.
        altered = list(self.profiles[1].nodes)
        altered[0] = replace(altered[0], duration=1, occupancy=1)
        bad = NativeProfile(tuple(altered), self.profiles[1].metadata)
        with self.assertRaises(ConfigurationError):
            pool_bound(
                (self.profiles[0], bad),
                self.builder.gpu_cost,
                resident=16,
                batch=1,
                replicas=2,
            )


if __name__ == "__main__":
    unittest.main()
