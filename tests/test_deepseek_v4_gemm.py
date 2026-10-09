from dataclasses import replace
import unittest
from unittest.mock import patch

from TokenSim.config.config import ParallelConfig
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.projection import attention_projections
from TokenSim.latency.operator_table import (
    OperatorTableLatencyBackend,
    OperatorEstimate,
)
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.analytical import analytical_model_for
from TokenSim.operator_data.manifest import WorkloadGrid, build_manifest
from TokenSim.operator_data.workload import grouped_gemm_work
from TokenSim.errors import ConfigurationError


class DeepSeekV4GemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")
        cls.device = cls.hardware.device("b200")

    def backend(self, tp=1):
        return OperatorTableLatencyBackend(
            device=self.device,
            model=self.model,
            parallel_config=ParallelConfig(tensor_parallel_size=tp),
            fallback="analytical_only",
        )

    def test_projection_geometry_tp8_and_tp4_partition_only_sharded_weights(self):
        for tp in (1, 4, 8):
            p = {x.name: x for x in attention_projections(self.model, tp)}
            self.assertEqual((p["fused_wqa_wkv"].n, p["fused_wqa_wkv"].k), (2048, 7168))
            self.assertEqual((p["wq_b"].n, p["wq_b"].k), (65536 // tp, 1536))
            self.assertEqual(
                (p["wo_a"].groups, p["wo_a"].n, p["wo_a"].k), (16 // tp, 1024, 4096)
            )
            self.assertEqual((p["wo_b"].n, p["wo_b"].k), (7168, 16384 // tp))
            self.assertEqual(
                sum(x.groups * x.n * x.k for x in p.values()),
                2048 * 7168 + (1536 * 65536 + 16 * 1024 * 4096 + 7168 * 16384) // tp,
            )
        with self.assertRaises(ConfigurationError):
            attention_projections(self.model, 3)

    def test_runtime_queries_native_projections_router_and_head_once_per_site(self):
        b = self.backend()
        calls = []

        def query(table, key):
            calls.append((table, key))
            return OperatorEstimate(1.0, "analytical", "test:shape")

        with patch.object(b, "_query", side_effect=query), patch.object(
            b, "_moe_layers", return_value=0
        ):
            b._compose(tokens=64, requests=7, attention_us=0, requests_list=[])
        matrices = [(t, k) for t, k in calls if t in ("gemm", "grouped_gemm")]
        self.assertIn(
            ("gemm", {"dtype": "fp8_block", "m": 64, "n": 2048, "k": 7168}), matrices
        )
        self.assertIn(
            (
                "grouped_gemm",
                {
                    "dtype": "fp8_block",
                    "out_dtype": "bf16",
                    "batch_size": 16,
                    "m": 64,
                    "n": 1024,
                    "k": 4096,
                },
            ),
            matrices,
        )
        self.assertIn(
            ("gemm", {"dtype": "bf16", "m": 64, "n": 384, "k": 7168}), matrices
        )
        self.assertIn(
            ("gemm", {"dtype": "bf16", "m": 7, "n": 129280, "k": 7168}), matrices
        )
        # Four attention matrices + gate, at 61 sites. Hash layers still run
        # the gate matrix for routing weights (model.py:1058/1098).
        self.assertAlmostEqual(b.stats.component_seconds["gemm"], 61 * 5e-6)
        self.assertNotIn(
            ("gemm", {"dtype": "fp8_block", "m": 64, "n": 66560, "k": 7168}), matrices
        )

    def test_grouped_work_has_disjoint_weights_and_bf16_output(self):
        work = grouped_gemm_work(16, 8, 1024, 4096, "fp8_block", "bf16")
        self.assertEqual(work.flops, 16 * 2 * 8 * 1024 * 4096)
        self.assertEqual(work.weight_bytes, 16 * 1024 * 4096)
        self.assertEqual(work.activation_bytes, 16 * 8 * (4096 + 1024 * 2))
        model = analytical_model_for(self.device.family)
        key = {
            "dtype": "fp8_block",
            "out_dtype": "bf16",
            "batch_size": 16,
            "m": 8,
            "n": 1024,
            "k": 4096,
        }
        estimate = model.estimate("grouped_gemm", key, self.device)
        self.assertEqual(estimate.overhead_us, 3.0)
        self.assertGreaterEqual(
            estimate.memory_time_us,
            work.weight_bytes / self.device.memory_bandwidth_bytes_per_s.value * 1e6,
        )

    def test_profile_counter_reset_keeps_cached_estimates_and_missing_checklist(self):
        b = self.backend()
        r = Request(0, 512, 4, 256)
        latency = b.estimate_step_latency([r])
        cache, report = b._cache.copy(), b.stats.missing
        self.assertGreater(sum(b.stats.component_seconds.values()), 0)
        b.reset_profile_stats()
        self.assertEqual(sum(b.stats.component_seconds.values()), 0)
        self.assertEqual(b.stats.step_count, 0)
        self.assertIs(b.stats.missing, report)
        self.assertEqual(b._cache, cache)
        self.assertEqual(b.estimate_step_latency([r]), latency)

    def test_collection_manifest_matches_runtime_projection_shapes(self):
        manifest = build_manifest(
            "dsv4-test",
            [self.model],
            grid=WorkloadGrid(batch_sizes=(1,), prefill_tokens=(512,), tp_sizes=(1, 8)),
        )
        self.assertTrue(manifest.keys["grouped_gemm"])
        for key in manifest.keys["grouped_gemm"]:
            self.assertEqual(key["dtype"], "fp8_block")
            self.assertEqual(key["out_dtype"], "bf16")
        router = [k for k in manifest.keys["gemm"] if k["n"] == 384]
        self.assertTrue(all(k["dtype"] == "bf16" for k in router))


if __name__ == "__main__":
    unittest.main()
