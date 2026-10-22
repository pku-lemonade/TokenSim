"""Mixed-layer compute, physical cache partitions and shared K/V projections."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from TokenSim.af.runtime.admission import rank_kv_bytes
from TokenSim.config.attention_geometry import AttentionGeometry
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request
from scripts.lpx_runtime import Layout
from scripts.lpx_runtime import configuration


class LayerAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = ModelSpec(
            "mixed",
            32,
            64,
            2,
            4,
            2,
            8,
            vocab_size=64,
            attention_windows=(4, 0),
            attention_geometries=(
                AttentionGeometry(8, 2),
                AttentionGeometry(16, 1, True),
            ),
            dtype="bf16",
            max_position_embeddings=1024,
        )

    def test_projection_sharing_preserves_both_transformed_caches(self):
        runtime = configuration(
            self.hardware, self.model, Layout(a_tp=2, f_ep=2), namespace="mixed"
        )
        binding = next(
            b for b in runtime.plan.rank_bindings if b.capability == "af_attention"
        )
        self.assertIn(
            "communication_workspace",
            dict(
                runtime.plan.physical_state(binding.physical_identity).protocol_buffers
            ),
        )
        local = runtime.plan.local_shape(
            binding.domain, binding.domain_rank, 1, "attention"
        )
        self.assertEqual(
            (local["head_dim"], local["num_kv_heads"], local["kv_projections"]),
            (16, 1, 1),
        )
        # Local layer: 4 cached positions * (K+V) * 1 head * 8 dims * BF16.
        # Global layer: 100 positions * (K+V) * 1 replicated head * 16 dims * BF16.
        self.assertEqual(rank_kv_bytes(runtime.plan, binding, 100), 4 * 32 + 100 * 64)
        components = [
            c
            for c in runtime.plan.physical_state(binding.physical_identity).components
            if c.category == "attention"
        ]
        weights = {c.layer_id: c.bytes for c in components}
        self.assertEqual(
            weights, {0: (2 * 32 * 16 + 2 * 32 * 8) * 2, 1: (2 * 32 * 32 + 32 * 16) * 2}
        )

    def test_prefill_queries_use_each_window_and_geometry(self):
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=self.model,
            fallback="analytical_only",
        )
        requests = [Request(0, 32, 2, 16)]
        observed = []
        original = OperatorTableLatencyBackend.query

        def query(owner, table, key):
            if table == "context_attention":
                observed.append(dict(key))
            return original(owner, table, key)

        with patch.object(OperatorTableLatencyBackend, "query", query):
            latency = backend.estimate_step_latency(requests)
        self.assertGreater(latency, 0)
        self.assertEqual(
            {(k["window_size"], k["head_dim"], k["num_kv_heads"]) for k in observed},
            {(4, 8, 2), (0, 16, 1)},
        )
        self.assertEqual(backend.stats.step_count, 1)

    def test_cached_prefill_uses_kv_reads_without_scaling_launch_overhead(self):
        model = replace(self.model, attention_geometries=(), attention_windows=())
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=model,
            fallback="analytical_only",
        )
        actual = backend._attention_prefill(1, 8, 400008)
        expected = backend.query(
            "context_attention",
            dict(
                attn_dtype=model.activation_dtype,
                kv_cache_dtype=model.kv_cache_dtype,
                batch_size=1,
                input_seq_len=8,
                context_len=400008,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                window_size=0,
            ),
        ).latency_us
        self.assertEqual(actual, expected)
        self.assertLess(actual, 100)

    def test_mxfp4_experts_reach_prefill_backend(self):
        model = replace(
            self.hardware.model("moe-toy"),
            attention_windows=(4, 0, 4, 0),
            operator_dtypes={"routed_experts": "mxfp4"},
        )
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=model,
            fallback="analytical_only",
        )
        self.assertEqual(backend.moe_dtype, "mxfp4")
        self.assertTrue(backend._layer_backends)
        self.assertTrue(
            all(child.moe_dtype == "mxfp4" for child in backend._layer_backends)
        )

    def test_catalog_geometry_and_parameter_count(self):
        model = self.hardware.model("gemma-4-31b-text")
        self.assertEqual(sum(model.attention_window(i) == 0 for i in range(60)), 10)
        self.assertEqual(model.attention_geometry(5), AttentionGeometry(512, 4, True))
        self.assertTrue(model.gated)
        self.assertTrue(30_000_000_000 < model.total_params() < 32_000_000_000)
        with self.assertRaises(ConfigurationError):
            replace(self.model, attention_geometries=(AttentionGeometry(8, 2),))


if __name__ == "__main__":
    unittest.main()
