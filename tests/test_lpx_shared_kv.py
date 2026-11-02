"""Shared KV bytes flow through native admission and explicit LP30 costs."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from TokenSim.af.providers.attention import attention_core
from TokenSim.af.runtime.layer_kv import layer_kv_partitions
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.kv_storage import head_storage_bytes
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.analytical.lp30 import estimate_lp30
from scripts.lpx.fidelity import load, with_fidelity
from scripts.lpx.io_pool import transformer_setup
from scripts.lpx.mfu_probe import MFUCase, runtime_for
from scripts.lpx.native_search import memory_report
from scripts.lpx.partition import native_footprint
from scripts.run_lpx_article_sweep import article_config


class SharedKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = article_config(
            load(
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/fidelity-study.json"
            )
        )
        _, cls.hardware, cls.baseline, cls.proxy = transformer_setup(
            cls.config, tp=16, queries=1
        )
        _, _, cls.shared, cls.shared_proxy = transformer_setup(
            cls.config, tp=16, queries=1, global_kv_layout="value_rope_fp8"
        )

    def test_model_roundtrip_and_incompatible_paths_reject(self):
        self.assertNotIn("global_kv_layout", self.baseline.to_dict())
        raw = asdict(self.shared)
        raw.pop("moe")
        parsed = ModelSpec.from_mapping(raw)
        self.assertEqual(parsed.global_kv_layout, "value_rope_fp8")
        self.assertEqual(
            parsed.attention_geometry(5), self.shared.attention_geometry(5)
        )
        self.assertEqual(parsed.to_dict()["global_kv_layout"], "value_rope_fp8")
        self.assertEqual(head_storage_bytes(self.shared, 5), 648)
        self.assertIsNone(head_storage_bytes(self.shared, 0))
        for values in (
            {"kv_cache_dtype": "bf16"},
            {"qk_norm": False},
            {"activation_dtype": "fp8"},
            {"global_kv_layout": "unknown"},
        ):
            with self.assertRaises(ConfigurationError):
                replace(self.shared, **values)
        with self.assertRaises(ConfigurationError):
            CacheConfig(16, self.hardware.device("groq_lp30"), self.shared)

    def test_native_global_cache_and_scratch_use_the_physical_format(self):
        case = MFUCase(16, 4, 10)
        base = runtime_for(self.hardware, self.proxy, case, 60)
        shared = runtime_for(self.hardware, self.shared_proxy, case, 60)
        old = native_footprint(base, 102000)
        new = native_footprint(shared, 102000)
        for layer in range(60):
            for rank in range(16):
                if (layer + 1) % 6:
                    self.assertEqual(new.kv[layer][rank], old.kv[layer][rank])
                else:
                    self.assertEqual(
                        new.kv[layer][rank] * 1024, old.kv[layer][rank] * 648
                    )
                    self.assertEqual(
                        new.blocks[layer][rank] * 1024, old.blocks[layer][rank] * 648
                    )
        self.assertTrue(all(new.buffers[i] > old.buffers[i] for i in range(16)))
        # The production allocator admits these leases and rejects an excess.
        active = runtime_for(self.hardware, self.shared_proxy, case, 10, (6,) * 10)
        self.assertGreater(
            memory_report(active, 102000, 14)["minimum_headroom_bytes"], 0
        )
        with self.assertRaises(ConfigurationError) as caught:
            memory_report(active, 102000, 15)
        self.assertIn("KV", str(caught.exception))

    def test_cost_prices_scales_reconstruction_and_dependent_query_transform(self):
        key = dict(
            attn_dtype="bf16",
            kv_cache_dtype="fp8",
            num_heads=8,
            num_kv_heads=1,
            head_dim=512,
            window_size=0,
            batch_size=1,
            context_len=256,
        )
        device = self.hardware.device("groq_lp30")
        base = estimate_lp30("generation_attention", key, device)
        extra_key = {**key, "kv_cache_layout": "value_rope_fp8"}
        estimates = []
        for kernel in ("assemble", "split_dot"):
            config = deepcopy(self.config)
            config["operator"]["shared_kv_kernel"] = kernel
            hardware = with_fidelity(self.hardware, config)
            estimate = estimate_lp30(
                "generation_attention", extra_key, hardware.device("groq_lp30")
            )
            self.assertEqual(estimate.work.flops, base.work.flops)
            self.assertEqual(estimate.work.kv_bytes * 1024, base.work.kv_bytes * 648)
            self.assertGreater(estimate.details["shared_kv_prefix_us"], 0)
            self.assertGreater(
                estimate.details["vector_operations"], base.details["vector_operations"]
            )
            self.assertGreater(
                estimate.details["shuffle_bytes"], base.details["shuffle_bytes"]
            )
            self.assertGreater(
                estimate.work.activation_bytes, base.work.activation_bytes
            )
            self.assertEqual(estimate.details["shared_kv"]["rotary_coordinates"], 128)
            estimates.append(estimate)
        self.assertGreater(
            estimates[0].details["shuffle_bytes"], estimates[1].details["shuffle_bytes"]
        )
        self.assertGreater(
            estimates[1].details["shared_kv_prefix_us"],
            estimates[0].details["shared_kv_prefix_us"],
        )

    def test_measured_tables_and_invalid_native_layer_geometry_are_rejected(self):
        provider = SimpleNamespace(model=self.shared)
        resolver = SimpleNamespace(
            device=self.hardware.device("groq_lp30"), package=object()
        )
        query = SimpleNamespace(layer_id=5, kv_cache_dtype=None)
        with self.assertRaises(ConfigurationError):
            attention_core(provider, resolver, query, 1, 8, 1, "bf16")
        native = runtime_for(self.hardware, self.shared_proxy, MFUCase(16, 4, 10), 60)
        layer = native.plan.layer_plan(5)
        for invalid in (
            replace(layer, k_eq_v=False),
            replace(layer, attention_window=1024),
        ):
            with self.assertRaises(ConfigurationError):
                layer_kv_partitions((invalid,), 1, 16, 1)
        with self.assertRaises(ConfigurationError):
            layer_kv_partitions((layer,), 1, 16, 2)


if __name__ == "__main__":
    unittest.main()
