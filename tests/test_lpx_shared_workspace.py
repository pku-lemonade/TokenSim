"""Explicit cache tiles, packing costs and native admission at the new boundary."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from TokenSim.af.runtime.lpx_memory import workspace_components
from TokenSim.config.kv_storage import shared_workspace_components
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.analytical.lp30 import estimate_lp30
from scripts.lpx.fidelity import load, validate
from scripts.lpx.io_pool import transformer_setup
from scripts.lpx.kv_attention import reference_attention
from scripts.lpx.kv_codec import encode, fold_query
from scripts.lpx.mfu_probe import MFUCase, runtime_for
from scripts.lpx.native_search import memory_report
from scripts.run_lpx_article_sweep import article_config


class SharedWorkspaceTest(unittest.TestCase):
    def config(self, kernel="assemble", workspace="assemble"):
        config = article_config(
            load(
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/fidelity-study.json"
            )
        )
        config["operator"]["shared_kv_kernel"] = kernel
        config["memory"]["shared_kv_workspace"] = workspace
        return config

    def test_explicit_tiles_retain_quantization_and_partial_score_storage(self):
        geometry = dict(
            head_dim=512, kv_heads=1, query_heads=8, queries=1, tile_keys=256, buffers=2
        )
        assembled = shared_workspace_components(kernel="assemble", **geometry)
        packed = shared_workspace_components(kernel="split_dot_packed", **geometry)
        self.assertEqual(assembled["physical_cache_tiles"], 512 * 648)
        self.assertEqual(assembled["decoded_value_and_rotary_tiles"], 512 * 640 * 2)
        self.assertEqual(assembled["key_tiles"], 512 * 512 * 2)
        self.assertEqual(packed["key_tiles"], 512 * 384 * 2)
        self.assertEqual(packed["partial_score_tile"], 8 * 256 * 4)
        self.assertEqual(packed["append_quantization"], 640 * 4 + 8)
        with self.assertRaises(ConfigurationError):
            shared_workspace_components(kernel="split_dot", **geometry)

    def test_native_compact_workspace_admits_fifteen_and_rejects_sixteen(self):
        for kernel, expected in (("assemble", 1581704), ("split_dot_packed", 1458824)):
            _, hardware, model, proxy = transformer_setup(
                self.config(kernel, kernel),
                tp=16,
                queries=1,
                global_kv_layout="value_rope_fp8",
            )
            runtime = runtime_for(hardware, proxy, MFUCase(16, 4, 10), 10, (6,) * 10)
            report = memory_report(runtime, 102000, 15)
            self.assertEqual(
                report["peak_rank"]["buffer_breakdown"]["lp30_live_scratch"], expected
            )
            with self.assertRaises(ConfigurationError):
                memory_report(runtime, 102000, 16)
            component = SimpleNamespace(
                category="attention",
                layer_id=5,
                shape=dict(
                    q_heads_local=8,
                    kv_heads_local=1,
                    head_dim=512,
                    kv_parallel_size=4,
                    o_heads_local=2,
                ).items(),
            )
            with self.assertRaises(ConfigurationError):
                workspace_components(
                    (component,),
                    model,
                    hardware.device("groq_lp30").analytical["lpx_memory"],
                )

    def test_pack_kernel_prices_bf16_gather_and_output_writes(self):
        _, hardware, _, _ = transformer_setup(
            self.config("split_dot_packed", "split_dot_packed"),
            tp=16,
            queries=1,
            global_kv_layout="value_rope_fp8",
        )
        device = hardware.device("groq_lp30")
        key = dict(
            attn_dtype="bf16",
            kv_cache_dtype="fp8",
            kv_cache_layout="value_rope_fp8",
            num_heads=8,
            num_kv_heads=1,
            head_dim=512,
            window_size=0,
            batch_size=1,
            context_len=256,
        )
        packed = estimate_lp30("generation_attention", key, device)
        from dataclasses import replace

        analytical = deepcopy(device.analytical)
        analytical["lpx_pipeline"]["shared_kv_kernel"] = "split_dot"
        strided = estimate_lp30(
            "generation_attention", key, replace(device, analytical=analytical)
        )
        self.assertEqual(packed.work.flops, strided.work.flops)
        self.assertEqual(packed.work.kv_bytes, strided.work.kv_bytes)
        self.assertEqual(
            packed.details["shuffle_bytes"] - strided.details["shuffle_bytes"],
            256 * 384 * 4,
        )
        self.assertGreater(packed.latency_us, strided.latency_us)

    def test_incompatible_kernel_workspace_policy_rejects(self):
        for kernel, workspace in (
            ("assemble", "split_dot_packed"),
            ("split_dot", "assemble"),
        ):
            with self.assertRaises(ConfigurationError):
                validate(self.config(kernel, workspace))

    def test_tiled_kernels_agree_with_full_quantized_attention(self):
        rng = np.random.default_rng(1729)
        for dim in (16, 512):
            value = rng.normal(size=(19, dim)).astype(np.float32)
            key = rng.normal(size=value.shape).astype(np.float32)
            query = rng.normal(size=(3, dim)).astype(np.float32)
            original = query.copy()
            gamma = rng.uniform(0.5, 1.5, size=dim).astype(np.float32)
            packed = encode(key, value)
            virtual, decoded = packed.decode()
            scores = fold_query(query, gamma) @ virtual.T / np.sqrt(dim)
            scores -= scores.max(axis=-1, keepdims=True)
            probability = np.exp(scores)
            probability /= probability.sum(axis=-1, keepdims=True)
            expected = probability @ decoded
            for kernel in ("assemble", "split_dot_packed"):
                for tile in (1, 7, 64):
                    result = reference_attention(
                        packed,
                        query,
                        gamma,
                        tile_keys=tile,
                        kernel=kernel,
                        scale=1 / np.sqrt(dim),
                    )
                    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-6)
            np.testing.assert_array_equal(query, original)


if __name__ == "__main__":
    unittest.main()
