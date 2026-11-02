"""Independent request rows preserve separate KV and boundary-token accounting."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.workload import work_for_table
from scripts.lpx.fidelity import load
from scripts.lpx.io_pool import transformer_profile, transformer_setup
from scripts.run_lpx_article_sweep import article_config


class IOBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = article_config(
            load(
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/fidelity-study.json"
            )
        )

    def test_total_query_rows_reserve_batch_times_queries(self):
        base, _, _, _ = transformer_setup(self.config, tp=16, queries=3)
        batch, _, model, _ = transformer_setup(self.config, tp=16, queries=3, batch=2)
        self.assertEqual(batch["memory"]["prefill_chunk"], 6)
        self.assertEqual(
            batch["memory"]["reserve_bytes"] - base["memory"]["reserve_bytes"],
            12 * (model.vocab_size - model.hidden_size),
        )
        self.assertEqual(self.config["memory"]["prefill_chunk"], 16)

    def test_capture_creates_two_independent_sessions_and_complete_cohorts(self):
        profile = SimpleNamespace(nodes=(), metadata={})
        with (
            tempfile.TemporaryDirectory() as cache,
            patch(
                "scripts.lpx.io_pool.cached_profile", return_value=profile
            ) as capture,
            patch(
                "scripts.lpx.io_pool.transformer_component",
                side_effect=lambda p, **kw: p,
            ),
        ):
            _, _, _, admission = transformer_profile(
                self.config,
                tp=16,
                kvp=4,
                pp=10,
                queries=1,
                batch=2,
                cache=cache,
                resident=4,
            )
            call = capture.call_args.kwargs
            self.assertEqual(call["batch"], 2)
            self.assertEqual(call["queries"], 1)
            self.assertEqual(call["contexts"], (101000, 101000))
            self.assertEqual(call["maximum_contexts"], (102000, 102000))
            self.assertEqual(admission["scratch_query_rows"], 2)
            self.assertEqual(admission["memory"]["resident_requests"], 4)
            with self.assertRaises(ConfigurationError):
                transformer_profile(
                    self.config,
                    tp=16,
                    kvp=4,
                    pp=10,
                    queries=1,
                    batch=2,
                    cache=cache,
                    resident=3,
                )

    def test_independent_batch_does_not_reuse_a_verifiers_prefix(self):
        common = dict(
            attn_dtype="bf16",
            kv_cache_dtype="fp8",
            num_heads=8,
            num_kv_heads=1,
            head_dim=512,
            window_size=0,
        )
        independent = work_for_table(
            "generation_attention", dict(**common, batch_size=2, context_len=128)
        )
        verification = work_for_table(
            "context_attention",
            dict(**common, batch_size=1, input_seq_len=2, context_len=128),
        )
        self.assertGreater(independent.kv_bytes, verification.kv_bytes)
        self.assertEqual(independent.kv_bytes, 2 * (128 + 1) * 2 * 512)


if __name__ == "__main__":
    unittest.main()
