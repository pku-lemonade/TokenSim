"""Shared-KV assistant structure, memory and query-head routing oracles."""

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import unittest

from TokenSim.errors import ConfigurationError
from scripts.lpx.assistant_pool import (
    AssistantSpec,
    assistant_memory,
    intersections,
    remote_workspace,
)
from scripts.lpx.fidelity import load
from scripts.run_lpx_article_sweep import article_config
from scripts.run_lpx_context_parallel import environment


class AssistantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = article_config(
            load(
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/fidelity-study.json"
            )
        )
        cls.hardware, cls.target, _, _ = environment(config)
        cls.assistant_config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/gemma4-31b-assistant-config.json"
            ).read_text()
        )
        cls.spec = AssistantSpec.from_config(cls.assistant_config, cls.target)

    def test_parameter_ledger_matches_official_tensor_count(self):
        parameters = self.spec.parameters()
        self.assertEqual(sum(parameters.values()), 469518596)
        self.assertEqual(parameters["tied_embedding_lm_head"], 1024 * 262144)
        self.assertEqual(
            parameters["attention_q_and_o"], 2 * 1024 * 32 * (3 * 256 + 512)
        )
        self.assertNotIn("attention_k_and_v", parameters)

    def test_bf16_shards_replicate_normalization_and_preserve_matrix_weights(self):
        rows = assistant_memory(self.spec, 2, "bf16", slots=1, resident=9)
        parameters = self.spec.parameters()
        norms = sum(
            parameters[name]
            for name in ("layer_norms", "final_norm", "query_norms", "layer_scalars")
        )
        self.assertEqual(
            sum(row["weights_bytes"] for row in rows), 2 * (469518596 + norms)
        )
        self.assertTrue(all(row["headroom_bytes"] > 0 for row in rows))
        self.assertEqual(rows[0]["persistent_feedback_bytes"], 9 * 2 * 5376)
        with self.assertRaises(ConfigurationError):
            assistant_memory(self.spec, 1, "bf16", slots=1, resident=9)
        with self.assertRaises(ConfigurationError):
            assistant_memory(self.spec, 3, "fp8", slots=1, resident=9)

    def test_head_intersections_partition_all_heads_in_both_directions(self):
        expected = [(0, 0, 8), (0, 1, 8), (1, 2, 8), (1, 3, 8)]
        self.assertEqual(intersections([16, 16], [8, 8, 8, 8]), expected)
        self.assertEqual(intersections([32], [2] * 16), [(0, i, 2) for i in range(16)])
        with self.assertRaises(ConfigurationError):
            intersections([32], [8, 8])

    def test_incomplete_geometry_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            replace(self.spec, dims=(256,))
        with self.assertRaises(ConfigurationError):
            replace(self.spec, kv_heads=(3, 16, 16, 4))
        with self.assertRaises(ConfigurationError):
            AssistantSpec.from_config({}, self.target)
        mismatched = deepcopy(self.assistant_config)
        mismatched["text_config"]["global_head_dim"] = 256
        with self.assertRaises(ConfigurationError):
            AssistantSpec.from_config(mismatched, self.target)

    def test_cache_owner_workspace_scales_with_local_kv_and_query_heads(self):
        self.assertEqual(
            remote_workspace(self.spec, tp=16, kvp=4, tile_keys=256), 1048576
        )
        self.assertEqual(
            remote_workspace(
                self.spec,
                tp=16,
                kvp=4,
                tile_keys=256,
                global_kv_layout="value_rope_fp8",
            ),
            2 * 1048576,
        )
        self.assertEqual(
            remote_workspace(self.spec, tp=16, kvp=16, tile_keys=256), 4 * 1048576
        )
        with self.assertRaises(ConfigurationError):
            remote_workspace(self.spec, tp=32, kvp=8, tile_keys=256)


if __name__ == "__main__":
    unittest.main()
