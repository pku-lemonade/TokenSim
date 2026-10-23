"""KV-parallel token ownership and attention partitions."""

from dataclasses import replace
from pathlib import Path
import unittest

from TokenSim.af.runtime.kv_sequence import (
    attention_partition,
    kvp_coordinates,
    owned_tokens,
    peak_resident_tokens,
    token_owner,
    visible_tokens,
)
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext


def brute_visible(context, window, kvp, rank, block):
    start = max(0, context - window) if window else 0
    return sum(
        1
        for position in range(start, context)
        if token_owner(position, kvp, block) == rank
    )


class KVOwnershipTest(unittest.TestCase):
    def test_closed_form_counts_match_enumeration_and_cover_context(self):
        for kvp in (1, 2, 3, 4):
            for block in (1, 5, 16):
                for context in (0, 1, 15, 16, 17, 63, 64, 65, 200):
                    counts = [owned_tokens(context, kvp, r, block) for r in range(kvp)]
                    self.assertEqual(sum(counts), context)
                    for rank, count in enumerate(counts):
                        self.assertEqual(
                            count, brute_visible(context, None, kvp, rank, block)
                        )

    def test_windowed_visibility_matches_enumeration(self):
        for kvp, block, window in ((2, 4, 10), (4, 16, 1024), (3, 5, 7)):
            for context in range(0, 3 * window + 2 * kvp * block, 7):
                counts = [
                    visible_tokens(context, window, kvp, r, block) for r in range(kvp)
                ]
                self.assertEqual(sum(counts), min(context, window))
                for rank, count in enumerate(counts):
                    self.assertEqual(
                        count, brute_visible(context, window, kvp, rank, block)
                    )

    def test_peak_residency_matches_scan_of_every_context(self):
        for kvp, block, window, maximum in (
            (2, 4, 10, 37),
            (4, 3, 10, 9),
            (3, 5, 7, 100),
            (4, 16, None, 102000),
        ):
            for rank in range(kvp):
                if window is None:
                    expected = owned_tokens(maximum, kvp, rank, block)
                else:
                    expected = max(
                        brute_visible(context, window, kvp, rank, block)
                        for context in range(maximum + 1)
                    )
                self.assertEqual(
                    peak_resident_tokens(maximum, window, kvp, rank, block), expected
                )

    def test_specified_gemma_residency_values(self):
        self.assertEqual(peak_resident_tokens(102000, 0, 4, 0, 16), 25504)
        self.assertEqual(peak_resident_tokens(102000, 0, 2, 0, 16), 51008)
        self.assertEqual(peak_resident_tokens(102000, 0, 1, 0, 16), 102000)

    def test_contiguous_control_uses_one_large_block(self):
        maximum, kvp = 100, 4
        block = -(-maximum // kvp)
        owners = [token_owner(position, kvp, block) for position in range(maximum)]
        self.assertEqual(owners, sorted(owners))
        self.assertEqual(
            [owned_tokens(maximum, kvp, r, block) for r in range(kvp)], [25] * 4
        )

    def test_consecutive_tp_ranks_form_kvp_groups(self):
        self.assertEqual(
            [kvp_coordinates(rank, 4) for rank in range(8)],
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )

    def test_invalid_groups_raise_configuration_errors(self):
        for call in (
            lambda: owned_tokens(4, 0, 0, 16),
            lambda: owned_tokens(4, 2, 2, 16),
            lambda: owned_tokens(4, 2, 0, 0),
            lambda: owned_tokens(-1, 2, 0, 16),
            lambda: token_owner(True, 2, 16),
            lambda: kvp_coordinates(0, 0),
        ):
            with self.assertRaises(ConfigurationError):
                call()


class AttentionPartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware = HardwareContext.load(Path(__file__).resolve().parents[3] / "data")
        cls.model = hardware.model("gemma-4-31b-text")

    def test_gemma_global_layers_remove_kv_replication(self):
        baseline = attention_partition(self.model, 5, 16)
        self.assertFalse(baseline.enabled)
        self.assertEqual(
            (baseline.query_heads_per_rank, baseline.kv_heads_per_rank), (2, 1)
        )
        self.assertEqual(baseline.kv_replica_factor, 4)
        split = attention_partition(self.model, 5, 16, 4)
        self.assertTrue(split.enabled)
        self.assertEqual(split.attention_tensor_parallel_size, 4)
        self.assertEqual(
            (
                split.query_heads_per_rank,
                split.output_heads_per_rank,
                split.kv_heads_per_rank,
                split.kv_replica_factor,
            ),
            (8, 2, 1, 1),
        )

    def test_layer_selection_keeps_windowed_layers_on_tensor_parallelism(self):
        local = attention_partition(self.model, 0, 16, 4)
        self.assertFalse(local.enabled)
        self.assertEqual(local.attention_tensor_parallel_size, 16)
        self.assertEqual(local.kv_heads_per_rank, 1)
        selected = attention_partition(self.model, 0, 16, 4, "all")
        self.assertEqual(
            (selected.kv_parallel_size, selected.kv_heads_per_rank), (4, 4)
        )
        self.assertEqual(selected.as_dict()["query_heads_per_rank"], 8)

    def test_unit_kv_parallelism_preserves_tensor_parallel_partition(self):
        for layer in range(self.model.num_layers):
            for tp in (1, 2, 4, 8, 16):
                partition = attention_partition(self.model, layer, tp)
                self.assertEqual(partition.attention_tensor_parallel_size, tp)
                self.assertEqual(
                    partition.query_heads_per_rank, partition.output_heads_per_rank
                )

    def test_invalid_partitions_raise_configuration_errors(self):
        uneven = replace(self.model, num_attention_heads=16)
        for call in (
            lambda: attention_partition(self.model, 5, 16, 3),
            lambda: attention_partition(self.model, 5, 16, 2, "local"),
            lambda: attention_partition(uneven, 5, 32, 2),
            lambda: attention_partition(self.model, 5, 0),
        ):
            with self.assertRaises(ConfigurationError):
                call()


if __name__ == "__main__":
    unittest.main()
