"""Numerical attention and projection controls for within-head output shards."""

import unittest

import numpy as np

from TokenSim.af.runtime.attention_output import attention_output_partition
from TokenSim.af.runtime.kv_sequence import token_owner
from TokenSim.af.providers.context_parallel import combine_payload_bytes
from TokenSim.errors import ConfigurationError


class OutputChannelPartitionTest(unittest.TestCase):
    def test_split_sequence_and_output_channels_match_complete_attention(self):
        rng = np.random.default_rng(9271)
        heads, kv_heads, dim, hidden = 8, 2, 8, 16
        for context in (3, 23):
            q = rng.normal(size=(heads, dim))
            k = rng.normal(size=(context, kv_heads, dim))
            v = rng.normal(size=(context, kv_heads, dim))
            projection = rng.normal(size=(heads * dim, hidden))
            expanded_k = np.repeat(k, heads // kv_heads, axis=1)
            expanded_v = np.repeat(v, heads // kv_heads, axis=1)
            logits = np.einsum("hd,nhd->hn", q, expanded_k) / np.sqrt(dim)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            attention = np.einsum(
                "hn,nhd->hd", exp / exp.sum(axis=1, keepdims=True), expanded_v
            )
            expected = attention.ravel() @ projection
            for tp, kvp in ((4, 2), (16, 8), (32, 8), (64, 16)):
                with self.subTest(context=context, tp=tp, kvp=kvp):
                    output = attention_output_partition(heads, dim, tp, kvp)
                    q_heads = heads // (tp // kvp)
                    actual = np.zeros(hidden)
                    retained = np.zeros_like(attention).ravel()
                    coverage = np.zeros(heads * dim, dtype=int)
                    for group in range(tp // kvp):
                        head_start = group * q_heads
                        partial = np.zeros((kvp, q_heads, dim))
                        lse = np.full((kvp, q_heads), -np.inf)
                        for rank in range(kvp):
                            positions = [
                                i
                                for i in range(context)
                                if token_owner(i, kvp, 2) == rank
                            ]
                            if not positions:
                                continue
                            scores = logits[
                                head_start : head_start + q_heads, positions
                            ]
                            m = scores.max(axis=1)
                            p = np.exp(scores - m[:, None])
                            sums = p.sum(axis=1)
                            lse[rank] = m + np.log(sums)
                            values = expanded_v[
                                positions, head_start : head_start + q_heads
                            ]
                            partial[rank] = np.einsum(
                                "hn,nhd->hd", p / sums[:, None], values
                            )
                        for destination in range(kvp):
                            global_rank = group * kvp + destination
                            begin = global_rank * output.width
                            end = begin + output.width
                            first_head = begin // dim - head_start
                            # All channels of a head share the same split-normalization weights.
                            selected_lse = lse[
                                :, first_head : first_head + output.heads
                            ]
                            m = selected_lse.max(axis=0)
                            weights = np.exp(selected_lse - m)
                            weights /= weights.sum(axis=0)
                            local_begin = destination * output.width
                            fragment = partial.reshape(kvp, -1)[
                                :, local_begin : local_begin + output.width
                            ]
                            fragment = fragment.reshape(
                                kvp, output.heads, output.head_dim
                            )
                            combined = (
                                (fragment * weights[:, :, None]).sum(axis=0).ravel()
                            )
                            retained[begin:end] = combined
                            coverage[begin:end] += 1
                            actual += combined @ projection[begin:end]
                            # Each source sends this slice and one LSE per retained head fragment.
                            packet = np.concatenate(
                                (fragment.ravel(), selected_lse.ravel())
                            )
                            self.assertEqual(
                                packet.size * 4,
                                combine_payload_bytes(
                                    1, output.heads, output.head_dim, kvp
                                ),
                            )
                    np.testing.assert_array_equal(coverage, np.ones(heads * dim))
                    np.testing.assert_allclose(
                        retained, attention.ravel(), rtol=1e-13, atol=1e-13
                    )
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_whole_heads_preserve_shape_and_fragments_record_exact_channels(self):
        whole = attention_output_partition(32, 512, 32, 8)
        self.assertEqual(whole.local_shape(7), {"output_heads": 1})
        fragment = attention_output_partition(32, 512, 64, 16)
        self.assertEqual(fragment.width, 256)
        self.assertEqual(
            fragment.local_shape(7),
            dict(
                output_heads=1,
                output_head_dim=256,
                output_channel_start=1792,
                output_channel_stop=2048,
            ),
        )
        self.assertEqual(combine_payload_bytes(1, 1, 256, 16), 16448)
        self.assertEqual(combine_payload_bytes(1, 1, 512, 8), 16416)

    def test_invalid_query_and_output_partitions_are_rejected(self):
        for heads, dim, tp, kvp in (
            (8, 8, 16, 1),
            (8, 7, 16, 8),
            (8, 8, 12, 3),
            (8, 8, 16, 3),
            (8, 8, 12, 2),
            (8, 8, 0, 1),
            (8, 8, 16, True),
        ):
            with self.subTest(heads=heads, dim=dim, tp=tp, kvp=kvp):
                with self.assertRaises(ConfigurationError):
                    attention_output_partition(heads, dim, tp, kvp)


if __name__ == "__main__":
    unittest.main()
