"""Physical FP8 payload, p-RoPE coordinates and shared-key arithmetic."""

from dataclasses import replace
import unittest

import numpy as np

from TokenSim.config.kv_storage import rotary_indices, shared_head_bytes
from TokenSim.errors import ConfigurationError
from scripts.lpx.kv_codec import LEVELS, encode, fold_query


class KVCodecTest(unittest.TestCase):
    def test_rotary_coordinates_follow_two_head_halves(self):
        self.assertEqual(rotary_indices(16), (0, 1, 8, 9))
        indices = rotary_indices(512)
        self.assertEqual(indices, (*range(64), *range(256, 320)))
        self.assertEqual(shared_head_bytes(512), 648)
        for dim in (0, 7, 12, True):
            with self.assertRaises(ConfigurationError):
                rotary_indices(dim)

    def test_reference_payload_includes_both_fp32_scales(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(3, 4, 512)).astype(np.float32)
        keys = rng.normal(size=values.shape).astype(np.float32)
        packed = encode(keys, values)
        self.assertEqual(packed.nbytes, 3 * 4 * 648)
        decoded_key, decoded_value = packed.decode()
        rotary = list(rotary_indices(512))
        plain = sorted(set(range(512)) - set(rotary))
        np.testing.assert_array_equal(
            decoded_key[..., plain], decoded_value[..., plain]
        )
        for original, decoded in (
            (values, decoded_value),
            (keys[..., rotary], decoded_key[..., rotary]),
        ):
            bound = np.max(np.abs(original), axis=-1, keepdims=True) / 28
            self.assertTrue(np.all(np.abs(decoded - original) <= bound + 1e-6))

    def test_shared_dot_identity_with_exactly_representable_payload(self):
        rng = np.random.default_rng(17)
        values = rng.choice([-8, -2, 1, 4], size=(7, 16)).astype(np.float32)
        keys = values.copy()
        gamma = np.arange(1, 17, dtype=np.float32)
        keys *= gamma
        # These coordinates are post-RoPE inputs to the codec.
        rotary = list(rotary_indices(16))
        keys[..., rotary] = rng.choice([-4, -1, 2, 8], size=(7, 4))
        values[..., 15] = 448
        keys[..., 15] = values[..., 15] * gamma[15]
        keys[..., rotary[0]] = 448
        query = rng.normal(size=(3, 16)).astype(np.float32)
        packed = encode(keys, values)
        virtual_keys, decoded_values = packed.decode()
        np.testing.assert_array_equal(decoded_values, values)
        reconstructed = virtual_keys.copy()
        plain = sorted(set(range(16)) - set(rotary))
        reconstructed[..., plain] *= gamma[plain]
        np.testing.assert_array_equal(reconstructed, keys)
        np.testing.assert_allclose(
            query @ keys.T,
            fold_query(query, gamma) @ virtual_keys.T,
            rtol=2e-6,
            atol=0.01,
        )

    def test_e4m3_rounding_zero_and_invalid_payload(self):
        self.assertEqual(float(LEVELS[-1]), 448)
        self.assertEqual(float(LEVELS[1]), 2**-9)
        values = np.zeros((1, 16), dtype=np.float32)
        values[0, :4] = (448, 1.0625, 1.1875, -1.1875)
        packed = encode(values, values)
        np.testing.assert_array_equal(packed.value_codes[0, :4], (126, 56, 58, 186))
        zeros = encode(np.zeros_like(values), np.zeros_like(values))
        np.testing.assert_array_equal(zeros.value_scales, [[1]])
        np.testing.assert_array_equal(zeros.decode()[1], np.zeros_like(values))
        bad = packed.value_codes.copy()
        bad[0, 0] = 127
        with self.assertRaises(ConfigurationError):
            replace(packed, value_codes=bad)
        for key, value in (([float("nan")] * 16, [0] * 16), ([0] * 8, [0] * 16)):
            with self.assertRaises(ConfigurationError):
                encode(key, value)
        with self.assertRaises(ConfigurationError):
            fold_query(values, np.ones(8))


if __name__ == "__main__":
    unittest.main()
