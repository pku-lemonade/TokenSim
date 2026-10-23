import unittest

from TokenSim.errors import ConfigurationError
from scripts.lpx.speculative_fit import (
    acceptance_for,
    committed_tokens,
    shared_match,
)


class SpeculativeFitTest(unittest.TestCase):
    def test_committed_tokens_matches_chain_expectation(self):
        self.assertEqual(committed_tokens(0.0, 4), 1.0)
        self.assertEqual(committed_tokens(1.0, 4), 5.0)
        # 1 + a + a^2 + a^3 for three draft tokens.
        self.assertAlmostEqual(committed_tokens(0.5, 3), 1.875)
        with self.assertRaises(ConfigurationError):
            committed_tokens(1.5, 3)

    def test_acceptance_inverts_committed_tokens(self):
        for draft in (1, 3, 7):
            for alpha in (0.1, 0.5, 0.9):
                tokens = committed_tokens(alpha, draft)
                self.assertAlmostEqual(acceptance_for(tokens, draft), alpha, places=9)
        self.assertIsNone(acceptance_for(0.9, 3))
        self.assertIsNone(acceptance_for(4.1, 3))

    def test_shared_match_intersects_every_band(self):
        # 300 and 330 microseconds at 1.5 tokens per step: 5,000 and 4,545 TPS.
        bands = [(4750, 5250), (4318, 4773)]
        match = shared_match([300e-6, 330e-6], bands, 1)
        self.assertIsNotNone(match)
        low, high = match
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)
        # A steeper 100K step leaves no common acceptance.
        self.assertIsNone(shared_match([300e-6, 400e-6], bands, 1))
        # No acceptance commits fewer than one token per step.
        self.assertIsNone(shared_match([100e-6, 110e-6], bands, 1))


if __name__ == "__main__":
    unittest.main()
