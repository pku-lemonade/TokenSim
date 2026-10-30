import unittest

from scripts.check_lpx_gpu_reference import finish_reason, token_fingerprint


class GpuReferenceTest(unittest.TestCase):
    def test_prompt_fingerprint_preserves_order_without_storing_tokens(self):
        result = token_fingerprint([3, 4, 5])
        self.assertEqual(result["tokens"], 3)
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(result, token_fingerprint([3, 4, 5]))
        self.assertNotEqual(result, token_fingerprint([5, 4, 3]))

    def test_eos_takes_precedence_at_the_length_boundary(self):
        self.assertEqual(finish_reason([4, 5], [5, 6], 2), "stop")
        self.assertEqual(finish_reason([4, 5], 5, 2), "stop")
        self.assertEqual(finish_reason([4, 5], None, 2), "length")
        self.assertEqual(finish_reason([4], [5], 2), "unknown")


if __name__ == "__main__":
    unittest.main()
