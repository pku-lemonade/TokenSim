import unittest
from util.agentx_accuracy import error_assessment


class AccuracyPolicyTest(unittest.TestCase):
    def test_boundary_and_both_directions_without_rounding(self):
        for value in (75, 125):
            self.assertEqual(error_assessment(value, 100)["status"], "acceptable")
        for value in (90, 110, 100):
            self.assertEqual(error_assessment(value, 100)["status"], "preferred")
        for value in (74.99999, 125.00001):
            self.assertEqual(error_assessment(value, 100)["status"], "outside")

    def test_missing_incompatible_and_zero_reference_are_not_fabricated(self):
        self.assertIsNone(error_assessment(None, 100)["acceptable"])
        self.assertIsNone(error_assessment(100, 100, comparable=False)["acceptable"])
        self.assertIsNone(error_assessment(float("nan"), 100)["acceptable"])
        self.assertEqual(error_assessment(0, 0)["status"], "preferred")
        self.assertEqual(error_assessment(1, 0)["status"], "outside")
        self.assertIsNone(error_assessment(1, 0)["delta_percent"])
