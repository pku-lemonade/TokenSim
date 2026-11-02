import unittest

from scripts import analyze_lpx_attention_sensitivity as sensitivity


def steps(low, high, coding):
    return dict(aa_10k=low, aa_100k=high, coding_16k=coding)


VERIFY = {
    1: steps(100.0, 110.0, 101.0),
    7: steps(160.0, 200.0, 163.0),
    8: steps(170.0, 220.0, 174.0),
}


class ScaledGrowthTest(unittest.TestCase):
    def test_zero_share_gives_every_row_count_the_one_row_growth(self):
        bound = sensitivity.scaled_growth(VERIFY, 0.0)
        self.assertEqual(bound[1], VERIFY[1])
        self.assertEqual(bound[8], steps(170.0, 180.0, 171.0))

    def test_shares_interpolate_the_growth_beyond_one_row(self):
        self.assertEqual(sensitivity.scaled_growth(VERIFY, 1.0), VERIFY)
        half = sensitivity.scaled_growth(VERIFY, 0.5)
        # The 8-row growth of 50 µs exceeds the one-row growth by 40 µs.
        self.assertAlmostEqual(half[8]["aa_100k"] - half[8]["aa_10k"], 30.0)
        self.assertEqual(half[8]["aa_10k"], 170.0)


class CapturedStepsTest(unittest.TestCase):
    def test_rows_are_grouped_by_variant_and_row_count(self):
        summary = dict(
            rows=[
                dict(variant="baseline", rows=1, step_us=VERIFY[1]),
                dict(variant="matrix", rows=8, step_us=VERIFY[8]),
            ]
        )
        self.assertEqual(
            sensitivity.captured_steps(summary),
            dict(baseline={1: VERIFY[1]}, matrix={8: VERIFY[8]}),
        )


class ReproductionTest(unittest.TestCase):
    def test_seven_rows_are_compared_separately(self):
        committed = {**VERIFY, 7: steps(165.0, 205.0, 168.0)}
        captured = {**VERIFY, 1: steps(100.05, 110.0, 101.0)}
        result = sensitivity.reproduction(captured, committed)
        self.assertAlmostEqual(result["maximum_difference_us"], 0.05)
        self.assertEqual(
            result["seven_rows_minus_interpolation_us"],
            steps(-5.0, -5.0, -5.0),
        )


class OverlapTest(unittest.TestCase):
    def test_overlap_of_two_windows(self):
        self.assertEqual(sensitivity.overlap([1.0, 5.0], [3.0, 8.0]), [3.0, 5.0])
        self.assertIsNone(sensitivity.overlap([1.0, 2.0], [3.0, 8.0]))
        self.assertIsNone(sensitivity.overlap(None, [3.0, 8.0]))


if __name__ == "__main__":
    unittest.main()
