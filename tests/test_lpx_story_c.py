import unittest

import numpy as np

from scripts import analyze_lpx_story_c as story


def verify_table(one_row, eight_rows):
    """Verification steps whose cost depends only on the row count."""
    return {
        rows: {
            k: one_row[k] + (eight_rows[k] - one_row[k]) * (rows - 1) / 7
            for k in one_row
        }
        for rows in range(1, 9)
    }


class AaSpeedTest(unittest.TestCase):
    def test_speed_counts_each_step_at_its_rows(self):
        verify = verify_table(dict(aa_10k=100.0), dict(aa_10k=170.0))
        # 10 one-row steps of 100 µs and 2 eight-row steps of 170 µs give
        # 30 tokens in 1,340 µs, counted at 1.5 native tokens per AA token.
        row = (30, [10, 0, 0, 0, 0, 0, 0, 2])
        speed = story.aa_speed(row, verify, "aa_10k", 0.0, 1.5)
        self.assertAlmostEqual(speed, 30 / 1340 * 1e6 / 1.5)
        slower = story.aa_speed(row, verify, "aa_10k", 5.0, 1.5)
        self.assertAlmostEqual(slower, 30 / 1400 * 1e6 / 1.5)


class WindowTest(unittest.TestCase):
    def test_window_is_none_when_one_context_never_fits(self):
        def speeds(extra):
            return dict(aa_10k=3382.0, aa_100k=2000.0)

        self.assertIsNone(story.shared_window(speeds, 7.0))

    def test_window_stops_at_the_upper_extra_time(self):
        def speeds(extra):
            return dict(aa_10k=3382.0 * (1 + 0.04 - extra / 100), aa_100k=3431.0)

        self.assertEqual(story.shared_window(speeds, 6.0), [0.0, 6.0])
        self.assertEqual(story.extra_grid(1.0), [0.0, 0.5, 1.0])


class BooksTest(unittest.TestCase):
    def test_books_are_ordered_from_least_copying(self):
        verify = verify_table(
            dict(aa_10k=280.0, aa_100k=290.0), dict(aa_10k=350.0, aa_100k=400.0)
        )
        plain = (100, [100, 0, 0, 0, 0, 0, 0, 0])
        copying = (300, [100, 0, 0, 0, 0, 0, 0, 25])
        per_context = dict(aa_10k=[copying, plain], aa_100k=[copying, plain])
        ordered = story.books(per_context, verify, 1.0, 7.0)
        self.assertEqual([b["book"] for b in ordered], [1, 0])
        self.assertAlmostEqual(ordered[0]["copy_share"], 0.0)
        # The plain book runs at 3,571 TPS at 10K with no extra time, 5.6%
        # fast, and enters the band at 2 µs, where 10K gives 3,546 TPS.
        self.assertEqual(ordered[0]["extra_us"], [2.0, 7.0])


class SpreadTest(unittest.TestCase):
    def test_spread_reports_whether_nvidia_lies_between_percentiles(self):
        rows = [(100 + i, [100, 0, 0, 0, 0, 0, 0, i % 5]) for i in range(40)]
        step = lambda rows, context: 100.0 * rows
        result = story.check_spread(rows, step, 0.0, np.random.default_rng(1))
        low, high = result["maximum_ratio_5_95"]
        self.assertLessEqual(low, high)
        self.assertEqual(
            result["nvidia_maximum_inside"],
            low <= story.NVIDIA_RATIOS["maximum"] <= high,
        )
        self.assertGreaterEqual(result["maximum_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
