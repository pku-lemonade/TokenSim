import unittest
from unittest import mock

from scripts import analyze_lpx_cc16k_peak as peak
from scripts import build_lpx_api_prompts as prompts


def completion_rows(count):
    return [
        dict(
            question_id=f"q{i:03d}",
            sub_category="code_completion" if i % 2 == 0 else "other",
            turns=[f"prompt {i}"],
            source="s",
            src_id=str(i),
        )
        for i in range(2 * count)
    ]


class SpeedCompletionTest(unittest.TestCase):
    def test_full_count_takes_every_completion_prompt(self):
        with mock.patch.object(
            prompts, "read_parquet", return_value=completion_rows(5)
        ):
            requests = list(prompts.speed_completion("unused", 5))
        self.assertEqual(
            sorted(r["meta"]["question_id"] for r in requests),
            [f"q{i:03d}" for i in range(0, 10, 2)],
        )

    def test_smaller_count_is_unchanged_by_the_option_default(self):
        with mock.patch.object(
            prompts, "read_parquet", return_value=completion_rows(5)
        ):
            first = [r["id"] for r in prompts.speed_completion("unused", 3)]
            second = [r["id"] for r in prompts.speed_completion("unused", 3)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


class PeakPricingTest(unittest.TestCase):
    step = staticmethod(lambda rows, context: 100.0 * rows)

    def test_completion_counts_requests_at_or_above_the_example(self):
        # 10 one-row steps of 100 µs give 10,000 TPS; 5 three-row steps of
        # 300 µs committing 20 tokens give 13,333 TPS.
        rows = [(10, [10]), (20, [0, 0, 5])]
        result = peak.completion(rows, self.step, 0.0)
        self.assertEqual(result["requests"], 2)
        self.assertAlmostEqual(result["maximum_tps"], 20 / (5 * 300) * 1e6)
        self.assertEqual(result["at_or_above_example"], 1)
        self.assertAlmostEqual(result["maximum_tau"], 4.0)

    def test_coding_window_is_none_when_no_extra_time_fits(self):
        rows = [(10, [10])] * 3
        too_slow = lambda rows, context: 300.0
        self.assertIsNone(peak.coding_window(rows, too_slow))

    def test_coding_window_brackets_the_published_median(self):
        rows = [(10, [10])] * 3
        slow = lambda rows, context: 150.0
        window = peak.coding_window(rows, slow)
        self.assertIsNotNone(window)
        for extra in window:
            speed = 1e6 / (150.0 + extra)
            self.assertLessEqual(abs(speed / 4767 - 1), 0.05)


if __name__ == "__main__":
    unittest.main()
