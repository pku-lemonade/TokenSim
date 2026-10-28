import copy
import unittest

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_gpu_acceptance import (
    aa_pricing,
    coding_speed,
    mtp_steps,
    paired,
    speed_summary,
    summarize,
)
from scripts.analyze_lpx_published_conditions import LAYOUT
from scripts.lpx.gpu_metrics import acceptance


def output(book="book-a", context=10000, drafts=2, accepted=3):
    return dict(
        id=f"aa-docs/{book}/{context // 1000}k/summary",
        set="aa-docs",
        thinking=True,
        meta=dict(book=book, context=context, style="summary"),
        config=dict(method="mtp", K=3, weight_precision="bf16", kv_precision="bf16"),
        turns=[
            dict(
                turn=0,
                prompt_tokens=context,
                completion_tokens=1 + drafts + accepted,
                drafts=drafts,
                draft_tokens=3 * drafts,
                accepted_tokens=accepted,
                accepted_per_position={
                    "0": min(accepted, drafts),
                    "1": max(accepted - drafts, 0),
                    "2": 0,
                },
                output_tps=50,
            )
        ],
    )


def speculative():
    return dict(
        rows=[
            dict(
                layout_name=LAYOUT,
                attention_mfu=0.5,
                draft_tokens=3,
                drafter=drafter,
                step_us=dict(
                    aa_10k=300 * factor, aa_100k=350 * factor, coding_16k=310 * factor
                ),
                draft_us=dict(
                    aa_10k=30 * factor, aa_100k=35 * factor, coding_16k=31 * factor
                ),
            )
            for drafter, factor in (("mtp-0.05", 1), ("mtp-0.10", 2))
        ]
    )


class GpuAcceptanceTest(unittest.TestCase):
    def test_mean_tau_and_pooled_tau_are_distinct(self):
        measured = summarize([output(), output("book-b", drafts=4, accepted=0)])[0]
        self.assertEqual(measured["turn_count"], 2)
        self.assertEqual(measured["tau"], 1.75)
        self.assertEqual(measured["pooled_tau"], 1.5)
        self.assertEqual(measured["per_row"], [[5, [0, 0, 0, 2]], [4, [0, 0, 0, 4]]])
        self.assertAlmostEqual(measured["per_position_acceptance"]["0"], 2 / 6)

    def test_configurations_and_missing_timing_stay_separate(self):
        original = output()
        quantized = copy.deepcopy(original)
        quantized["config"]["weight_precision"] = "fp8"
        quantized["turns"][0]["output_tps"] = None
        results = summarize([original, quantized])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["untimed_turns"], 1)
        self.assertIsNone(results[1]["gpu_speed"])
        with self.assertRaises(WorkloadValidationError):
            summarize([original, original])

    def test_ngram_does_not_claim_fixed_verification_rows(self):
        measured = output()
        measured["config"]["method"] = "ngram"
        result = summarize([measured])[0]
        self.assertNotIn("per_row", result)
        self.assertNotIn("per_row", result["turns"][0])

    def test_pairing_joins_book_identity_and_reports_missing_books(self):
        low = output()
        high = output(context=100000, accepted=0)
        results = summarize([output("book-b"), high, low, output("book-c", 100000)])
        result = paired(results)[0]
        self.assertEqual(result["paired_books"], 1)
        self.assertEqual(result["only_10k"], ["book-b"])
        self.assertEqual(result["only_100k"], ["book-c"])
        expected = (
            acceptance(high["turns"][0], 3)["equivalent_acceptance"]
            - acceptance(low["turns"][0], 3)["equivalent_acceptance"]
        )
        self.assertAlmostEqual(result["mean_difference"], expected)
        self.assertEqual(result["higher_at_100k"], 0)

    def test_aa_prices_context_specific_tau_and_both_costs(self):
        results = summarize([output(), output(context=100000, accepted=0)])
        prices = aa_pricing(results, speculative(), ratio=2)
        self.assertEqual(len(prices), 2)
        self.assertAlmostEqual(prices[0]["aa_tps"]["aa_10k"], 2.5 / 300 * 1e6 / 2)
        self.assertAlmostEqual(prices[0]["aa_tps"]["aa_100k"], 1 / 350 * 1e6 / 2)
        self.assertAlmostEqual(
            prices[1]["aa_tps"]["aa_10k"], prices[0]["aa_tps"]["aa_10k"] / 2
        )
        self.assertIn("extra_us", prices[0])
        self.assertEqual(aa_pricing(summarize([output()]), speculative(), 1), [])
        with self.assertRaises(WorkloadValidationError):
            aa_pricing(results, speculative(), 0)

    def test_missing_or_duplicate_simulation_rows_fail(self):
        with self.assertRaises(WorkloadValidationError):
            mtp_steps(speculative(), 7, "mtp-0.05")
        duplicate = speculative()
        duplicate["rows"].append(duplicate["rows"][0])
        with self.assertRaises(WorkloadValidationError):
            mtp_steps(duplicate, 3, "mtp-0.05")

    def test_coding_prices_observed_context_and_drafter_time(self):
        turn = dict(decode_tokens=20, prompt_tokens=100, per_row=[20, [0, 0, 0, 10]])
        observed = []

        def step(rows, context):
            observed.append((rows, context))
            return 150

        speed = coding_speed(turn, step, lambda rows, context: 50, 3)
        self.assertEqual(speed, 10000)
        self.assertEqual(observed, [(4, 110)])
        plain = coding_speed(turn, step, lambda rows, context: 0, 0)
        self.assertAlmostEqual(plain, 1e6 / 150)
        with self.assertRaises(WorkloadValidationError):
            coding_speed(turn, lambda rows, context: -1, lambda rows, context: 0, 3)
        turn["per_row"] = [20, [1, 0, 0, 9]]
        with self.assertRaises(WorkloadValidationError):
            coding_speed(turn, step, lambda rows, context: 50, 3)

    def test_speed_summary_percentiles_and_failures(self):
        result = speed_summary([1, 2, 3, 4, 5])
        self.assertEqual(result["median_tps"], 3)
        self.assertAlmostEqual(result["p80_tps"], 4.2)
        self.assertAlmostEqual(result["p80_ratio"], 1.4)
        self.assertAlmostEqual(result["maximum_ratio"], 5 / 3)
        for invalid in ([], [0], [-1], [float("nan")]):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(WorkloadValidationError),
            ):
                speed_summary(invalid)


if __name__ == "__main__":
    unittest.main()
