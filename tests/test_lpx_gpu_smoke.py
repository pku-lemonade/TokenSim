import copy
from types import SimpleNamespace
import unittest

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_gpu_smoke import compare, estimate_runtime, first_difference


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(character) for character in text])


def smoke_output(method="plain"):
    return dict(
        id="coding/1",
        config=dict(
            method=method,
            K=3 if method == "mtp" else 0,
            requests_sha256="requests",
            environment_sha256="environment",
            weight_precision="bf16",
            kv_precision="bf16",
            model="gemma",
            max_tokens_override=None,
        ),
        turns=[
            dict(
                turn=0,
                reasoning="",
                content="hello",
                finish_reason="stop",
                completion_tokens=6,
                drafts=2,
                draft_tokens=6,
                accepted_tokens=3,
                accepted_per_position={"0": 2, "1": 1, "2": 0},
                output_tps=10,
            )
        ],
    )


class GpuSmokeTest(unittest.TestCase):
    def test_first_difference_and_prefix_length(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1], [1, 2]), 1)
        self.assertEqual(first_difference([], [1]), 0)

    def test_comparison_counts_requests_and_retokenized_difference(self):
        plain, mtp = smoke_output(), smoke_output("mtp")
        result = compare([plain], [mtp], CharacterTokenizer())
        self.assertEqual(result["identical_requests"], 1)
        self.assertEqual(result["counter_identity_residuals"], {0: 1})
        mtp["turns"][0]["content"] = "help!"
        result = compare([plain], [mtp], CharacterTokenizer())
        self.assertEqual(result["identical_requests"], 0)
        self.assertEqual(
            result["requests"][0]["turns"][0]["first_different_retokenized_position"], 3
        )

    def test_comparison_rejects_mismatched_configuration_and_ids(self):
        plain, mtp = smoke_output(), smoke_output("mtp")
        mtp["config"]["max_tokens_override"] = 128
        with self.assertRaises(WorkloadValidationError):
            compare([plain], [mtp], CharacterTokenizer())
        mtp = smoke_output("mtp")
        mtp["id"] = "different"
        with self.assertRaises(WorkloadValidationError):
            compare([plain], [mtp], CharacterTokenizer())
        with self.assertRaises(WorkloadValidationError):
            compare([plain, plain], [mtp], CharacterTokenizer())

    def test_comparison_checks_all_turns(self):
        plain, mtp = smoke_output(), smoke_output("mtp")
        for row in (plain, mtp):
            another = copy.deepcopy(row["turns"][0])
            another["turn"] = 1
            row["turns"].append(another)
        mtp["turns"][1]["finish_reason"] = "length"
        result = compare([plain], [mtp], CharacterTokenizer())
        self.assertEqual(result["request_pairs"], 1)
        self.assertEqual(result["identical_requests"], 0)
        self.assertEqual(result["counter_identity_residuals"], {0: 2})

    def test_native_token_difference_is_distinct_from_text_difference(self):
        plain, mtp = smoke_output(), smoke_output("mtp")
        plain["turns"][0]["token_ids"] = [2, 3, 4]
        mtp["turns"][0]["token_ids"] = [2, 9, 4]
        result = compare([plain], [mtp], CharacterTokenizer())
        turn = result["requests"][0]["turns"][0]
        self.assertEqual(turn["first_different_token_position"], 1)
        self.assertIsNone(turn["first_different_retokenized_position"])
        self.assertEqual(result["identical_requests"], 0)

    def test_runtime_uses_turn_lengths_and_configuration_count(self):
        requests, reference = [], []
        for dataset, thinking in (
            ("aa-docs", True),
            ("speed-coding", True),
            ("speed-coding", False),
            ("speed-cc16k", False),
        ):
            identity = f"{dataset}/{thinking}"
            requests.append(
                dict(
                    id=identity,
                    set=dataset,
                    reasoning=thinking,
                    turns=["question", "follow-up"],
                    max_tokens=101,
                )
            )
            reference.append(
                dict(
                    id=identity,
                    turns=[
                        dict(usage=dict(completion_tokens=21)),
                        dict(usage=dict(completion_tokens=31)),
                    ],
                )
            )
        result = estimate_runtime(requests, reference, smoke_output())
        measured = result["tasks"][0]
        self.assertEqual(measured["requests"], 2)
        self.assertEqual(measured["reference_decode_tokens"], 100)
        self.assertAlmostEqual(measured["estimated_generation_hours"], 10 / 3600)
        self.assertAlmostEqual(
            measured["generation_hours_if_all_outputs_hit_cap"], 40 / 3600
        )
        self.assertEqual(result["tasks"][-1]["reference_decode_tokens"], 1020)
        with self.assertRaises(WorkloadValidationError):
            estimate_runtime(requests, [], smoke_output())


if __name__ == "__main__":
    unittest.main()
