import copy
import json
from types import SimpleNamespace
import unittest

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_gpu_ngram import (
    compare,
    matching_rules,
    validate_server_command,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        return SimpleNamespace(ids=list(map(ord, text)))


def records():
    turn = dict(
        turn=0,
        content="abc",
        reasoning="",
        prompt_tokens=5,
        completion_tokens=4,
        drafts=0,
        draft_tokens=0,
        accepted_tokens=0,
        accepted_per_position={str(position): 0 for position in range(7)},
        emitted_steps=4,
        preemptions=0,
    )
    request = dict(
        id="coding/a",
        set="speed-coding",
        meta={},
        config=dict(method="ngram", K=7),
        turns=[turn],
    )
    exported = dict(
        id=request["id"],
        turn=0,
        meta={},
        messages=[
            dict(role="user", content="xyz"),
            dict(role="assistant", content="abc", reasoning=""),
        ],
    )
    return request, exported


class GpuNgramComparisonTest(unittest.TestCase):
    def test_server_command_requires_recorded_lookup_bounds_and_batch_one(self):
        config = dict(
            method="ngram",
            num_speculative_tokens=7,
            prompt_lookup_min=1,
            prompt_lookup_max=3,
        )
        command = dict(
            argv=[
                "vllm",
                "--max-num-seqs",
                "1",
                "--speculative-config",
                json.dumps(config),
            ]
        )
        self.assertEqual(validate_server_command(command), "unknown")
        command["environment"] = {"VLLM_USE_V2_MODEL_RUNNER": "1"}
        self.assertEqual(validate_server_command(command), "v2-gpu")
        command["environment"]["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        self.assertEqual(validate_server_command(command), "v1-cpu")
        config["prompt_lookup_min"] = 2
        command["argv"][-1] = json.dumps(config)
        with self.assertRaises(WorkloadValidationError):
            validate_server_command(command)
        with self.assertRaises(WorkloadValidationError):
            validate_server_command(dict(argv=["vllm"]))

    def test_identity_join_and_distinct_token_units(self):
        request, exported = records()
        another = copy.deepcopy(request)
        another_export = copy.deepcopy(exported)
        another["id"] = another_export["id"] = "coding/b"
        result = compare(
            [another, request], [exported, another_export], CharacterTokenizer()
        )
        self.assertEqual(result["sets"][0]["turn_count"], 2)
        self.assertTrue(result["sets"][0]["expectation_within_0_03"])
        turn = result["turns"][0]
        self.assertEqual(turn["id"], "coding/a")
        self.assertEqual(turn["native"]["decode_tokens"], 3)
        self.assertEqual(turn["replay"]["output_tokens"], 5)
        self.assertEqual(turn["native"]["tau"], 1)
        self.assertEqual(turn["replay"]["tau"], 1)
        self.assertNotIn("messages", turn)
        self.assertNotIn("content", turn)

    def test_runner_specific_rules_preserve_numerical_comparison(self):
        request, exported = records()
        results = [
            compare([request], [exported], CharacterTokenizer(), implementation)
            for implementation in ("unknown", "v1-cpu", "v2-gpu")
        ]
        self.assertEqual(results[0]["sets"], results[1]["sets"])
        self.assertEqual(results[1]["sets"], results[2]["sets"])
        self.assertIn("earliest", results[1]["differences"][0])
        self.assertNotIn("earliest", " ".join(results[2]["differences"]))
        self.assertIn("last sampled token", results[2]["differences"][1])
        self.assertEqual(results[2]["native_implementation"], "v2-gpu")
        with self.assertRaises(WorkloadValidationError):
            matching_rules("guessed")

    def test_mismatched_or_duplicate_exports_fail(self):
        request, exported = records()
        for exports in ([], [exported, exported]):
            with self.assertRaises(WorkloadValidationError):
                compare([request], exports, CharacterTokenizer())
        exported["messages"][1]["content"] = "wrong"
        with self.assertRaises(WorkloadValidationError):
            compare([request], [exported], CharacterTokenizer())

    def test_foreign_configuration_and_preemptions_fail(self):
        request, exported = records()
        request["config"]["method"] = "mtp"
        with self.assertRaises(WorkloadValidationError):
            compare([request], [exported], CharacterTokenizer())
        request["config"]["method"] = "ngram"
        request["turns"][0]["preemptions"] = 1
        with self.assertRaises(WorkloadValidationError):
            compare([request], [exported], CharacterTokenizer())


if __name__ == "__main__":
    unittest.main()
