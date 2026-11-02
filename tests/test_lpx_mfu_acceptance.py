"""Measured acceptance keeps counter weights and precision differences explicit."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_mfu_acceptance import audit, project, summarize


class MeasuredAcceptanceTest(unittest.TestCase):
    def test_turn_mean_and_round_weighted_commit_lengths_differ(self):
        rows = [
            dict(tau=2.0, decode_tokens=4, drafts=2, quality_ok=True),
            dict(tau=3.0, decode_tokens=12, drafts=4, quality_ok=True),
        ]
        result = summarize(rows)
        self.assertEqual(result["mean_tau"], 2.5)
        self.assertEqual(result["pooled_tau"], 16 / 6)
        self.assertEqual(result["turns"], 2)
        with self.assertRaises(WorkloadValidationError):
            summarize([])

    def test_gpu_precision_is_not_silently_replaced_by_modeled_precision(self):
        program = dict(
            draft_tokens=3,
            draft_dtype="bf16",
            slots=3,
            chips=4,
            native_transformer=dict(
                proxy_model=dict(dtype="fp8", kv_cache_dtype="fp8")
            ),
            round_replay=dict(aggregate_tps=10.0, convergence={}),
            useful_work_per_committed_token=dict(total=1000),
        )
        measured = dict(
            config=dict(
                K=3,
                weight_precision="fp8",
                kv_precision="bf16",
                drafter_precision="bf16",
            ),
            source="fixture",
            sha256="fixture",
            style="quotes",
            turns=2,
            mean_tau=2.5,
            pooled_tau=16 / 6,
            all_quality_checks_passed=True,
        )
        result = project(program, measured, 1e6)
        self.assertEqual(result["precision_mismatches"], ["kv_precision"])
        self.assertEqual(result["aggregate_tps_from_turn_mean"], 25)
        self.assertAlmostEqual(result["mfu_percent_from_pooled_rounds"], 2 / 3)
        with self.assertRaises(WorkloadValidationError):
            project(
                program, {**measured, "config": {**measured["config"], "K": 1}}, 1e6
            )

    def test_raw_audit_retains_flagged_outputs_and_rejects_duplicate_counters(self):
        turn = dict(
            content="answer",
            token_ids=[1, 2, 3, 4, 5],
            completion_tokens=5,
            drafts=2,
            draft_tokens=6,
            accepted_tokens=2,
            accepted_per_position={"0": 1, "1": 1, "2": 0},
            emitted_steps=3,
            preemptions=0,
            finish_reason="length",
            turn=0,
        )
        request = dict(
            id="book/100k",
            set="aa-docs",
            meta=dict(context=100000, style="quotes", book="book"),
            config=dict(method="mtp", K=3, max_tokens_override=None),
            turns=[turn],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "outputs.jsonl"
            path.write_text(json.dumps(request) + "\n")
            result = audit(path)
            self.assertEqual(result[0]["turns"], 1)
            self.assertEqual(result[0]["pooled_tau"], 2)
            self.assertFalse(result[0]["all_quality_checks_passed"])
            path.write_text((json.dumps(request) + "\n") * 2)
            with self.assertRaises(WorkloadValidationError):
                audit(path)
            invalid = deepcopy(request)
            invalid["turns"][0]["emitted_steps"] = 7
            path.write_text(json.dumps(invalid) + "\n")
            with self.assertRaises(WorkloadValidationError):
                audit(path)


if __name__ == "__main__":
    unittest.main()
