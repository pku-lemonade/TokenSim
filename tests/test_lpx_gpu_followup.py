import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import run_lpx_gpu_followup as followup
from scripts import run_lpx_gpu_study as study


def fixture(work):
    request = dict(
        id="coding/a", set="speed-coding", reasoning=False, meta={}, turns=["prompt"]
    )
    inputs = work / "inputs/requests.jsonl"
    inputs.parent.mkdir()
    inputs.write_text(json.dumps(request) + "\n")
    row = dict(
        id=request["id"],
        set=request["set"],
        thinking=False,
        meta={},
        config=dict(
            method="plain",
            K=0,
            weight_precision="bf16",
            kv_precision="bf16",
            max_tokens_override=None,
            requests_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        ),
        turns=[
            dict(
                turn=0,
                completion_tokens=4,
                emitted_steps=4,
                preemptions=0,
                content="answer",
                finish_reason="stop",
                token_ids=[1, 2, 3, 4],
            )
        ],
    )
    return row


class GpuFollowupTest(unittest.TestCase):
    def test_commands_support_plain_ngram_and_remaining_draft_length(self):
        base = ["vllm", "serve", "target"]
        self.assertEqual(study.server_command(base, 0, method="plain"), base)
        ngram = json.loads(study.server_command(base, 7, method="ngram")[-1])
        self.assertEqual(
            ngram,
            dict(
                method="ngram",
                num_speculative_tokens=7,
                prompt_lookup_min=1,
                prompt_lookup_max=3,
            ),
        )
        self.assertNotIn("model", ngram)
        self.assertEqual(
            json.loads(study.server_command(base, 5)[-1])["num_speculative_tokens"], 5
        )
        for method, draft_length in (
            ("plain", 3),
            ("ngram", 3),
            ("mtp", 0),
            ("invalid", 7),
        ):
            with self.assertRaises(WorkloadValidationError):
                study.server_command(base, draft_length, method=method)

    def test_plan_covers_only_remaining_mtp_combinations(self):
        tasks = followup.plan("remaining-mtp")
        self.assertEqual([row[1] for row in tasks], [1, 5, 3, 7])
        self.assertTrue(all(followup.DOCS in row[2] for row in tasks[:2]))
        self.assertTrue(
            all(
                followup.DOCS not in row[2] and followup.CODING not in row[2]
                for row in tasks[2:]
            )
        )
        self.assertEqual(followup.plan("plain"), [("plain", 0, [followup.CODING])])
        with self.assertRaises(WorkloadValidationError):
            followup.plan("unknown")

    def test_ngram_server_requires_explicit_v2_environment_when_reusing(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "server-process.json").touch()
            base = dict(argv=["vllm", "serve", "target"], environment={})
            with (
                patch.object(
                    study, "read_json", side_effect=[base, dict(draft_length=7)]
                ),
                patch.object(study, "server_matches", return_value=True) as matches,
                patch.object(study, "healthy", return_value=True),
            ):
                study.start_server(work, 7, profile="triton", method="ngram")
            self.assertEqual(matches.call_args.args[2]["VLLM_USE_V2_MODEL_RUNNER"], "1")

    def test_fp8_comparison_uses_mean_tau_and_keeps_group_differences(self):
        summaries = []
        for taus in ((2, 2.5, 3, 3.5), (2.2, 2.5, 3, 3.5)):
            summaries.append(
                [
                    dict(
                        data=f"aa-docs-{index}",
                        turn_count=16,
                        tau=tau,
                        equivalent_acceptance=followup.equivalent_acceptance(tau, 3),
                        quality={},
                    )
                    for index, tau in enumerate(taus)
                ]
            )
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for name in (
                "gpu-triton-mtp-k3-outputs.jsonl",
                "gpu-triton-fp8-mtp-k3-outputs.jsonl",
            ):
                (work / name).write_text("[]")
            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(followup, "validate_outputs", return_value=[]),
                patch.object(followup, "summarize", side_effect=summaries),
            ):
                followup.compare_fp8(work)
            result = json.loads((work / "gpu-triton-fp8-comparison.json").read_text())
            self.assertAlmostEqual(
                result["combined"]["fp8"], followup.equivalent_acceptance(2.8, 3)
            )
            self.assertTrue(result["combined"]["within_0_03"])
            self.assertFalse(result["groups"][0]["within_0_03"])
            self.assertEqual(len(result["sources"]), 2)

    def test_warmups_controls_and_measurements_have_separate_paths(self):
        with patch.object(study, "run_script") as run:
            paths = [
                followup.collect(Path("/tmp/work"), "plain", 0, followup.DOCS, phase)
                for phase in ("warmup", "warmup", "control", "measurement")
            ]
        self.assertEqual(len(set(paths)), 4)
        calls = run.call_args_list
        self.assertIn("--max-tokens", calls[0].args)
        self.assertNotIn("--max-tokens", calls[2].args)
        self.assertNotIn("--limit", calls[3].args)
        self.assertNotIn(
            paths[3],
            (
                study.artifact("mtp-k3-outputs.jsonl", "triton"),
                study.artifact("plain-coding-smoke.jsonl", "triton"),
            ),
        )

    def test_full_coverage_requires_metadata_precision_and_every_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            row = fixture(work)
            output = work / "outputs.jsonl"

            def check(rows):
                output.write_text("".join(json.dumps(item) + "\n" for item in rows))
                return followup.validate_outputs(
                    work, output, [followup.CODING], "plain", 0
                )

            self.assertEqual(check([row]), [row])
            for rows in ([], [row, row]):
                with self.assertRaises(WorkloadValidationError):
                    check(rows)
            for field, value in (
                ("meta", {"different": True}),
                ("turns", []),
                ("thinking", True),
            ):
                changed = copy.deepcopy(row)
                changed[field] = value
                with self.assertRaises(WorkloadValidationError):
                    check([changed])
            for field, value in (
                ("weight_precision", "fp8"),
                ("max_tokens_override", 128),
                ("requests_sha256", "wrong"),
            ):
                changed = copy.deepcopy(row)
                changed["config"][field] = value
                with self.assertRaises(WorkloadValidationError):
                    check([changed])

    def test_quality_flags_are_retained_for_measurements_but_controls_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            row = fixture(work)
            row["turns"][0]["content"] = ""
            output = work / "outputs.jsonl"
            output.write_text(json.dumps(row) + "\n")
            followup.validate_outputs(work, output, [followup.CODING], "plain", 0)
            validation = json.loads((work / "outputs-validation.json").read_text())
            self.assertEqual(validation["empty_answers"], 1)
            with self.assertRaises(WorkloadValidationError):
                followup.validate_control(output, "plain", 0)

    def test_counter_identity_and_preemptions_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            row = fixture(Path(directory))
            followup.validate_counters([row], "plain", 0)
            for field, value in (("preemptions", 1), ("emitted_steps", 3)):
                changed = copy.deepcopy(row)
                changed["turns"][0][field] = value
                with self.assertRaises(WorkloadValidationError):
                    followup.validate_counters([changed], "plain", 0)
            with self.assertRaises(WorkloadValidationError):
                followup.validate_counters([row], "mtp", 3)

    def test_failed_control_stops_owned_server_without_collecting_full_set(self):
        with (
            patch.object(study, "validate_document_control"),
            patch.object(study, "start_server"),
            patch.object(study, "read_json", return_value={}),
            patch.object(study, "write_json"),
            patch.object(
                followup, "collect", return_value=Path("control.jsonl")
            ) as collect,
            patch.object(
                followup, "validate_control", side_effect=ValueError("bad control")
            ),
            patch.object(study, "stop_server") as stop,
            patch.object(followup, "analyze") as analyze,
        ):
            with self.assertRaisesRegex(ValueError, "bad control"):
                followup.run_stage(Path("/tmp/work"), "plain")
        self.assertEqual(
            [call.args[-1] for call in collect.call_args_list], ["warmup", "control"]
        )
        stop.assert_called_once()
        analyze.assert_not_called()

    def test_busy_gpu_and_duplicate_queue_do_not_start_servers(self):
        for lock_name in ("runner.lock", "followup.lock"):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                with (
                    (work / lock_name).open("w") as lock,
                    patch.object(study, "ARTIFACTS", work),
                    patch.object(followup, "run_stage") as run,
                ):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        followup.main(["--work-dir", str(work)])
                    run.assert_not_called()

    def test_failed_fp8_predecessor_prevents_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "gpu-triton-fp8-status.json").write_text('{"status":"failed"}')
            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(followup, "run_stage") as run,
            ):
                with self.assertRaisesRegex(WorkloadValidationError, "predecessor"):
                    followup.main(["--work-dir", str(work), "--after-fp8"])
                run.assert_not_called()
            state = json.loads((work / "gpu-triton-followup-status.json").read_text())
            self.assertEqual(state["status"], "failed")

    def test_all_stages_are_recorded_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(followup, "run_stage") as run,
            ):
                self.assertEqual(followup.main(["--work-dir", str(work)]), 0)
            self.assertEqual(
                [call.args[1] for call in run.call_args_list],
                ["plain", "ngram", "remaining-mtp"],
            )
            state = json.loads((work / "gpu-triton-followup-status.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(
                state["completed_stages"], ["plain", "ngram", "remaining-mtp"]
            )


if __name__ == "__main__":
    unittest.main()
