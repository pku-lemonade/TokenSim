import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts.run_lpx_gpu_study import (
    artifact,
    collect,
    owns_server,
    priority_one,
    pricing_ratio,
    server_command,
    server_matches,
    smoke_tests,
    stop_server,
    validate_document_control,
)


class GpuRunnerTest(unittest.TestCase):
    def test_document_gate_requires_completed_uncapped_nonrepeating_mtp(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            row = dict(
                config=dict(method="mtp", K=3, max_tokens_override=None),
                turns=[
                    dict(content="answer", finish_reason="stop", token_ids=[1, 2, 3])
                ],
            )
            with patch("scripts.run_lpx_gpu_study.ARTIFACTS", work):
                validate_document_control("original")
                with self.assertRaises(WorkloadValidationError):
                    validate_document_control("triton")
                path = work / "gpu-triton-mtp-k3-smoke.jsonl"
                path.write_text(json.dumps(row) + "\n")
                validate_document_control("triton")
                for field, value in (
                    ("finish_reason", "length"),
                    ("content", ""),
                    ("token_ids", [1] * 128),
                ):
                    changed = json.loads(json.dumps(row))
                    changed["turns"][0][field] = value
                    path.write_text(json.dumps(changed) + "\n")
                    with self.assertRaises(WorkloadValidationError):
                        validate_document_control("triton")
                row["config"]["max_tokens_override"] = 512
                path.write_text(json.dumps(row) + "\n")
                with self.assertRaises(WorkloadValidationError):
                    validate_document_control("triton")

    def test_profile_paths_and_labels_are_separate(self):
        self.assertNotEqual(
            artifact("outputs.jsonl"), artifact("outputs.jsonl", "triton")
        )
        with self.assertRaises(WorkloadValidationError):
            artifact("outputs.jsonl", "unknown")
        with patch("scripts.run_lpx_gpu_study.run_script") as run:
            collect(Path("/tmp/work"), 3, "aa-docs", profile="triton")
        arguments = run.call_args.args
        self.assertIn("t1-triton-mtp-k3", arguments)
        self.assertIn(artifact("mtp-k3-outputs.jsonl", "triton"), arguments)

    def test_server_reuse_requires_command_and_environment_match(self):
        state = dict(
            pid=os.getpid(), argv=["vllm"], environment={"HF_HUB_OFFLINE": "1"}
        )
        with (
            patch("scripts.run_lpx_gpu_study.owns_server", return_value=True),
            patch(
                "scripts.run_lpx_gpu_study.Path.read_bytes",
                return_value=b"HF_HUB_OFFLINE=1\0",
            ),
        ):
            self.assertTrue(server_matches(state, ["vllm"], state["environment"]))
            self.assertFalse(
                server_matches(
                    state,
                    ["vllm", "--attention-config", "triton"],
                    state["environment"],
                )
            )
            self.assertFalse(server_matches(state, ["vllm"], {"HF_HUB_OFFLINE": "0"}))
        with (
            patch("scripts.run_lpx_gpu_study.owns_server", return_value=True),
            patch(
                "scripts.run_lpx_gpu_study.Path.read_bytes",
                return_value=b"HF_HUB_OFFLINE=0\0",
            ),
        ):
            self.assertFalse(server_matches(state, ["vllm"], state["environment"]))

    def test_triton_prerequisites_do_not_reuse_original_results(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "gpu-runtime-estimates.json").touch()
            (work / "gpu-smoke-comparison.json").touch()
            with patch("scripts.run_lpx_gpu_study.ARTIFACTS", work):
                with self.assertRaises(WorkloadValidationError):
                    priority_one(work, profile="triton")

    def test_triton_smoke_propagates_profile_to_servers_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "gpu-triton-plain-coding-smoke.jsonl").write_text("{}\n" * 10)
            with (
                patch("scripts.run_lpx_gpu_study.ARTIFACTS", work),
                patch("scripts.run_lpx_gpu_study.start_server") as start,
                patch("scripts.run_lpx_gpu_study.collect") as measured,
                patch("scripts.run_lpx_gpu_study.run_script") as analyze,
            ):
                smoke_tests(work, profile="triton")
            self.assertEqual([call.args[1] for call in start.call_args_list], [1, 3])
            for call in start.call_args_list + measured.call_args_list:
                self.assertEqual(call.kwargs["profile"], "triton")
            self.assertEqual(measured.call_args.args[1:], (3, "aa-docs"))
            self.assertNotIn("max_tokens", measured.call_args.kwargs)
            paths = [
                value for value in analyze.call_args.args if isinstance(value, Path)
            ]
            self.assertTrue(
                all(
                    "gpu-triton-" in path.name
                    for path in paths
                    if path.suffix == ".jsonl"
                )
            )
            self.assertIn(work / "gpu-triton-smoke-comparison.json", paths)

    def test_warmup_tags_prevent_reusing_a_previous_server_warmup(self):
        with patch("scripts.run_lpx_gpu_study.run_script") as run:
            paths = []
            for tag in (100, 200):
                collect(
                    Path("/tmp/work"),
                    3,
                    "aa-docs",
                    limit=1,
                    max_tokens=128,
                    phase="warmup",
                    warmup_tag=tag,
                )
                arguments = run.call_args.args
                paths.append(arguments[arguments.index("--outputs") + 1])
            self.assertNotEqual(*paths)
            with self.assertRaises(WorkloadValidationError):
                collect(Path("/tmp/work"), 3, "aa-docs", warmup_tag=100)
            with self.assertRaises(WorkloadValidationError):
                collect(
                    Path("/tmp/work"),
                    3,
                    "aa-docs",
                    phase="warmup",
                    warmup_tag="../escape",
                )

    def test_priority_sweep_collects_both_coding_configurations_before_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "gpu-runtime-estimates.json").touch()
            (work / "gpu-smoke-comparison.json").touch()
            with (
                patch("scripts.run_lpx_gpu_study.ARTIFACTS", work),
                patch("scripts.run_lpx_gpu_study.start_server"),
                patch("scripts.run_lpx_gpu_study.stop_server"),
                patch("scripts.run_lpx_gpu_study.collect") as measured,
                patch("scripts.run_lpx_gpu_study.run_script"),
                patch(
                    "scripts.run_lpx_gpu_study.measure_ratio",
                    return_value=dict(gemma_tokens=1006, o200k_tokens=1000),
                ),
            ):
                priority_one(work)
            datasets = [
                (call.args[1], call.args[2])
                for call in measured.call_args_list
                if call.kwargs.get("phase") != "warmup"
            ]
            self.assertEqual(
                datasets,
                [
                    (3, "speed-coding"),
                    (7, "speed-coding"),
                    (3, "aa-docs"),
                    (7, "aa-docs"),
                ],
            )

    def test_token_ratio_uses_pooled_counts_and_recorded_threshold(self):
        self.assertEqual(
            pricing_ratio([dict(gemma_tokens=1006, o200k_tokens=1000)]), 1.006
        )
        self.assertEqual(
            pricing_ratio([dict(gemma_tokens=1016, o200k_tokens=1000)]), 1.006
        )
        self.assertEqual(
            pricing_ratio([dict(gemma_tokens=1050, o200k_tokens=1000)]), 1.05
        )
        self.assertEqual(
            pricing_ratio(
                [
                    dict(gemma_tokens=100, o200k_tokens=100),
                    dict(gemma_tokens=230, o200k_tokens=200),
                ]
            ),
            1.1,
        )
        with self.assertRaises(WorkloadValidationError):
            pricing_ratio([])

    def test_server_command_preserves_plain_settings_and_pins_assistant(self):
        base = ["vllm", "serve", "gemma", "--max-num-seqs", "1"]
        command = server_command(base, 3)
        self.assertEqual(command[: len(base)], base)
        config = json.loads(command[-1])
        self.assertEqual(config["method"], "mtp")
        self.assertEqual(config["num_speculative_tokens"], 3)
        self.assertEqual(len(config["revision"]), 40)
        self.assertEqual(base, ["vllm", "serve", "gemma", "--max-num-seqs", "1"])
        with self.assertRaises(WorkloadValidationError):
            server_command(base, 2)
        with self.assertRaises(WorkloadValidationError):
            server_command(command, 7)

    def test_collection_separates_capped_smoke_from_full_measurements(self):
        with patch("scripts.run_lpx_gpu_study.run_script") as run:
            collect(
                Path("/tmp/work"), 1, "aa-docs", limit=1, max_tokens=128, phase="smoke"
            )
            arguments = run.call_args.args
            self.assertIn("--max-tokens", arguments)
            self.assertIn("--context", arguments)
            self.assertIn("t0-mtp-k1", arguments)
            collect(Path("/tmp/work"), 3, "speed-coding")
            arguments = run.call_args.args
            self.assertNotIn("--max-tokens", arguments)
            self.assertIn("t1-mtp-k3", arguments)
            self.assertIn("off", arguments)
            measurement_path = arguments[arguments.index("--outputs") + 1]
            collect(Path("/tmp/work"), 3, "speed-coding", limit=10, phase="smoke")
            arguments = run.call_args.args
            smoke_path = arguments[arguments.index("--outputs") + 1]
            self.assertNotEqual(smoke_path, measurement_path)
            self.assertIn("t0-mtp-k3", arguments)
            self.assertNotIn("--context", arguments)
            collect(
                Path("/tmp/work"), 3, "aa-docs", limit=1, max_tokens=128, phase="warmup"
            )
            arguments = run.call_args.args
            warmup_path = arguments[arguments.index("--outputs") + 1]
            self.assertNotIn(warmup_path, (measurement_path, smoke_path))
            self.assertIn("t1-mtp-k3-warmup", arguments)
            self.assertIn("--context", arguments)
            with self.assertRaises(WorkloadValidationError):
                collect(Path("/tmp/work"), 3, "aa-docs", phase="unknown")

    def test_stop_refuses_foreign_process(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            state = dict(pid=os.getpid(), argv=["not-this-process"])
            (work / "server-process.json").write_text(json.dumps(state))
            self.assertFalse(owns_server(state))
            with patch("scripts.run_lpx_gpu_study.os.killpg") as kill:
                with self.assertRaises(WorkloadValidationError):
                    stop_server(work)
                kill.assert_not_called()

    def test_stop_clears_stale_state_after_gpu_and_port_are_free(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            state = work / "server-process.json"
            state.write_text(json.dumps(dict(pid=999999999, argv=["vllm"])))
            with (
                patch("scripts.run_lpx_gpu_study.healthy", return_value=False),
                patch(
                    "scripts.run_lpx_gpu_study.subprocess.check_output",
                    return_value="0\n",
                ),
            ):
                stop_server(work)
            self.assertFalse(state.exists())

    def test_unrecorded_server_and_missing_prerequisites_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with patch("scripts.run_lpx_gpu_study.healthy", return_value=True):
                with self.assertRaises(WorkloadValidationError):
                    stop_server(work)
            with patch("scripts.run_lpx_gpu_study.ARTIFACTS", work):
                with self.assertRaises(WorkloadValidationError):
                    priority_one(work)
                with self.assertRaises(WorkloadValidationError):
                    smoke_tests(work)


if __name__ == "__main__":
    unittest.main()
