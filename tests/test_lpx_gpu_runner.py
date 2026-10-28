import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts.run_lpx_gpu_study import (
    collect,
    owns_server,
    priority_one,
    pricing_ratio,
    server_command,
    smoke_tests,
    stop_server,
)


class GpuRunnerTest(unittest.TestCase):
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
