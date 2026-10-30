import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import run_lpx_gpu_cc302 as collector
from scripts import run_lpx_gpu_followup as followup
from tests.test_lpx_gpu_followup import fixture


class GpuCc302Test(unittest.TestCase):
    def test_collection_keeps_uncapped_measurements_and_warmups_separate(self):
        with patch.object(collector.study, "run_script") as run:
            measured = collector.collect(Path("expanded.jsonl"), "plain", 0)
            arguments = run.call_args.args
            self.assertIn(Path("expanded.jsonl"), arguments)
            self.assertIn("cc302-plain", arguments)
            self.assertNotIn("--max-tokens", arguments)
            self.assertNotIn("--limit", arguments)
            warmup = collector.collect(Path("expanded.jsonl"), "ngram", 7, True)
            arguments = run.call_args.args
            self.assertIn("--max-tokens", arguments)
            self.assertIn("--limit", arguments)
            self.assertNotEqual(measured, warmup)

    def test_validation_accepts_explicit_request_file(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            row = fixture(work)
            original = work / "inputs/requests.jsonl"
            expanded = work / "inputs/expanded.jsonl"
            original.rename(expanded)
            output = work / "outputs.jsonl"
            output.write_text(json.dumps(row) + "\n")
            rows = followup.validate_outputs(
                work,
                output,
                [("speed-coding", "off")],
                "plain",
                0,
                requests_path=expanded,
            )
            self.assertEqual(len(rows), 1)

    def test_runner_does_not_recollect_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            output = work / "gpu-cc302-plain-outputs.jsonl"
            output.write_text("{}\n" * 302)
            with (
                patch.object(collector.study, "ARTIFACTS", work),
                patch.object(collector, "audit_inputs"),
                patch.object(collector, "validate") as validate,
                patch.object(collector.study, "start_server") as start,
                patch.object(collector.study, "stop_server") as stop,
            ):
                self.assertEqual(collector.run(work, Path("requests"), "plain"), 0)
                validate.assert_called_once()
                start.assert_not_called()
                stop.assert_not_called()

    def test_failure_preserves_status_and_releases_owned_server(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with (
                patch.object(collector.study, "ARTIFACTS", work),
                patch.object(collector, "audit_inputs"),
                patch.object(
                    collector.study,
                    "start_server",
                    side_effect=WorkloadValidationError("startup"),
                ),
                patch.object(collector.study, "stop_server") as stop,
            ):
                with self.assertRaises(WorkloadValidationError):
                    collector.run(work, Path("requests"), "all")
                stop.assert_called_once_with(work)
                state = json.loads(
                    (work / "gpu-cc302-collection-status.json").read_text()
                )
                self.assertEqual(state["status"], "failed")
                self.assertTrue(state["gpu_released"])
