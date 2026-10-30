import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import run_lpx_gpu_fp8 as fp8
from scripts import run_lpx_gpu_study as study


class GpuFp8Test(unittest.TestCase):
    def test_fp8_target_keeps_separate_unquantized_assistant(self):
        command = study.server_command(["vllm", "serve", "target"], 3, "fp8")
        self.assertEqual(command[command.index("--quantization") + 1], "fp8")
        speculative = json.loads(command[command.index("--speculative-config") + 1])
        self.assertEqual(speculative["model"], study.ASSISTANT)
        self.assertIsNone(speculative.get("quantization"))
        self.assertNotIn("--quantization", study.server_command(["vllm"], 3))
        for base, precision in ((["vllm"], "int8"), (["--quantization", "fp8"], "fp8")):
            with self.assertRaises(WorkloadValidationError):
                study.server_command(base, 3, precision)

    def test_fp8_artifacts_and_precision_do_not_mix_with_bf16(self):
        self.assertNotEqual(
            study.artifact("mtp-k3-outputs.jsonl", fp8.PROFILE),
            study.artifact("mtp-k3-outputs.jsonl", "triton"),
        )
        with patch.object(study, "run_script") as run:
            study.collect(Path("/tmp/work"), 3, "aa-docs", profile=fp8.PROFILE)
        arguments = run.call_args.args
        self.assertEqual(arguments[arguments.index("--weight-precision") + 1], "fp8")
        self.assertIn("t1-triton-fp8-mtp-k3", arguments)

    def test_failed_fp8_control_prevents_full_collection(self):
        with (
            patch.object(
                study,
                "validate_document_control",
                side_effect=[None, ValueError("unhealthy")],
            ),
            patch.object(study, "start_server"),
            patch.object(study, "collect") as collect,
            patch.object(study, "measure_ratio") as measure,
        ):
            with self.assertRaisesRegex(ValueError, "unhealthy"):
                fp8.collect_fp8(Path("/tmp/work"))
        self.assertEqual(len(collect.call_args_list), 2)
        self.assertTrue(
            all(call.kwargs["limit"] == 1 for call in collect.call_args_list)
        )
        measure.assert_not_called()

    def test_success_collects_uncapped_documents_then_prices_separately(self):
        with (
            patch.object(study, "validate_document_control") as gate,
            patch.object(study, "start_server"),
            patch.object(study, "collect") as collect,
            patch.object(
                study,
                "measure_ratio",
                return_value=dict(gemma_tokens=1006, o200k_tokens=1000),
            ),
            patch.object(study, "run_script") as analyze,
        ):
            fp8.collect_fp8(Path("/tmp/work"))
        self.assertEqual(
            [call.args[0] for call in gate.call_args_list], ["triton", fp8.PROFILE]
        )
        self.assertNotIn("max_tokens", collect.call_args.kwargs)
        self.assertNotIn("limit", collect.call_args.kwargs)
        self.assertIn(
            study.artifact("mtp-k3-acceptance.json", fp8.PROFILE),
            analyze.call_args.args,
        )

    def test_failure_records_error_and_releases_owned_server(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(
                    fp8, "collect_fp8", side_effect=ValueError("startup failed")
                ),
                patch.object(study, "stop_server") as stop,
            ):
                with self.assertRaisesRegex(ValueError, "startup failed"):
                    fp8.main(["--work-dir", str(work)])
            state = json.loads((work / "gpu-triton-fp8-status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["error"], "startup failed")
            stop.assert_called_once_with(work)


if __name__ == "__main__":
    unittest.main()
