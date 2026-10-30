import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import run_lpx_gpu_profile as profile
from scripts import run_lpx_gpu_study as study
from scripts.lpx.gpu_profile import (
    file_record,
    profiler_config,
    read_profile_turn,
    stream_step_us,
    timing_comparison,
    trace_summary,
    verify_records,
)


def output_fixture(work, name="output", elapsed=3):
    inputs = work / "inputs/requests.jsonl"
    inputs.parent.mkdir(exist_ok=True)
    inputs.write_text("[]\n")
    environment = work / "gpu-environment.json"
    environment.write_text("{}\n")
    row = dict(
        id="aa-docs/book/10k/summary",
        set="aa-docs",
        thinking=True,
        meta=dict(context=10000),
        config=dict(
            method="plain",
            K=0,
            max_tokens_override=256,
            weight_precision="bf16",
            kv_precision="bf16",
            requests_sha256=file_record(inputs)["sha256"],
            environment_sha256=file_record(environment)["sha256"],
        ),
        turns=[
            dict(
                turn=0,
                completion_tokens=4,
                emitted_steps=4,
                preemptions=0,
                finish_reason="stop",
                token_ids=[1, 2, 3, 4],
                decode_s=elapsed,
            )
        ],
    )
    path = work / f"{name}.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path, row


class GpuProfileTest(unittest.TestCase):
    def test_profiler_keeps_frontend_shapes_and_both_activities(self):
        config = profiler_config(Path("/tmp/profile-test"))
        self.assertEqual(config["torch_profiler_activities"], ["CPU", "CUDA"])
        self.assertTrue(config["torch_profiler_record_shapes"])
        self.assertTrue(config["torch_profiler_with_stack"])
        self.assertFalse(config["torch_profiler_dump_cuda_time_total"])
        self.assertFalse(config["ignore_frontend"])
        with self.assertRaises(WorkloadValidationError):
            profiler_config(study.ROOT / "traces")

    def test_stream_timing_normalizes_different_output_lengths(self):
        control = dict(completion_tokens=4, decode_s=3, token_ids=[1, 2, 3, 4])
        measured = dict(completion_tokens=3, decode_s=4, token_ids=[1, 2, 9])
        result = timing_comparison(control, measured)
        self.assertEqual(result["control_stream_step_us"], 1e6)
        self.assertEqual(result["profiled_stream_step_us"], 2e6)
        self.assertEqual(result["profiled_over_control"], 2)
        self.assertEqual(result["first_different_token"], 2)
        self.assertFalse(result["identical_tokens"])
        for changed in (
            dict(control, completion_tokens=1),
            dict(control, decode_s=0),
            dict(control, decode_s=float("nan")),
        ):
            with self.assertRaises(WorkloadValidationError):
                stream_step_us(changed)

    def test_profile_record_requires_capped_plain_counter_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            path, row = output_fixture(work)
            self.assertEqual(read_profile_turn(path, 10000)[0], row)
            for mutate in (
                lambda changed: changed["config"].update(max_tokens_override=None),
                lambda changed: changed["turns"][0].update(emitted_steps=3),
                lambda changed: changed.update(thinking=False),
                lambda changed: changed["turns"][0].update(token_ids=[1]),
            ):
                changed = copy.deepcopy(row)
                mutate(changed)
                path.write_text(json.dumps(changed) + "\n")
                with self.assertRaises(WorkloadValidationError):
                    read_profile_turn(path, 10000)

    def test_trace_summary_and_hashes_detect_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump(
                    dict(
                        traceEvents=[
                            dict(cat="kernel"),
                            dict(cat="cpu_op"),
                            dict(ph="M"),
                        ]
                    ),
                    handle,
                )
            result = trace_summary(path)
            self.assertEqual(result["kernel_events"], 1)
            self.assertEqual(result["event_count"], 3)
            record = file_record(path)
            verify_records([record])
            path.write_bytes(b"changed")
            with self.assertRaises(WorkloadValidationError):
                verify_records([record])
            path.unlink()
            with self.assertRaises(WorkloadValidationError):
                verify_records([record])

    def test_invalid_trace_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            for value in ({}, dict(traceEvents=[]), dict(traceEvents=[None])):
                path.write_text(json.dumps(value))
                with self.assertRaises(WorkloadValidationError):
                    trace_summary(path)

    def test_profile_and_eager_flags_are_part_of_server_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "server-process.json").touch()
            base = dict(argv=["vllm", "serve", "target"], environment={})
            config = profiler_config(work / "traces")
            with (
                patch.object(
                    study, "read_json", side_effect=[base, dict(draft_length=0)]
                ),
                patch.object(study, "server_matches", return_value=True) as matches,
                patch.object(study, "healthy", return_value=True),
            ):
                study.start_server(
                    work,
                    0,
                    profile="triton",
                    method="plain",
                    profiler_config=config,
                    enforce_eager=True,
                )
            command = matches.call_args.args[1]
            self.assertIn("--enforce-eager", command)
            self.assertEqual(
                json.loads(command[command.index("--profiler-config") + 1]), config
            )
            self.assertNotIn("--profiler-config", base["argv"])

    def test_profiler_stops_even_if_generation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            control, _ = output_fixture(work)
            with (
                patch.object(
                    profile,
                    "collect",
                    side_effect=[control, ValueError("generation failed")],
                ),
                patch.object(profile, "profile_request") as request,
            ):
                with self.assertRaisesRegex(ValueError, "generation failed"):
                    profile.collect_case(work, "graphs", 10000, work / "traces", {})
            self.assertEqual(
                [call.args[0] for call in request.call_args_list], ["start", "stop"]
            )

    def test_collection_records_kernel_traces_and_resumes_verified_case(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            traces = work / "traces"
            traces.mkdir()
            control, _ = output_fixture(work, "control", 3)
            measured, _ = output_fixture(work, "profiled", 6)

            def stop_trace(action, prefix=None):
                if action == "stop":
                    (traces / "trace.json").write_text(
                        json.dumps(dict(traceEvents=[dict(cat="kernel")]))
                    )

            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(profile, "collect", side_effect=[control, measured]),
                patch.object(profile, "profile_request", side_effect=stop_trace),
            ):
                result = profile.collect_case(work, "graphs", 10000, traces, {})
                self.assertEqual(result["analysis_status"], "pending")
                self.assertEqual(result["trace_summaries"][0]["kernel_events"], 1)
                self.assertEqual(profile.completed_case(work, "graphs", 10000), result)
                (work / "inputs/requests.jsonl").write_text("changed")
                with self.assertRaises(WorkloadValidationError):
                    profile.completed_case(work, "graphs", 10000)

    def test_cpu_only_profile_does_not_count_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            traces = work / "traces"
            traces.mkdir()
            control, _ = output_fixture(work, "control")
            measured, _ = output_fixture(work, "profiled")

            def stop_trace(action, prefix=None):
                if action == "stop":
                    (traces / "trace.json").write_text(
                        json.dumps(dict(traceEvents=[dict(cat="cpu_op")]))
                    )

            with (
                patch.object(profile, "collect", side_effect=[control, measured]),
                patch.object(profile, "profile_request", side_effect=stop_trace),
            ):
                with self.assertRaisesRegex(WorkloadValidationError, "no GPU kernel"):
                    profile.collect_case(work, "graphs", 10000, traces, {})

    def test_wait_requires_successful_full_followup(self):
        complete = dict(
            status="complete", completed_stages=["plain", "ngram", "remaining-mtp"]
        )
        with (
            patch.object(
                study,
                "read_json",
                side_effect=[dict(status="running", pid=1), complete],
            ),
            patch.object(profile, "worker_alive", return_value=True),
            patch.object(profile.time, "sleep") as sleep,
        ):
            profile.wait_for_followup()
            sleep.assert_called_once_with(60)
        for state in (
            dict(status="failed"),
            dict(status="complete", completed_stages=["plain"]),
        ):
            with patch.object(study, "read_json", return_value=state):
                with self.assertRaises(WorkloadValidationError):
                    profile.wait_for_followup()

    def test_queue_waits_before_acquiring_gpu_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)

            def waited():
                self.assertFalse((work / "runner.lock").exists())

            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(profile, "wait_for_followup", side_effect=waited) as wait,
                patch.object(profile, "collect_profiles") as collect,
            ):
                self.assertEqual(
                    profile.main(["--work-dir", str(work), "--after-followup"]), 0
                )
            wait.assert_called_once()
            collect.assert_called_once_with(work)
            state = json.loads((work / "gpu-triton-t6-status.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["analysis_status"], "pending")

    def test_failed_predecessor_does_not_start_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with (
                patch.object(study, "ARTIFACTS", work),
                patch.object(
                    profile, "wait_for_followup", side_effect=ValueError("failed")
                ),
                patch.object(profile, "collect_profiles") as collect,
            ):
                with self.assertRaises(ValueError):
                    profile.main(["--work-dir", str(work), "--after-followup"])
            collect.assert_not_called()
            self.assertEqual(
                json.loads((work / "gpu-triton-t6-status.json").read_text())["status"],
                "failed",
            )

    def test_server_cleanup_on_startup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with (
                patch.object(profile, "completed_case", return_value=None),
                patch.object(study, "start_server", side_effect=ValueError("startup")),
                patch.object(study, "stop_server") as stop,
            ):
                with self.assertRaisesRegex(ValueError, "startup"):
                    profile.collect_profiles(work)
            stop.assert_called_once_with(work)


if __name__ == "__main__":
    unittest.main()
