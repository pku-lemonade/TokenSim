import fcntl
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import monitor_lpx_gpu_study as monitor

THREAD = "00000000-0000-4000-8000-000000000001"


class GpuMonitorTest(unittest.TestCase):
    def test_watch_paths_require_unique_names(self):
        self.assertEqual(set(monitor.watch_paths(None)), {"fp8", "followup"})
        self.assertEqual(
            monitor.watch_paths(["t6=/tmp/t6.json"]), dict(t6=Path("/tmp/t6.json"))
        )
        for values in (["missing"], ["=path"], ["name="], ["same=a", "same=b"]):
            with self.assertRaises(WorkloadValidationError):
                monitor.watch_paths(values)

    def test_fp8_completion_does_not_interrupt_followup(self):
        with (
            patch.object(
                monitor,
                "read_json",
                side_effect=[dict(status="complete"), dict(status="running", pid=1)],
            ),
            patch.object(monitor, "worker_alive", return_value=True),
        ):
            statuses, event = monitor.observe(
                dict(fp8=Path("fp8"), followup=Path("followup"))
            )
        self.assertIsNone(event)
        self.assertEqual(statuses["fp8"]["status"], "complete")

    def test_complete_failure_and_lost_process_events(self):
        for state, alive, kind in (
            ("complete", True, "complete"),
            ("failed", True, "worker-failure"),
            ("running", False, "worker-failure"),
        ):
            with (
                patch.object(
                    monitor, "read_json", return_value=dict(status=state, pid=1)
                ),
                patch.object(monitor, "worker_alive", return_value=alive),
            ):
                _, event = monitor.observe(dict(job=Path("job")))
            self.assertEqual(event["kind"], kind)

    def test_invalid_status_and_pid_fail(self):
        with patch.object(monitor, "read_json", return_value=dict(status="unknown")):
            with self.assertRaises(WorkloadValidationError):
                monitor.observe(dict(job=Path("job")))
        for pid in (None, -1, True):
            with self.assertRaises(WorkloadValidationError):
                monitor.worker_alive(dict(pid=pid))

    def test_transient_event_is_not_delivered(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            event = dict(kind="complete")
            with (
                patch.object(monitor, "observe", side_effect=[({}, event), ({}, None)]),
                patch.object(
                    monitor.time, "sleep", side_effect=[None, KeyboardInterrupt]
                ),
                patch.object(monitor, "notify") as notify,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5)
                notify.assert_not_called()

    def test_success_is_saved_and_not_delivered_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with (
                patch.object(
                    monitor, "observe", return_value=({}, dict(kind="complete"))
                ),
                patch.object(monitor.time, "sleep"),
                patch.object(
                    monitor, "notify", return_value=subprocess.CompletedProcess([], 0)
                ) as notify,
            ):
                self.assertEqual(
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5), 0
                )
                self.assertEqual(
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5), 0
                )
            notify.assert_called_once()
            self.assertEqual(json.loads(state_path.read_text())["status"], "delivered")

    def test_failed_queue_command_retries_without_restarting_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            results = [
                subprocess.CompletedProcess([], 1, stderr="unavailable"),
                subprocess.CompletedProcess([], 0),
            ]
            with (
                patch.object(
                    monitor, "observe", return_value=({}, dict(kind="worker-failure"))
                ),
                patch.object(monitor.time, "sleep"),
                patch.object(monitor, "notify", side_effect=results) as notify,
            ):
                self.assertEqual(
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5), 0
                )
            self.assertEqual(notify.call_count, 2)
            self.assertEqual(json.loads(state_path.read_text())["attempts"], 2)

    def test_timeout_requires_inspection_before_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with (
                patch.object(
                    monitor, "observe", return_value=({}, dict(kind="complete"))
                ),
                patch.object(monitor.time, "sleep"),
                patch.object(
                    monitor,
                    "notify",
                    side_effect=subprocess.TimeoutExpired("codex", 120),
                ) as notify,
            ):
                self.assertEqual(
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5), 1
                )
                with self.assertRaises(WorkloadValidationError):
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5)
            notify.assert_called_once()
            self.assertEqual(
                json.loads(state_path.read_text())["status"], "delivery-uncertain"
            )

    def test_persistent_read_errors_trigger_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with (
                patch.object(
                    monitor, "observe", side_effect=ValueError("incomplete JSON")
                ) as observe,
                patch.object(monitor.time, "sleep"),
                patch.object(
                    monitor, "notify", return_value=subprocess.CompletedProcess([], 0)
                ) as notify,
            ):
                monitor.monitor({}, THREAD, Path("codex"), state_path, 5)
            self.assertEqual(observe.call_count, 4)
            self.assertEqual(notify.call_args.args[2]["kind"], "monitor-read-error")

    def test_duplicate_monitor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with state_path.with_suffix(".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    monitor.monitor({}, THREAD, Path("codex"), state_path, 5)

    def test_notification_targets_existing_thread_without_model_override(self):
        event = dict(kind="complete")
        with patch.object(monitor.subprocess, "run") as run:
            monitor.notify(
                Path("/usr/bin/codex"), THREAD, event, dict(job=Path("/tmp/job.json"))
            )
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["/usr/bin/codex", "queue", "--thread", THREAD])
        self.assertNotIn("--model", command)
        self.assertIn("T6", command[-1])
        self.assertIn("NEW status file", command[-1])


if __name__ == "__main__":
    unittest.main()
