import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import monitor_lpx_gpu_periodic as timer


class GpuPeriodicTest(unittest.TestCase):
    def test_checks_repeat_until_entire_workflow_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = Path(directory) / "workflow.json"
            state = Path(directory) / "timer.json"
            workflow.write_text(json.dumps(dict(status="running")))
            clock = {"now": 1000}
            deliveries = []

            def sleep(seconds):
                clock["now"] += seconds
                if len(deliveries) == 3:
                    workflow.write_text(json.dumps(dict(status="complete")))

            def notify(*args):
                deliveries.append(clock["now"])
                return subprocess.CompletedProcess([], 0, "queued", "")

            with (
                patch.object(timer.time, "time", side_effect=lambda: clock["now"]),
                patch.object(timer.time, "sleep", side_effect=sleep),
                patch.object(timer, "notify", side_effect=notify),
            ):
                timer.monitor(
                    workflow,
                    Path("task.md"),
                    state,
                    "thread",
                    Path("codex"),
                    initial=True,
                )
            self.assertEqual(deliveries, [1000, 2800, 4600])
            self.assertEqual(json.loads(state.read_text())["status"], "complete")

    def test_missing_workflow_and_queue_failure_do_not_stop_timer(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = Path(directory) / "missing.json"
            state = Path(directory) / "timer.json"
            with (
                patch.object(timer.time, "time", return_value=1000),
                patch.object(timer.time, "sleep", side_effect=KeyboardInterrupt),
                patch.object(
                    timer,
                    "notify",
                    return_value=subprocess.CompletedProcess([], 1, "", "offline"),
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    timer.monitor(
                        workflow,
                        Path("task.md"),
                        state,
                        "thread",
                        Path("codex"),
                        initial=True,
                    )
            recorded = json.loads(state.read_text())
            self.assertEqual(recorded["status"], "monitoring")
            self.assertEqual(recorded["next_due_unix"], 1060)
            self.assertEqual(recorded["deliveries"], 0)
            self.assertIn("read_error", recorded)

    def test_restart_preserves_deadline_and_rejects_wrong_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = Path(directory) / "workflow.json"
            workflow.write_text(json.dumps(dict(status="running")))
            state = Path(directory) / "timer.json"
            with (
                patch.object(timer.time, "time", return_value=1000),
                patch.object(timer.time, "sleep", side_effect=KeyboardInterrupt),
                patch.object(timer, "notify") as notify,
            ):
                for unused in range(2):
                    with self.assertRaises(KeyboardInterrupt):
                        timer.monitor(
                            workflow, Path("task.md"), state, "thread", Path("codex")
                        )
                notify.assert_not_called()
                self.assertEqual(json.loads(state.read_text())["next_due_unix"], 2800)
                with self.assertRaises(WorkloadValidationError):
                    timer.monitor(
                        workflow, Path("task.md"), state, "other", Path("codex")
                    )
