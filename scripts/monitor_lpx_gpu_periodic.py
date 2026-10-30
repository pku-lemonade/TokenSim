"""Queue recurring checks until collection, analysis and publication finish."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.monitor_lpx_gpu_study import notify, read_json, save


def message_for(workflow, task, interval):
    return (
        f"The user requested recurring checks every {interval:g} seconds until "
        f"every TODO is finished. Read {task}, {workflow}, and the current worker "
        "status, output counts and logs before acting. Continue the C0-C4 GPU "
        "code-completion study, including replay, native n-gram comparison, "
        "analysis, tests and publication. Preserve completed outputs and the "
        "recorded expectations. Respect runner.lock; never launch duplicate "
        "collectors or stop unrelated GPU processes. If collection is active, "
        "check progress and continue independent work when possible. Keep this "
        "timer running through analysis and publication. Only mark the workflow "
        "complete when every TODO is finished. Report publication failures "
        "without claiming completion. A check queued during an active turn "
        "may execute after that turn finishes."
    )


def monitor(workflow, task, state_path, thread, codex, interval=1800, initial=False):
    if not math.isfinite(interval) or interval < 5:
        raise WorkloadValidationError("Interval must be finite and at least five")
    if workflow.resolve() == state_path.resolve():
        raise WorkloadValidationError("Timer state must not overwrite workflow")
    with state_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = time.time()
        state = dict(
            thread=thread,
            workflow=str(workflow),
            task=str(task),
            interval_seconds=interval,
            started_unix=now,
            next_due_unix=now if initial else now + interval,
            deliveries=0,
            attempts=0,
        )
        if state_path.exists():
            previous = read_json(state_path)
            for field in ("thread", "workflow", "task", "interval_seconds"):
                if previous.get(field) != state[field]:
                    raise WorkloadValidationError("Timer state belongs to another job")
            state = previous
        state.update(status="monitoring", pid=os.getpid())
        while True:
            now = time.time()
            try:
                current = read_json(workflow)
                if not isinstance(current, dict) or "status" not in current:
                    raise WorkloadValidationError("Workflow needs an explicit status")
                state["workflow_status"] = current["status"]
                state.pop("read_error", None)
                if current["status"] == "complete":
                    state.update(status="complete", finished_unix=now)
                    save(state_path, state)
                    return 0
            except (OSError, ValueError, WorkloadValidationError) as error:
                state["read_error"] = str(error)
            state["checked_unix"] = now
            if now >= state["next_due_unix"]:
                due = state["next_due_unix"]
                state.update(
                    status="submitting",
                    attempts=state["attempts"] + 1,
                    last_due_unix=due,
                    next_due_unix=due
                    + (math.floor((now - due) / interval) + 1) * interval,
                )
                save(state_path, state)
                try:
                    result = notify(
                        codex, thread, {}, {}, message_for(workflow, task, interval)
                    )
                except subprocess.TimeoutExpired:
                    state["last_delivery_error"] = "Queue timeout; delivery uncertain"
                except OSError as error:
                    state["last_delivery_error"] = str(error)
                    state["next_due_unix"] = min(state["next_due_unix"], now + 60)
                else:
                    state["queue_returncode"] = result.returncode
                    if result.returncode == 0:
                        state.update(
                            deliveries=state["deliveries"] + 1,
                            delivered_unix=time.time(),
                            queue_receipt=result.stdout[-2000:],
                        )
                        state.pop("last_delivery_error", None)
                        print(
                            f"Queued periodic check {state['deliveries']} to {thread}",
                            flush=True,
                        )
                    else:
                        state["last_delivery_error"] = result.stderr[-2000:]
                        state["next_due_unix"] = min(state["next_due_unix"], now + 60)
            state["status"] = "monitoring"
            save(state_path, state)
            time.sleep(max(0, min(60, state["next_due_unix"] - time.time())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--thread", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--interval-seconds", type=float, default=1800)
    parser.add_argument("--initial-ping", action="store_true")
    args = parser.parse_args(argv)
    try:
        uuid.UUID(args.thread or "")
    except ValueError:
        parser.error("thread must be an existing Codex session UUID")
    if not args.codex or not Path(args.codex).is_file():
        parser.error("codex must identify the installed executable")
    if not args.task.is_file() or not args.state.parent.is_dir():
        parser.error("task file and state directory must exist")
    return monitor(
        args.workflow.resolve(),
        args.task.resolve(),
        args.state.resolve(),
        args.thread,
        Path(args.codex),
        args.interval_seconds,
        args.initial_ping,
    )


if __name__ == "__main__":
    raise SystemExit(main())
