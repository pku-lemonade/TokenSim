"""Watch GPU status files and queue one continuation to an existing Codex thread.

No inference or model polling occurs while waiting. Completion, worker failure,
or persistent unreadable status files trigger a message after two matching
observations. A local lock and saved delivery state prevent normal duplicate
submissions. The monitor must be running on the same host as the workers.
"""

import argparse
import fcntl
import json
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
from scripts.run_lpx_gpu_study import ARTIFACTS, read_json, write_json

ACTIVE = {"running", "waiting-for-gpu", "queued"}
TERMINAL = {"complete", "failed", "interrupted"}


def save(path, state):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, state)
    temporary.replace(path)


def watch_paths(values):
    if not values:
        return {
            name: ARTIFACTS / f"gpu-triton-{name}-status.json"
            for name in ("fp8", "followup")
        }
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise WorkloadValidationError("Each --watch needs a unique NAME=PATH")
        result[name] = Path(path).resolve()
    return result


def worker_alive(status):
    pid = status.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise WorkloadValidationError("Active status needs a positive worker PID")
    process = Path(f"/proc/{pid}")
    try:
        return bool(process.joinpath("cmdline").read_bytes())
    except FileNotFoundError:
        return False


def observe(paths):
    statuses = {name: read_json(path) for name, path in paths.items()}
    for name, state in statuses.items():
        if not isinstance(state, dict) or state.get("status") not in ACTIVE | TERMINAL:
            raise WorkloadValidationError(f"Invalid worker status: {name}")
    failures = sorted(
        name
        for name, state in statuses.items()
        if state["status"] in {"failed", "interrupted"}
    )
    lost = sorted(
        name
        for name, state in statuses.items()
        if state["status"] in ACTIVE and not worker_alive(state)
    )
    if failures or lost:
        return statuses, dict(
            kind="worker-failure", failed=failures, missing_process=lost
        )
    if all(state["status"] == "complete" for state in statuses.values()):
        return statuses, dict(kind="complete", completed=sorted(statuses))
    return statuses, None


def continuation_message(event, paths):
    return (
        "The user requested this automatic continuation when GPU work finishes. "
        f"The LPX monitor observed {json.dumps(event, sort_keys=True)}. "
        f"Continue the TODO in {ROOT / 'docs/lpx-gpu-gemma-todo.md'}. "
        "Read the current status files and logs before acting: "
        + ", ".join(str(path) for path in paths.values())
        + ". Verify completed artifacts, update results and checklists, and resolve failures without discarding outputs. "
        "Continue unfinished work, including T6 graph/eager decode profiling. "
        "Respect runner.lock and do not interfere with active GPU measurements. "
        "If another long run is launched, re-arm this monitor with --watch pointing at its NEW status file; "
        "do not re-arm against already completed runs. Do not declare the study complete while tasks remain."
    )


def notify(codex, thread, event, paths):
    return subprocess.run(
        [
            str(codex),
            "queue",
            "--thread",
            thread,
            "--message",
            continuation_message(event, paths),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def monitor(paths, thread, codex, state_path, interval, rearm=False):
    with state_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists() and not rearm:
            previous = read_json(state_path)
            if previous.get("thread") != thread:
                raise WorkloadValidationError(
                    "Existing monitor belongs to another thread"
                )
            if previous.get("status") == "delivered":
                return 0
            if previous.get("status") in {"submitting", "delivery-uncertain"}:
                raise WorkloadValidationError(
                    "Inspect the thread queue before rearming an uncertain submission"
                )
        state = dict(
            status="monitoring",
            pid=os.getpid(),
            thread=thread,
            paths={name: str(path) for name, path in paths.items()},
            interval_seconds=interval,
            started_unix=time.time(),
            attempts=0,
        )
        candidate, observations, read_failures = None, 0, 0
        while True:
            try:
                statuses, event = observe(paths)
                state["workers"] = statuses
                state.pop("last_read_error", None)
                read_failures = 0
            except (OSError, ValueError, WorkloadValidationError) as error:
                read_failures += 1
                state["last_read_error"] = str(error)
                event = (
                    dict(
                        kind="monitor-read-error",
                        paths=sorted(str(path) for path in paths.values()),
                    )
                    if read_failures >= 3
                    else None
                )
            observations = (
                observations + 1
                if event is not None and event == candidate
                else int(event is not None)
            )
            candidate = event
            state.update(
                checked_unix=time.time(),
                event=event,
                matching_observations=observations,
            )
            save(state_path, state)
            if event is not None and observations >= 2:
                state.update(status="submitting", attempts=state["attempts"] + 1)
                save(state_path, state)
                try:
                    result = notify(codex, thread, event, paths)
                except subprocess.TimeoutExpired:
                    state.update(
                        status="delivery-uncertain",
                        error="Queue command timed out; inspect the thread before retrying",
                    )
                    save(state_path, state)
                    return 1
                except OSError as error:
                    state.update(status="monitoring", last_delivery_error=str(error))
                else:
                    state["queue_returncode"] = result.returncode
                    if result.returncode == 0:
                        state.update(status="delivered", delivered_unix=time.time())
                        save(state_path, state)
                        print(
                            f"Continuation queued to {thread}: {event['kind']}",
                            flush=True,
                        )
                        return 0
                    state.update(
                        status="monitoring", last_delivery_error=result.stderr[-1000:]
                    )
                save(state_path, state)
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--watch", action="append", metavar="NAME=PATH")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument(
        "--once", action="store_true", help="Read status without queuing any message"
    )
    parser.add_argument(
        "--rearm",
        action="store_true",
        help="Start a new watch after inspecting any previous delivery",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 5:
        parser.error("poll-seconds must be finite and at least five")
    paths = watch_paths(args.watch)
    if args.once:
        statuses, event = observe(paths)
        print(json.dumps(dict(workers=statuses, event=event), indent=2))
        return 0
    try:
        uuid.UUID(args.thread or "")
    except ValueError:
        parser.error("thread must be the existing Codex session UUID")
    if not args.codex or not Path(args.codex).is_file():
        parser.error("codex must identify the installed CLI executable")
    if args.work_dir.resolve().is_relative_to(ROOT) or not args.work_dir.is_dir():
        parser.error("work-dir must be an existing directory outside the repository")
    return monitor(
        paths,
        args.thread,
        Path(args.codex),
        args.work_dir / "codex-monitor-state.json",
        args.poll_seconds,
        args.rearm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
