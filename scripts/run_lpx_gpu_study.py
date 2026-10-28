"""Run T0 MTP or the priority-1 sweep on the recorded, dedicated GPU server.

Use the repository Python environment and keep --work-dir outside the repo.
The runner reuses the recorded plain command, owns only servers it records,
and waits for GPU memory release between configurations. An exclusive file
lock prevents two runners from sharing speculative counters.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.lpx.gpu_metrics import counters

ARTIFACTS = ROOT / "experiments/lpx"
ASSISTANT = "google/gemma-4-31B-it-assistant"
ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def server_command(base, draft_length):
    if draft_length not in (1, 3, 7):
        raise WorkloadValidationError("This runner supports MTP K=1,3,7")
    if "--speculative-config" in base:
        raise WorkloadValidationError(
            "The recorded base command must be plain decoding"
        )
    return [
        *base,
        "--speculative-config",
        json.dumps(
            dict(
                method="mtp",
                model=ASSISTANT,
                revision=ASSISTANT_REVISION,
                num_speculative_tokens=draft_length,
            )
        ),
    ]


def healthy():
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/health", timeout=3
        ) as response:
            return response.status == 200
    except OSError:
        return False


def owns_server(state):
    proc = Path(f"/proc/{state['pid']}")
    try:
        argv = proc.joinpath("cmdline").read_bytes().decode().split("\0")[:-1]
        expected = state["argv"]
        return (
            bool(argv)
            and argv[-len(expected) :] == expected
            and os.getpgid(state["pid"]) == state["pid"]
        )
    except (FileNotFoundError, ProcessLookupError):
        return False


def stop_server(work):
    state_path = work / "server-process.json"
    if not state_path.exists():
        if healthy():
            raise WorkloadValidationError("An unrecorded server occupies port 8000")
        return
    state = read_json(state_path)
    pid = state["pid"]
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        argv = proc.joinpath("cmdline").read_bytes().decode().split("\0")[:-1]
        if argv:
            if not owns_server(state):
                raise WorkloadValidationError(
                    "Recorded PID is not the owned server process group"
                )
            os.killpg(pid, signal.SIGTERM)
    for attempt in range(90):
        memory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if all(float(value) < 1024 for value in memory.splitlines()) and not healthy():
            state_path.unlink(missing_ok=True)
            return
        time.sleep(2)
    raise WorkloadValidationError(
        "GPU memory or server port did not clear after shutdown"
    )


def start_server(work, draft_length):
    state_path = work / "server-process.json"
    if state_path.exists():
        state = read_json(state_path)
        if (
            state.get("draft_length") == draft_length
            and owns_server(state)
            and healthy()
        ):
            return
    stop_server(work)
    base = read_json(ARTIFACTS / "gpu-server-plain-active-command.json")
    argv = server_command(base["argv"], draft_length)
    log = work / f"server-mtp-k{draft_length}-{time.time_ns()}.log"
    with log.open("w") as handle:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=dict(os.environ, **base["environment"]),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = dict(
        pid=process.pid,
        argv=argv,
        environment=base["environment"],
        draft_length=draft_length,
        log=str(log),
        started_unix=time.time(),
    )
    write_json(state_path, state)
    with (ARTIFACTS / "gpu-mtp-server-commands.jsonl").open("a") as handle:
        handle.write(json.dumps(state) + "\n")
    for attempt in range(180):
        if process.poll() is not None:
            raise WorkloadValidationError(f"MTP server exited; inspect {log}")
        if healthy():
            if "method='mtp'" not in log.read_text():
                raise WorkloadValidationError(
                    "Startup log does not confirm the MTP method"
                )
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/metrics", timeout=10
            ) as response:
                measured = counters(response.read().decode(), "google/gemma-4-31B-it")
            if not {
                "drafts",
                "draft_tokens",
                "accepted_tokens",
                "accepted_per_position",
            }.issubset(measured):
                raise WorkloadValidationError(
                    "MTP server lacks the required speculative counters"
                )
            return
        time.sleep(10)
    raise WorkloadValidationError(f"MTP server startup timed out; inspect {log}")


def run_script(name, *arguments):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
        cwd=ROOT,
        check=True,
    )


def collect(
    work, draft_length, dataset, limit=None, max_tokens=None, phase="measurement"
):
    if phase not in ("measurement", "smoke", "warmup"):
        raise WorkloadValidationError("Unknown collection phase")
    if phase == "smoke":
        suffix = "smoke" if dataset == "aa-docs" else "coding-smoke"
        label = f"t0-mtp-k{draft_length}"
    else:
        suffix = "outputs" if phase == "measurement" else "warmup"
        label = f"t1-mtp-k{draft_length}"
        if phase == "warmup":
            label += "-warmup"
    output = ARTIFACTS / f"gpu-mtp-k{draft_length}-{suffix}.jsonl"
    arguments = [
        "--requests",
        work / "inputs/requests.jsonl",
        "--outputs",
        output,
        "--environment",
        ARTIFACTS / "gpu-environment.json",
        "--method",
        "mtp",
        "--draft-tokens",
        draft_length,
        "--run-label",
        label,
        "--set",
        dataset,
        "--thinking",
        "on" if dataset == "aa-docs" else "off",
    ]
    if phase != "measurement" and dataset == "aa-docs":
        arguments += ["--context", 10000]
    if limit:
        arguments += ["--limit", limit]
    if max_tokens:
        arguments += ["--max-tokens", max_tokens]
    run_script("generate_lpx_gpu_outputs.py", *arguments)


def smoke_tests(work):
    plain = ARTIFACTS / "gpu-plain-coding-smoke.jsonl"
    if not plain.exists() or len(plain.read_text().splitlines()) != 10:
        raise WorkloadValidationError(
            "Complete the ten plain coding requests before MTP T0"
        )
    start_server(work, 1)
    collect(work, 1, "aa-docs", limit=1, max_tokens=128, phase="smoke")
    start_server(work, 3)
    collect(work, 3, "speed-coding", limit=10, phase="smoke")
    run_script(
        "analyze_lpx_gpu_smoke.py",
        "--plain",
        plain,
        "--mtp",
        ARTIFACTS / "gpu-mtp-k3-coding-smoke.jsonl",
        "--tokenizer",
        work / "inputs/tokenizer.json",
        "--summary",
        ARTIFACTS / "gpu-smoke-comparison.json",
    )


def pricing_ratio(totals):
    native = sum(row["gemma_tokens"] for row in totals)
    external = sum(row["o200k_tokens"] for row in totals)
    if native <= 0 or external <= 0:
        raise WorkloadValidationError(
            "Token-ratio pricing requires nonempty output text"
        )
    ratio = native / external
    return ratio if abs(ratio - 1.006) > 0.01 + 1e-12 else 1.006


def measure_ratio(work, draft_length):
    rows = work / f"rows-mtp-k{draft_length}"
    run_script(
        "generate_lpx_gpu_outputs.py",
        "--requests",
        work / "inputs/requests.jsonl",
        "--outputs",
        ARTIFACTS / f"gpu-mtp-k{draft_length}-outputs.jsonl",
        "--run-label",
        "export",
        "--export",
        rows,
    )
    arguments = ["--gemma-tokenizer", work / "inputs/tokenizer.json"]
    for path in sorted(rows.glob("aa-docs-*.json")):
        records = read_json(path)["rows"]
        for part in ("thinking", "answer"):
            if any(row["row"].get(part) for row in records):
                arguments += ["--text", f"aa-output={path}#{part}"]
    summary = ARTIFACTS / f"gpu-mtp-k{draft_length}-token-ratio.json"
    run_script("measure_lpx_token_ratio.py", *arguments, "--summary", summary)
    return read_json(summary)["categories"]["aa-output"]


def priority_one(work):
    if (
        not (ARTIFACTS / "gpu-runtime-estimates.json").exists()
        or not (ARTIFACTS / "gpu-smoke-comparison.json").exists()
    ):
        raise WorkloadValidationError("Complete T0 and the runtime estimate before T1")
    ratios = []
    for draft_length in (3, 7):
        start_server(work, draft_length)
        collect(work, draft_length, "aa-docs", limit=1, max_tokens=128, phase="warmup")
        collect(work, draft_length, "aa-docs")
        collect(work, draft_length, "speed-coding")
        measured = measure_ratio(work, draft_length)
        ratios.append(measured)
        run_script(
            "analyze_lpx_gpu_acceptance.py",
            "--outputs",
            ARTIFACTS / f"gpu-mtp-k{draft_length}-outputs.jsonl",
            "--summary",
            ARTIFACTS / f"gpu-mtp-k{draft_length}-acceptance.json",
            "--ratio",
            pricing_ratio([measured]),
        )
    run_script(
        "analyze_lpx_gpu_acceptance.py",
        "--outputs",
        ARTIFACTS / "gpu-mtp-k3-outputs.jsonl",
        ARTIFACTS / "gpu-mtp-k7-outputs.jsonl",
        "--summary",
        ARTIFACTS / "gpu-mtp-acceptance.json",
        "--ratio",
        pricing_ratio(ratios),
    )
    stop_server(work)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument("--stage", choices=("t0-mtp", "priority1"), required=True)
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(ROOT):
        parser.error("work-dir must be outside the repository")
    with (args.work_dir / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            stage=args.stage,
            pid=os.getpid(),
            status="running",
            started_unix=time.time(),
        )
        path = args.work_dir / "runner-status.json"
        write_json(path, state)
        try:
            (smoke_tests if args.stage == "t0-mtp" else priority_one)(args.work_dir)
        except Exception as error:
            state.update(status="failed", error=str(error), finished_unix=time.time())
            write_json(path, state)
            raise
        state.update(status="complete", finished_unix=time.time())
        write_json(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
