"""Run T0 MTP or the priority-1 sweep on the recorded, dedicated GPU server.

Use the repository Python environment and keep --work-dir outside the repo.
The runner uses the recorded original or Triton plain command and owns only servers it records,
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
from scripts.lpx.gpu_metrics import counters, output_quality

ARTIFACTS = ROOT / "experiments/lpx"
ASSISTANT = "google/gemma-4-31B-it-assistant"
ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"


def artifact(name, profile="original"):
    if profile not in ("original", "triton", "triton-fp8"):
        raise WorkloadValidationError("Unknown serving profile")
    prefix = "gpu-" if profile == "original" else f"gpu-{profile}-"
    return ARTIFACTS / f"{prefix}{name}"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def server_command(base, draft_length, weight_precision="bf16", method="mtp"):
    allowed = {"plain": (0,), "mtp": (1, 3, 5, 7), "ngram": (7,)}
    if draft_length not in allowed.get(method, ()):
        raise WorkloadValidationError("Unsupported method and draft length")
    if "--speculative-config" in base:
        raise WorkloadValidationError(
            "The recorded base command must be plain decoding"
        )
    if weight_precision not in ("bf16", "fp8") or "--quantization" in base:
        raise WorkloadValidationError("Expected an unquantized base and BF16 or FP8")
    command = [*base, *(["--quantization", "fp8"] if weight_precision == "fp8" else [])]
    if method != "plain":
        config = dict(method=method, num_speculative_tokens=draft_length)
        if method == "mtp":
            config.update(model=ASSISTANT, revision=ASSISTANT_REVISION)
        else:
            config.update(prompt_lookup_min=1, prompt_lookup_max=3)
        command += ["--speculative-config", json.dumps(config)]
    return command


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


def server_matches(state, argv, environment):
    if state.get("argv") != argv or state.get("environment") != environment:
        return False
    if not owns_server(state):
        return False
    try:
        entries = Path(f"/proc/{state['pid']}/environ").read_bytes().split(b"\0")
        actual = dict(entry.decode().split("=", 1) for entry in entries if entry)
    except (FileNotFoundError, ProcessLookupError):
        return False
    return all(actual.get(key) == value for key, value in environment.items())


def start_server(
    work,
    draft_length,
    profile="original",
    method="mtp",
    profiler_config=None,
    enforce_eager=False,
):
    base_name = (
        "gpu-server-plain-active-command.json"
        if profile == "original"
        else "gpu-server-triton-command.json"
    )
    commands_path = artifact(f"{method}-server-commands.jsonl", profile)
    base = read_json(ARTIFACTS / base_name)
    if method == "ngram":
        base["environment"]["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    precision = "fp8" if profile == "triton-fp8" else "bf16"
    argv = server_command(base["argv"], draft_length, precision, method)
    if profiler_config is not None:
        if method != "plain":
            raise WorkloadValidationError("Profiling controls require plain decoding")
        argv += ["--profiler-config", json.dumps(profiler_config)]
    if enforce_eager:
        argv += ["--enforce-eager"]
    state_path = work / "server-process.json"
    if state_path.exists():
        state = read_json(state_path)
        if (
            state.get("draft_length") == draft_length
            and server_matches(state, argv, base["environment"])
            and healthy()
        ):
            return
    stop_server(work)
    log = work / f"server-{profile}-{method}-k{draft_length}-{time.time_ns()}.log"
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
        method=method,
        profile=profile,
        log=str(log),
        started_unix=time.time(),
    )
    write_json(state_path, state)
    with commands_path.open("a") as handle:
        handle.write(json.dumps(state) + "\n")
    for attempt in range(180):
        if process.poll() is not None:
            raise WorkloadValidationError(f"{method} server exited; inspect {log}")
        if healthy():
            if (
                profile.startswith("triton")
                and "Using AttentionBackendEnum.TRITON_ATTN backend."
                not in log.read_text()
            ):
                raise WorkloadValidationError(
                    "Startup log does not confirm Triton attention"
                )
            if method == "plain":
                return
            if method == "ngram" and "Using V2 Model Runner" not in log.read_text():
                raise WorkloadValidationError("Startup log does not confirm V2")
            if f"method='{method}'" not in log.read_text():
                raise WorkloadValidationError(
                    f"Startup log does not confirm the {method} method"
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
                    f"{method} server lacks the required speculative counters"
                )
            return
        time.sleep(10)
    raise WorkloadValidationError(f"{method} server startup timed out; inspect {log}")


def run_script(name, *arguments):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
        cwd=ROOT,
        check=True,
    )


def collect(
    work,
    draft_length,
    dataset,
    limit=None,
    max_tokens=None,
    phase="measurement",
    warmup_tag=None,
    profile="original",
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
    if warmup_tag is not None:
        if phase != "warmup" or not str(warmup_tag).isdigit():
            raise WorkloadValidationError(
                "Warmup tags must be numeric and used only for warmups"
            )
        suffix += f"-{warmup_tag}"
        label += f"-{warmup_tag}"
    output = artifact(f"mtp-k{draft_length}-{suffix}.jsonl", profile)
    if profile != "original":
        label = label.replace("-mtp-", f"-{profile}-mtp-")
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
        "--weight-precision",
        "fp8" if profile == "triton-fp8" else "bf16",
    ]
    if phase != "measurement" and dataset == "aa-docs":
        arguments += ["--context", 10000]
    if limit:
        arguments += ["--limit", limit]
    if max_tokens:
        arguments += ["--max-tokens", max_tokens]
    run_script("generate_lpx_gpu_outputs.py", *arguments)


def smoke_tests(work, profile="original"):
    plain = artifact("plain-coding-smoke.jsonl", profile)
    if not plain.exists() or len(plain.read_text().splitlines()) != 10:
        raise WorkloadValidationError(
            "Complete the ten plain coding requests before MTP T0"
        )
    start_server(work, 1, profile=profile)
    collect(work, 1, "aa-docs", limit=1, max_tokens=128, phase="smoke", profile=profile)
    start_server(work, 3, profile=profile)
    collect(work, 3, "speed-coding", limit=10, phase="smoke", profile=profile)
    run_script(
        "analyze_lpx_gpu_smoke.py",
        "--plain",
        plain,
        "--mtp",
        artifact("mtp-k3-coding-smoke.jsonl", profile),
        "--tokenizer",
        work / "inputs/tokenizer.json",
        "--summary",
        artifact("smoke-comparison.json", profile),
    )
    if profile == "triton":
        collect(work, 3, "aa-docs", limit=1, phase="smoke", profile=profile)


def pricing_ratio(totals):
    native = sum(row["gemma_tokens"] for row in totals)
    external = sum(row["o200k_tokens"] for row in totals)
    if native <= 0 or external <= 0:
        raise WorkloadValidationError(
            "Token-ratio pricing requires nonempty output text"
        )
    ratio = native / external
    return ratio if abs(ratio - 1.006) > 0.01 + 1e-12 else 1.006


def measure_ratio(work, draft_length, profile="original"):
    rows = work / f"rows-{profile}-mtp-k{draft_length}"
    run_script(
        "generate_lpx_gpu_outputs.py",
        "--requests",
        work / "inputs/requests.jsonl",
        "--outputs",
        artifact(f"mtp-k{draft_length}-outputs.jsonl", profile),
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
    summary = artifact(f"mtp-k{draft_length}-token-ratio.json", profile)
    run_script("measure_lpx_token_ratio.py", *arguments, "--summary", summary)
    return read_json(summary)["categories"]["aa-output"]


def validate_document_control(profile):
    if profile == "original":
        return
    path = artifact("mtp-k3-smoke.jsonl", profile)
    if not path.exists():
        raise WorkloadValidationError(
            "Complete the profile's full-cap MTP document control"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1 or len(rows[0]["turns"]) != 1:
        raise WorkloadValidationError("Expected one document control turn")
    row = rows[0]
    turn = row["turns"][0]
    quality = output_quality(turn)
    if (
        row["config"]["max_tokens_override"] is not None
        or row["config"]["method"] != "mtp"
        or row["config"]["K"] != 3
        or turn["finish_reason"] != "stop"
        or quality["empty_answer"] is not False
        or quality["repetition_flag"] is not False
    ):
        raise WorkloadValidationError(
            "The full-cap MTP document control is not healthy"
        )


def priority_one(work, profile="original"):
    if (
        not artifact("runtime-estimates.json", profile).exists()
        or not artifact("smoke-comparison.json", profile).exists()
    ):
        raise WorkloadValidationError("Complete T0 and the runtime estimate before T1")
    validate_document_control(profile)
    ratios = []
    for draft_length in (3, 7):
        start_server(work, draft_length, profile=profile)
        collect(
            work,
            draft_length,
            "aa-docs",
            limit=1,
            max_tokens=128,
            phase="warmup",
            warmup_tag=time.time_ns(),
            profile=profile,
        )
        collect(work, draft_length, "speed-coding", profile=profile)
    for draft_length in (3, 7):
        start_server(work, draft_length, profile=profile)
        collect(
            work,
            draft_length,
            "aa-docs",
            limit=1,
            max_tokens=128,
            phase="warmup",
            warmup_tag=time.time_ns(),
            profile=profile,
        )
        collect(work, draft_length, "aa-docs", profile=profile)
        measured = measure_ratio(work, draft_length, profile=profile)
        ratios.append(measured)
        run_script(
            "analyze_lpx_gpu_acceptance.py",
            "--outputs",
            artifact(f"mtp-k{draft_length}-outputs.jsonl", profile),
            "--summary",
            artifact(f"mtp-k{draft_length}-acceptance.json", profile),
            "--ratio",
            pricing_ratio([measured]),
        )
    run_script(
        "analyze_lpx_gpu_acceptance.py",
        "--outputs",
        artifact("mtp-k3-outputs.jsonl", profile),
        artifact("mtp-k7-outputs.jsonl", profile),
        "--summary",
        artifact("mtp-acceptance.json", profile),
        "--ratio",
        pricing_ratio(ratios),
    )
    stop_server(work)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument("--stage", choices=("t0-mtp", "priority1"), required=True)
    parser.add_argument("--profile", choices=("original", "triton"), default="original")
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(ROOT):
        parser.error("work-dir must be outside the repository")
    with (args.work_dir / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            stage=args.stage,
            profile=args.profile,
            pid=os.getpid(),
            status="running",
            started_unix=time.time(),
        )
        path = args.work_dir / "runner-status.json"
        write_json(path, state)
        try:
            (smoke_tests if args.stage == "t0-mtp" else priority_one)(
                args.work_dir, profile=args.profile
            )
        except Exception as error:
            state.update(status="failed", error=str(error), finished_unix=time.time())
            write_json(path, state)
            raise
        state.update(status="complete", finished_unix=time.time())
        write_json(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
