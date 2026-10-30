"""Collect capped graph/eager T6 controls and CPU/CUDA profiler traces.

Use --after-followup to wait for successful completion of the existing queue
before acquiring runner.lock. Trace attribution remains a separate analysis:
completion of this worker means trace collection, not completion of T6.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts import run_lpx_gpu_study as study
from scripts.lpx.gpu_profile import (
    file_record,
    profiler_config,
    read_profile_turn,
    timing_comparison,
    trace_inventory,
    trace_summary,
    verify_records,
)
from scripts.monitor_lpx_gpu_study import worker_alive

MODES = ("graphs", "eager")
CONTEXTS = (10000, 100000)


def artifact(name):
    return study.artifact(f"t6-{name}", "triton")


def wait_for_followup():
    path = study.artifact("followup-status.json", "triton")
    read_failures = 0
    while True:
        try:
            state = study.read_json(path)
            read_failures = 0
        except json.JSONDecodeError:
            read_failures += 1
            if read_failures >= 3:
                raise
            time.sleep(60)
            continue
        if state["status"] == "complete":
            if state.get("completed_stages") != ["plain", "ngram", "remaining-mtp"]:
                raise WorkloadValidationError(
                    "T6 prerequisite must cover the full follow-up queue"
                )
            return
        if state["status"] not in (
            "queued",
            "running",
            "waiting-for-gpu",
        ) or not worker_alive(state):
            raise WorkloadValidationError("Follow-up predecessor failed or exited")
        time.sleep(60)


def collect(work, label, context, maximum=256):
    if context not in CONTEXTS or maximum not in (128, 256):
        raise WorkloadValidationError("Unsupported T6 context or output cap")
    path = artifact(f"{label}.jsonl")
    study.run_script(
        "generate_lpx_gpu_outputs.py",
        "--requests",
        work / "inputs/requests.jsonl",
        "--outputs",
        path,
        "--environment",
        study.ARTIFACTS / "gpu-environment.json",
        "--method",
        "plain",
        "--draft-tokens",
        0,
        "--drafter-precision",
        "not-applicable",
        "--run-label",
        f"t6-triton-{label}",
        "--set",
        "aa-docs",
        "--thinking",
        "on",
        "--context",
        context,
        "--limit",
        1,
        "--max-tokens",
        maximum,
    )
    return path


def profile_request(action, prefix=None):
    if action not in ("start", "stop") or (action == "stop" and prefix is not None):
        raise WorkloadValidationError("Invalid profiler action")
    body = dict(profile_prefix=prefix) if prefix is not None else {}
    request = urllib.request.Request(
        f"http://127.0.0.1:8000/{action}_profile",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status != 200:
            raise WorkloadValidationError(
                f"Profiler {action} returned {response.status}"
            )


def result_path(mode, context):
    return artifact(f"{mode}-{context}-result.json")


def completed_case(work, mode, context):
    path = result_path(mode, context)
    if not path.exists():
        return None
    result = study.read_json(path)
    if (
        result.get("mode") != mode
        or result.get("context") != context
        or result.get("status") != "collected"
    ):
        raise WorkloadValidationError(
            "Stored profiling case has a different configuration"
        )
    if len(result.get("outputs", [])) != 2 or not result.get("traces"):
        raise WorkloadValidationError("Stored profiling case lacks outputs or traces")
    for source, field in (
        (work / "inputs/requests.jsonl", "requests_sha256"),
        (study.ARTIFACTS / "gpu-environment.json", "environment_sha256"),
    ):
        if file_record(source)["sha256"] != result.get(field):
            raise WorkloadValidationError(
                "Profiling input or environment record changed"
            )
    verify_records([*result["outputs"], *result["traces"]])
    return result


def collect_case(work, mode, context, directory, server):
    label = f"{mode}-{context}-{time.time_ns()}"
    print(f"T6 {mode} context={context}: unprofiled control", flush=True)
    control_path = collect(work, f"{label}-control", context)
    control_row, control = read_profile_turn(control_path, context)
    before = {record["path"] for record in trace_inventory(directory)}
    print(f"T6 {mode} context={context}: CPU/CUDA profile", flush=True)
    profile_request("start", f"t6-{label}")
    try:
        profiled_path = collect(work, f"{label}-profiled", context)
    finally:
        profile_request("stop")
    profiled_row, profiled = read_profile_turn(profiled_path, context)
    if (
        control_row["id"] != profiled_row["id"]
        or control_row["config"]["requests_sha256"]
        != profiled_row["config"]["requests_sha256"]
    ):
        raise WorkloadValidationError("Profile and control input identities differ")
    traces = [
        record for record in trace_inventory(directory) if record["path"] not in before
    ]
    if not any(
        record["bytes"] > 0 and record["path"].endswith((".json", ".json.gz"))
        for record in traces
    ):
        raise WorkloadValidationError("Profiler produced no new trace file")
    summaries = [
        trace_summary(Path(record["path"]))
        for record in traces
        if record["path"].endswith((".json", ".json.gz"))
    ]
    if not any(summary["kernel_events"] for summary in summaries):
        raise WorkloadValidationError("Profiler traces contain no GPU kernel events")
    result = dict(
        mode=mode,
        context=context,
        status="collected",
        analysis_status="pending",
        requests_sha256=control_row["config"]["requests_sha256"],
        environment_sha256=control_row["config"]["environment_sha256"],
        server=server,
        outputs=[file_record(control_path), file_record(profiled_path)],
        traces=traces,
        trace_summaries=summaries,
        timing=timing_comparison(control, profiled),
        captured_unix=time.time(),
        note="Traces include prefill and decode; category attribution and instrumentation-overhead interpretation remain pending",
    )
    study.write_json(result_path(mode, context), result)
    return result


def collect_profiles(work):
    cases = []
    for mode in MODES:
        existing = {
            context: completed_case(work, mode, context) for context in CONTEXTS
        }
        if all(existing.values()):
            cases.extend(existing.values())
            continue
        directory = work / f"t6-traces-{mode}-{time.time_ns()}"
        try:
            study.start_server(
                work,
                0,
                profile="triton",
                method="plain",
                profiler_config=profiler_config(directory),
                enforce_eager=mode == "eager",
            )
            server = study.read_json(work / "server-process.json")
            with artifact("server-commands.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(mode=mode, **server)) + "\n")
            collect(work, f"{mode}-warmup-{time.time_ns()}", 10000, maximum=128)
            for context in CONTEXTS:
                cases.append(
                    existing[context]
                    or collect_case(work, mode, context, directory, server)
                )
        finally:
            study.stop_server(work)
    study.write_json(
        artifact("collection.json"),
        dict(
            cases=cases,
            analysis_status="pending",
            required_attribution=[
                "prefill versus decode",
                "transformer layers",
                "vocabulary head",
                "logits and sampling",
                "CPU scheduling and kernel gaps",
                "output streaming",
                "unclassified time",
            ],
            note="GPU host costs are a size reference and do not directly transfer to LP30",
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument("--after-followup", action="store_true")
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(ROOT) or not args.work_dir.is_dir():
        parser.error("work-dir must exist outside the repository")
    with (args.work_dir / "profile.lock").open("w") as profile_lock:
        fcntl.flock(profile_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            stage="t6-collection",
            pid=os.getpid(),
            status="queued",
            after_followup=args.after_followup,
            analysis_status="pending",
            started_unix=time.time(),
        )
        path = artifact("status.json")
        study.write_json(path, state)
        try:
            if args.after_followup:
                wait_for_followup()
            with (args.work_dir / "runner.lock").open("w") as lock:
                fcntl.flock(
                    lock, fcntl.LOCK_EX | (0 if args.after_followup else fcntl.LOCK_NB)
                )
                state.update(status="running")
                study.write_json(path, state)
                collect_profiles(args.work_dir)
        except Exception as error:
            state.update(status="failed", error=str(error), finished_unix=time.time())
            study.write_json(path, state)
            raise
        state.update(status="complete", finished_unix=time.time())
        study.write_json(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
