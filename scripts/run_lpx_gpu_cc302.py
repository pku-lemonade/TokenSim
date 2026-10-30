"""Collect the prescribed 302 plain and native n-gram code-completion requests."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts import run_lpx_gpu_study as study
from scripts.analyze_lpx_gpu_smoke import read_outputs
from scripts.monitor_lpx_gpu_study import save
from scripts.run_lpx_gpu_followup import validate_outputs

ORIGINAL_SHA256 = "7abb2f93587b01649c27e262e041afe96dd68cc055908d80f35610b5de9506ae"
EXPANDED_SHA256 = "c4815c8625696784735a924c51a042f554c1f27e309539f87126c30c134b776a"


def artifact(name):
    return study.ARTIFACTS / f"gpu-cc302-{name}"


def audit_inputs(work, requests):
    original = work / "inputs/requests.jsonl"
    rebuilt = work / "inputs/requests-rebuilt.jsonl"
    records = []
    for path, expected in (
        (original, ORIGINAL_SHA256),
        (rebuilt, ORIGINAL_SHA256),
        (requests, EXPANDED_SHA256),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise WorkloadValidationError(f"Request hash differs from plan: {path}")
        records.append(dict(path=str(path), sha256=digest))
    earlier = {
        row["id"] for row in read_outputs(original) if row["set"] == "speed-cc16k"
    }
    selected = [row for row in read_outputs(requests) if row["set"] == "speed-cc16k"]
    identities = {row["id"] for row in selected}
    if (
        len(earlier) != 40
        or len(selected) != 302
        or len(identities) != 302
        or not earlier <= identities
    ):
        raise WorkloadValidationError(
            "Expected 302 unique completions including the earlier 40"
        )
    if any(
        row["reasoning"] is not False
        or row["max_tokens"] != 4096
        or len(row["turns"]) != 1
        for row in selected
    ):
        raise WorkloadValidationError(
            "Expected single-turn 4096-token completions without thinking"
        )
    result = dict(
        status="verified",
        inputs=records,
        requests=302,
        earlier_requests=40,
        earlier_ids_included=True,
    )
    save(artifact("inputs.json"), result)
    return result


def collect(requests, method, draft_length, warmup=False):
    label = "plain" if method == "plain" else "ngram-k7"
    suffix = f"warmup-{time.time_ns()}" if warmup else "outputs"
    output = artifact(f"{label}-{suffix}.jsonl")
    arguments = [
        "--requests",
        requests,
        "--outputs",
        output,
        "--environment",
        study.ARTIFACTS / "gpu-environment.json",
        "--method",
        method,
        "--draft-tokens",
        draft_length,
        "--drafter-precision",
        "not-applicable",
        "--run-label",
        f"cc302-{label}",
        "--set",
        "speed-cc16k",
        "--thinking",
        "off",
    ]
    if warmup:
        arguments += ["--limit", 1, "--max-tokens", 128]
    study.run_script("generate_lpx_gpu_outputs.py", *arguments)
    return output


def validate(work, requests, output, method, draft_length):
    rows = validate_outputs(
        work,
        output,
        [("speed-cc16k", "off")],
        method,
        draft_length,
        requests_path=requests,
    )
    if len(rows) != 302:
        raise WorkloadValidationError("Expected all 302 code-completion outputs")
    return rows


def run(work, requests, stage):
    with (work / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_inputs(work, requests)
        state_path = artifact("collection-status.json")
        state = dict(
            status="running",
            pid=os.getpid(),
            started_unix=time.time(),
            completed=[],
            analysis_status="pending",
        )
        save(state_path, state)
        server_started = False
        try:
            for method, draft_length in (("plain", 0), ("ngram", 7)):
                if stage not in ("all", method):
                    continue
                label = "plain" if method == "plain" else "ngram-k7"
                output = artifact(f"{label}-outputs.jsonl")
                state.update(stage=label, outputs=str(output))
                save(state_path, state)
                complete = output.exists() and len(read_outputs(output)) == 302
                if not complete:
                    server_started = True
                    study.start_server(
                        work,
                        draft_length,
                        profile="triton",
                        method=method,
                        commands_path=artifact("server-commands.jsonl"),
                    )
                    save(
                        artifact(f"{label}-server-command.json"),
                        study.read_json(work / "server-process.json"),
                    )
                    collect(requests, method, draft_length, warmup=True)
                    collect(requests, method, draft_length)
                validate(work, requests, output, method, draft_length)
                state["completed"].append(label)
                save(state_path, state)
            state.update(status="complete", finished_unix=time.time())
        except Exception as error:
            state.update(status="failed", error=str(error), finished_unix=time.time())
            raise
        finally:
            if server_started:
                try:
                    study.stop_server(work)
                    state["gpu_released"] = True
                except Exception as error:
                    state.update(
                        status="failed", cleanup_error=str(error), gpu_released=False
                    )
            save(state_path, state)
        return 0 if state["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "plain", "ngram"), default="all")
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(
        ROOT
    ) or args.requests.resolve().is_relative_to(ROOT):
        parser.error(
            "work-dir and prompt-containing requests must remain outside the repository"
        )
    return run(args.work_dir, args.requests, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
