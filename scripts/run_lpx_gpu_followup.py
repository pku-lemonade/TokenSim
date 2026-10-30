"""Queue isolated Triton T5, T4 and remaining T1 runs after the FP8 sweep.

Use --after-fp8 to wait for the shared GPU lock and require successful T3
completion. An additional lock rejects duplicate queues. Original priority-1
outputs remain unchanged; controls, warmups and measurements stay separate.
"""

import argparse
from collections import Counter
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
from scripts.analyze_lpx_gpu_acceptance import summarize
from scripts.analyze_lpx_gpu_smoke import read_outputs
from scripts.lpx.gpu_metrics import acceptance, ngram_acceptance, output_quality
from scripts.measure_lpx_lookup_acceptance import equivalent_acceptance

DOCS = ("aa-docs", "on")
CODING = ("speed-coding", "off")


def artifact(name):
    return study.artifact(f"followup-{name}", "triton")


def plan(stage):
    if stage == "plain":
        return [("plain", 0, [CODING])]
    if stage == "ngram":
        return [("ngram", 7, [CODING, DOCS])]
    if stage == "remaining-mtp":
        return [
            ("mtp", draft_length, selections)
            for draft_length, selections in (
                (1, [CODING, ("speed-coding", "on"), DOCS]),
                (5, [CODING, ("speed-coding", "on"), DOCS]),
                (3, [("speed-coding", "on"), ("speed-cc16k", "off")]),
                (7, [("speed-coding", "on"), ("speed-cc16k", "off")]),
            )
        ]
    raise WorkloadValidationError("Unknown follow-up stage")


def collect(work, method, draft_length, selection, phase="measurement"):
    dataset, thinking = selection
    suffix = "outputs" if phase == "measurement" else phase
    if phase == "warmup":
        suffix += f"-{time.time_ns()}"
    if phase not in ("measurement", "warmup", "control"):
        raise WorkloadValidationError("Unknown collection phase")
    label = f"{method}-k{draft_length}-{suffix}"
    path = artifact(f"{label}.jsonl")
    arguments = [
        "--requests",
        work / "inputs/requests.jsonl",
        "--outputs",
        path,
        "--environment",
        study.ARTIFACTS / "gpu-environment.json",
        "--method",
        method,
        "--draft-tokens",
        draft_length,
        "--drafter-precision",
        "bf16" if method == "mtp" else "not-applicable",
        "--run-label",
        f"followup-triton-{label}",
        "--set",
        dataset,
        "--thinking",
        thinking,
    ]
    if phase != "measurement":
        arguments += ["--context", 10000, "--limit", 1]
    if phase == "warmup":
        arguments += ["--max-tokens", 128]
    study.run_script("generate_lpx_gpu_outputs.py", *arguments)
    return path


def validate_counters(rows, method, draft_length):
    for row in rows:
        if row["config"]["method"] != method or row["config"]["K"] != draft_length:
            raise WorkloadValidationError("Counter control configuration differs")
        for turn in row["turns"]:
            if turn.get("preemptions") != 0:
                raise WorkloadValidationError(
                    "Counter validation requires zero preemptions"
                )
            if method == "ngram":
                ngram_acceptance(turn, draft_length)
                continue
            steps = (
                turn["completion_tokens"] if method == "plain" else turn["drafts"] + 1
            )
            if turn.get("emitted_steps") != steps:
                raise WorkloadValidationError("Emitted-step identity failed")
            if method == "mtp":
                diagnostic = acceptance(turn, draft_length)
                if (
                    diagnostic["unproposed_tokens"]
                    or not -draft_length <= diagnostic["counter_identity_residual"] <= 0
                ):
                    raise WorkloadValidationError(
                        "MTP terminal counter identity failed"
                    )


def validate_control(path, method, draft_length):
    rows = read_outputs(path)
    if (
        len(rows) != 1
        or len(rows[0]["turns"]) != 1
        or rows[0]["config"]["max_tokens_override"] is not None
    ):
        raise WorkloadValidationError("Expected one uncapped document control")
    quality = output_quality(rows[0]["turns"][0])
    if (
        quality["finish_reason"] != "stop"
        or quality["empty_answer"] is not False
        or quality["repetition_flag"] is not False
    ):
        raise WorkloadValidationError("Document control failed quality checks")
    validate_counters(rows, method, draft_length)


def validate_outputs(work, path, selections, method, draft_length, precision="bf16"):
    requests_path = work / "inputs/requests.jsonl"
    digest = hashlib.sha256(requests_path.read_bytes()).hexdigest()
    expected = {
        row["id"]: row
        for row in read_outputs(requests_path)
        if (row["set"], "on" if row["reasoning"] else "off") in selections
    }
    rows = read_outputs(path)
    if (
        not expected
        or len(rows) != len(expected)
        or {row["id"] for row in rows} != expected.keys()
    ):
        raise WorkloadValidationError(
            "Measurement does not cover the expected request identities"
        )
    for row in rows:
        original = expected[row["id"]]
        config = row["config"]
        if (
            config["max_tokens_override"] is not None
            or config["requests_sha256"] != digest
            or config["weight_precision"] != precision
            or config["kv_precision"] != "bf16"
            or row["meta"] != original["meta"]
            or row["thinking"] != original["reasoning"]
            or [turn["turn"] for turn in row["turns"]]
            != list(range(len(original["turns"])))
        ):
            raise WorkloadValidationError(
                "Measurement configuration or turn coverage differs"
            )
    validate_counters(rows, method, draft_length)
    quality = [output_quality(turn) for row in rows for turn in row["turns"]]
    result = dict(
        outputs=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        requests=len(rows),
        turns=len(quality),
        exact_input_coverage=True,
        finish_reasons=dict(Counter(row["finish_reason"] for row in quality)),
        empty_answers=sum(row["empty_answer"] is True for row in quality),
        repetition_flags=sum(row["repetition_flag"] is True for row in quality),
        note="All completed measurement turns retained, including quality flags",
    )
    study.write_json(path.with_name(path.stem + "-validation.json"), result)
    return rows


def export_rows(work, path):
    directory = work / f"rows-{path.stem}"
    study.run_script(
        "generate_lpx_gpu_outputs.py",
        "--requests",
        work / "inputs/requests.jsonl",
        "--outputs",
        path,
        "--run-label",
        "export",
        "--export",
        directory,
    )
    return directory


def document_ratio(work, path):
    directory = export_rows(work, path)
    arguments = ["--gemma-tokenizer", work / "inputs/tokenizer.json"]
    for export in sorted(directory.glob("aa-docs-*.json")):
        records = study.read_json(export)["rows"]
        for part in ("thinking", "answer"):
            if any(row["row"].get(part) for row in records):
                arguments += ["--text", f"aa-output={export}#{part}"]
    summary = path.with_name(path.stem + "-token-ratio.json")
    study.run_script("measure_lpx_token_ratio.py", *arguments, "--summary", summary)
    return study.read_json(summary)["categories"]["aa-output"]


def compare_fp8(work):
    path = study.artifact("mtp-k3-outputs.jsonl", "triton-fp8")
    measured = validate_outputs(work, path, [DOCS], "mtp", 3, "fp8")
    reference_path = study.artifact("mtp-k3-outputs.jsonl", "triton")
    reference = validate_outputs(work, reference_path, [DOCS, CODING], "mtp", 3)
    summaries = [
        [row for row in summarize(records) if row["data"].startswith("aa-docs-")]
        for records in (reference, measured)
    ]
    baseline = {row["data"]: row for row in summaries[0]}
    groups = []
    for row in summaries[1]:
        before = baseline[row["data"]]
        difference = row["equivalent_acceptance"] - before["equivalent_acceptance"]
        groups.append(
            dict(
                data=row["data"],
                turns=row["turn_count"],
                bf16=before["equivalent_acceptance"],
                fp8=row["equivalent_acceptance"],
                difference=difference,
                within_0_03=abs(difference) < 0.03,
                quality=row["quality"],
            )
        )
    if len(groups) != 4 or any(
        sum(row["turn_count"] for row in summary) != 64 for summary in summaries
    ):
        raise WorkloadValidationError(
            "FP8 comparison requires all four document groups"
        )
    means = [
        sum(row["tau"] * row["turn_count"] for row in summary) / 64
        for summary in summaries
    ]
    alphas = [equivalent_acceptance(mean, 3) for mean in means]
    study.write_json(
        study.artifact("comparison.json", "triton-fp8"),
        dict(
            sources={
                str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                for source in (reference_path, path)
            },
            groups=groups,
            combined=dict(
                bf16=alphas[0],
                fp8=alphas[1],
                difference=alphas[1] - alphas[0],
                within_0_03=abs(alphas[1] - alphas[0]) < 0.03,
            ),
            note="BF16 assistant and KV cache; FP8 target; all turns retained; quality flags must accompany interpretation",
        ),
    )


def analyze(work, stage, paths):
    if stage == "ngram":
        study.run_script(
            "analyze_lpx_gpu_ngram.py",
            "--outputs",
            paths[0],
            "--rows",
            export_rows(work, paths[0]),
            "--gemma-tokenizer",
            work / "inputs/tokenizer.json",
            "--summary",
            artifact("ngram-comparison.json"),
            "--server-command",
            artifact("ngram-k7-server-command.json"),
            "--validated-engine-steps",
        )
        study.run_script(
            "analyze_lpx_gpu_acceptance.py",
            "--outputs",
            *paths,
            "--summary",
            artifact("ngram-acceptance.json"),
            "--validated-ngram-engine-steps",
        )
        return
    ratios = [
        study.read_json(
            study.artifact(f"mtp-k{draft_length}-token-ratio.json", "triton")
        )["categories"]["aa-output"]
        for draft_length in (3, 7)
    ]
    if stage == "remaining-mtp":
        ratios += [document_ratio(work, path) for path in paths[:2]]
    originals = [
        study.artifact(f"mtp-k{draft_length}-outputs.jsonl", "triton")
        for draft_length in (3, 7)
    ]
    study.run_script(
        "analyze_lpx_gpu_acceptance.py",
        "--outputs",
        *originals,
        *paths,
        "--summary",
        artifact(f"{stage}-acceptance.json"),
        "--ratio",
        study.pricing_ratio(ratios),
    )


def run_stage(work, stage):
    study.validate_document_control("triton")
    if stage == "ngram":
        validate_outputs(work, artifact("plain-k0-outputs.jsonl"), [CODING], "plain", 0)
        validate_control(study.artifact("mtp-k3-smoke.jsonl", "triton"), "mtp", 3)
    paths = []
    for method, draft_length, selections in plan(stage):
        try:
            study.start_server(work, draft_length, profile="triton", method=method)
            study.write_json(
                artifact(f"{method}-k{draft_length}-server-command.json"),
                study.read_json(work / "server-process.json"),
            )
            collect(work, method, draft_length, DOCS, "warmup")
            validate_control(
                collect(work, method, draft_length, DOCS, "control"),
                method,
                draft_length,
            )
            for selection in selections:
                path = collect(work, method, draft_length, selection)
            validate_outputs(work, path, selections, method, draft_length)
            paths.append(path)
        finally:
            study.stop_server(work)
    analyze(work, stage, paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument(
        "--stage", choices=("plain", "ngram", "remaining-mtp", "all"), default="all"
    )
    parser.add_argument("--after-fp8", action="store_true")
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(ROOT):
        parser.error("work-dir must be outside the repository")
    with (args.work_dir / "followup.lock").open("w") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            pid=os.getpid(),
            stage=args.stage,
            status="waiting-for-gpu",
            after_fp8=args.after_fp8,
            started_unix=time.time(),
            completed_stages=[],
        )
        status_path = artifact("status.json")
        study.write_json(status_path, state)
        try:
            with (args.work_dir / "runner.lock").open("w") as lock:
                fcntl.flock(
                    lock, fcntl.LOCK_EX | (0 if args.after_fp8 else fcntl.LOCK_NB)
                )
                if args.after_fp8:
                    if (
                        study.read_json(study.artifact("status.json", "triton-fp8"))[
                            "status"
                        ]
                        != "complete"
                    ):
                        raise WorkloadValidationError(
                            "FP8 predecessor did not complete successfully"
                        )
                    compare_fp8(args.work_dir)
                stages = (
                    ("plain", "ngram", "remaining-mtp")
                    if args.stage == "all"
                    else (args.stage,)
                )
                for stage in stages:
                    state.update(status="running", active_stage=stage)
                    study.write_json(status_path, state)
                    run_stage(args.work_dir, stage)
                    state["completed_stages"].append(stage)
                    study.write_json(status_path, state)
        except Exception as error:
            state.update(status="failed", error=str(error), finished_unix=time.time())
            study.write_json(status_path, state)
            raise
        state.update(status="complete", finished_unix=time.time())
        study.write_json(status_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
