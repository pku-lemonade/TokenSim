"""Audit measured 100K Gemma acceptance and transport it onto fixed LP30 DAGs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import number
from TokenSim.errors import WorkloadValidationError
from scripts.lpx.gpu_metrics import acceptance, output_quality
from scripts.search_lpx_native import write_json

DEFAULT_RAW = (
    "gpu-triton-fp8-mtp-k3-outputs.jsonl",
    "gpu-triton-mtp-k3-outputs.jsonl",
    "gpu-triton-mtp-k7-outputs.jsonl",
    "gpu-triton-followup-mtp-k1-outputs.jsonl",
    "gpu-triton-followup-mtp-k5-outputs.jsonl",
)


def summarize(rows):
    """Retain all turns and distinguish turn means from round-weighted means."""
    if not rows:
        raise WorkloadValidationError("Acceptance summary needs measured turns")
    tau = np.array([row["tau"] for row in rows])
    return dict(
        turns=len(rows),
        mean_tau=float(tau.mean()),
        pooled_tau=sum(row["decode_tokens"] for row in rows)
        / sum(row["drafts"] for row in rows),
        min_tau=float(tau.min()),
        max_tau=float(tau.max()),
        p50_tau=float(np.median(tau)),
        decode_tokens=sum(row["decode_tokens"] for row in rows),
        verification_rounds=sum(row["drafts"] for row in rows),
        all_quality_checks_passed=all(row["quality_ok"] for row in rows),
        rows=rows,
    )


def audit(path):
    groups = defaultdict(list)
    configurations = {}
    seen = set()
    for line in path.read_text().splitlines():
        request = json.loads(line)
        if request["set"] != "aa-docs" or request["meta"]["context"] != 100000:
            continue
        config = request["config"]
        if config["method"] != "mtp" or config["max_tokens_override"] is not None:
            raise WorkloadValidationError("Expected uncapped MTP document measurements")
        key = request["meta"]["style"]
        if key in configurations and configurations[key] != config:
            raise WorkloadValidationError(
                "Acceptance group mixes serving configurations"
            )
        configurations[key] = config
        for turn in request["turns"]:
            identity = request["id"], turn["turn"]
            if identity in seen:
                raise WorkloadValidationError(
                    "Acceptance input repeats a turn identity"
                )
            seen.add(identity)
            measured = acceptance(turn, config["K"])
            quality = output_quality(turn)
            if (
                not -config["K"] <= measured["counter_identity_residual"] <= 0
                or measured["unproposed_tokens"] != 0
                or turn["emitted_steps"] != turn["drafts"] + 1
                or turn["preemptions"] != 0
            ):
                raise WorkloadValidationError(
                    "Acceptance counters fail the run invariants"
                )
            groups[key].append(
                dict(
                    id=identity[0],
                    turn=identity[1],
                    book=request["meta"]["book"],
                    drafts=turn["drafts"],
                    **measured,
                    quality=quality,
                    quality_ok=(
                        quality["finish_reason"] == "stop"
                        and quality["empty_answer"] is False
                        and quality["repetition_flag"] is False
                    ),
                )
            )
    if not groups:
        raise WorkloadValidationError("Acceptance input has no 100K document turns")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    validation_path = path.with_name(path.stem + "-validation.json")
    receipt = None
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        if validation.get("sha256") != digest or not validation.get(
            "exact_input_coverage"
        ):
            raise WorkloadValidationError(
                "Raw acceptance input differs from its coverage receipt"
            )
        receipt = str(validation_path)
    return [
        dict(
            source=str(path),
            sha256=digest,
            context=100000,
            style=style,
            config=configurations[style],
            coverage_receipt=receipt,
            complete_sixteen_book_group=len(rows) == 16
            and len({row["book"] for row in rows}) == 16,
            **summarize(rows),
        )
        for style, rows in sorted(groups.items())
    ]


def project(program, measured, peak_per_chip):
    """Price observed committed lengths without treating GPU precision as LP30 data."""
    number(peak_per_chip, "acceptance projection chip peak", positive=True)
    if program["draft_tokens"] != measured["config"]["K"]:
        raise WorkloadValidationError("Measured and modeled draft lengths differ")
    actual = measured["config"]
    proxy = program["native_transformer"]["proxy_model"]
    modeled = dict(
        weight_precision=proxy["dtype"],
        kv_precision=proxy["kv_cache_dtype"],
        drafter_precision=program["draft_dtype"],
    )
    if proxy.get("global_kv_layout", "separate") != "separate":
        modeled["global_kv_layout"] = proxy["global_kv_layout"]
    mismatches = [key for key, value in modeled.items() if actual.get(key) != value]
    rounds = program["round_replay"]["aggregate_tps"]
    work = program["useful_work_per_committed_token"]["total"]
    peak = program["chips"] * peak_per_chip
    return dict(
        draft_tokens=program["draft_tokens"],
        chips=program["chips"],
        draft_dtype=program["draft_dtype"],
        slots=program["slots"],
        style=measured["style"],
        acceptance_source=measured["source"],
        acceptance_sha256=measured["sha256"],
        measured_turns=measured["turns"],
        modeled_precision=modeled,
        measured_precision={key: actual.get(key, "unrecorded") for key in modeled},
        precision_mismatches=mismatches,
        all_quality_checks_passed=measured["all_quality_checks_passed"],
        mean_tau=measured["mean_tau"],
        pooled_tau=measured["pooled_tau"],
        rounds_per_second=rounds,
        aggregate_tps_from_turn_mean=rounds * measured["mean_tau"],
        aggregate_tps_from_pooled_rounds=rounds * measured["pooled_tau"],
        mfu_percent_from_turn_mean=100 * rounds * measured["mean_tau"] * work / peak,
        mfu_percent_from_pooled_rounds=100
        * rounds
        * measured["pooled_tau"]
        * work
        / peak,
        convergence=program["round_replay"]["convergence"],
        scope="Empirical GPU acceptance transported onto a fixed-context LP30 DAG. Precision mismatches and different execution kernels prevent treating this as measured LP30 throughput or matching-precision acceptance.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        nargs="+",
        default=[ROOT / "experiments/lpx" / name for name in DEFAULT_RAW],
    )
    parser.add_argument("--program-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-per-chip", type=float, default=315e15 / 256)
    args = parser.parse_args(argv)
    try:
        groups = [group for path in args.raw for group in audit(path)]
        projections = []
        for path in args.program_results:
            program = json.loads(path.read_text())
            for group in groups:
                if group["config"]["K"] == program["draft_tokens"]:
                    projections.append(
                        dict(
                            program_artifact=str(path),
                            **project(program, group, args.peak_per_chip),
                        )
                    )
        write_json(
            args.output,
            dict(
                measurements=groups,
                projections=projections,
                scope="Raw 100K document counters and quality checks are recomputed without exclusions. Turn-mean and pooled-round acceptance are both retained. All-device LP30 projections remain conditional on precision and kernel transport.",
            ),
        )
        print(
            json.dumps(
                dict(
                    groups=len(groups),
                    projections=len(projections),
                    output=str(args.output),
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, WorkloadValidationError) as exc:
        print(f"LPX MFU acceptance: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
