#!/usr/bin/env python3
"""Compare identified experiment results against matching reference points."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.agentx_reference import load_experiments


def percent_delta(actual: float | None, reference: float | None) -> float | None:
    if actual is None or reference in (None, 0):
        return None
    return (actual / reference - 1) * 100


def cache_rates(
    result: dict[str, Any], backend: str
) -> tuple[float | None, float | None]:
    metrics = result.get("agentx_metrics", {})
    if "gpu_cache_hit_rate" in metrics:
        return metrics["gpu_cache_hit_rate"], metrics.get(
            "cpu_cache_hit_rate", 0 if backend == "none" else None
        )
    denominator = result.get("mooncake_cache_query_tokens")
    if backend == "mooncake" and denominator:
        gpu, cpu = result.get("mooncake_local_gpu_hit_tokens"), result.get(
            "mooncake_memory_hit_tokens"
        )
        return (
            gpu / denominator if gpu is not None else None,
            cpu / denominator if cpu is not None else None,
        )
    return result.get("prefix_cache_hit_rate"), 0.0 if backend == "none" else None


def comparison_row(
    exp: dict[str, Any],
    result: dict[str, Any],
    reference: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["agentx_metrics"]
    if (
        metadata["id"] != exp["id"]
        or reference["reference_id"] != exp["reference"]["id"]
    ):
        raise ValueError("experiment or reference identity mismatch")
    # Normalize by the actual simulated hardware, never by reference.num_gpus
    # or the old output_token_throughput_per_gpu_tps (which counted workers).
    actual_gpus = metadata["num_gpus"]
    if not isinstance(actual_gpus, int) or actual_gpus <= 0:
        raise ValueError("actual result num_gpus must be a positive integer")
    duration = metrics["profile_duration_s"]
    if duration <= 0:
        raise ValueError("positive result observation duration required")
    differences = []
    for key in (
        "num_gpus",
        "hardware",
        "concurrency",
        "parallelism",
        "cache_backend",
        "model",
    ):
        if metadata.get(key) != reference.get(key):
            differences.append(key)
    if metrics.get("concurrency") != metadata.get("concurrency"):
        raise ValueError("result concurrency disagrees with its manifest")
    if metadata.get("reference_id") != reference["reference_id"]:
        differences.append("reference_id")
    if metadata.get("profiling_window_s") != exp["profiling_window_s"]:
        differences.append("profiling_window_s")
    if metadata.get("trace_count") != exp["trace_count"]:
        differences.append("trace_count")
    comparable = not differences
    gpu, cpu = cache_rates(result, metadata["cache_backend"])
    ref = reference["metrics"]
    itl = metrics.get("itl_s", metrics.get("tpot_s", {}))
    actual = {
        "output_tput_per_gpu": metrics["output_tokens"] / duration / actual_gpus,
        "input_tput_per_gpu": metrics["input_tokens"] / duration / actual_gpus,
        "p90_ttft": metrics.get("ttft_s", {}).get("p90"),
        "p90_itl": itl.get("p90"),
        "mean_e2e": metrics.get("e2e_s", {}).get("mean"),
    }
    row = {
        "experiment_id": exp["id"],
        "reference_id": reference["reference_id"],
        "seed": metadata.get("seed"),
        "hardware": metadata["hardware"],
        "concurrency": metadata["concurrency"],
        "num_gpus": actual_gpus,
        "reference_num_gpus": reference["num_gpus"],
        "parallelism": json.dumps(metadata["parallelism"], sort_keys=True),
        "cache_backend": metadata["cache_backend"],
        "comparable": comparable,
        "incompatible_fields": ",".join(differences),
        "profile_duration_s": duration,
        "request_count": metrics["request_count"],
        "gpu_hit_rate": gpu,
        "cpu_hit_rate": cpu,
        "missing_cache_metrics": ",".join(
            k for k, v in (("gpu", gpu), ("cpu", cpu)) if v is None
        ),
        "cache_rate_denominator": metadata.get("cache_rate_denominator"),
        "reference_gpu_hit_rate": ref.get("gpu_cache_hit_rate"),
        "reference_cpu_hit_rate": ref.get("cpu_cache_hit_rate"),
        "simulated_kv_capacity_tokens": result.get("kv_cache_capacity_tokens_total"),
        "reference_logged_equivalent_kv_tokens": ref.get("kv_cache_pool_tokens"),
        "reference_kv_semantics": reference["kv_cache_pool_tokens_semantics"],
    }
    for name, value in actual.items():
        row[name] = value
        row[f"reference_{name}"] = ref.get(name)
        row[f"{name}_delta_pct"] = (
            percent_delta(value, ref.get(name)) if comparable else None
        )
    same_denominator = metadata.get("cache_rate_denominator") == "all_prompt_tokens"
    for name, value, reference_value in (
        ("gpu", gpu, ref.get("gpu_cache_hit_rate")),
        ("cpu", cpu, ref.get("cpu_cache_hit_rate")),
    ):
        row[f"{name}_hit_delta_pp"] = (
            (value - reference_value) * 100
            if comparable
            and same_denominator
            and value is not None
            and reference_value is not None
            else None
        )
    return row


def compare(
    root: Path, matrix: Path, manifest_path: Path, reference_root: Path | None = None
) -> dict[str, Any]:
    experiments = {e["id"]: e for e in load_experiments(matrix)}
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "result manifest must contain identified experiments with result_file and actual num_gpus"
        )
    rows, seen = [], set()
    for metadata in entries:
        identity = (metadata["id"], metadata.get("seed"))
        if identity in seen:
            raise ValueError(f"duplicate experiment/seed: {identity}")
        seen.add(identity)
        exp = experiments[metadata["id"]]
        result_path = root / metadata["result_file"]
        result = json.loads(result_path.read_text())
        reference_path = (reference_root or root) / exp["reference"]["metrics_file"]
        reference = json.loads(reference_path.read_text())
        row = comparison_row(exp, result, reference, metadata)
        row["result_file"] = str(result_path)
        rows.append(row)
    report = {
        "methodology": {
            "matrix": str(matrix.resolve()),
            "manifest": str(manifest_path.resolve()),
            "normalization": "actual simulated GPU count from experiment metadata",
            "missing_values": "null, never a fabricated zero",
            "capacity": "logged equivalent reference tokens are not usable capacity",
        },
        "rows": rows,
    }
    (root / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument(
        "--matrix", type=Path, default=Path("data/experiments/agentx.yaml")
    )
    parser.add_argument("--result-manifest", type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="base directory containing the matrix's reference/ paths",
    )
    args = parser.parse_args()
    compare(
        args.result_root,
        args.matrix,
        args.result_manifest or args.result_root / "run_manifest.json",
        args.reference_root,
    )
    print(args.result_root / "comparison.json")
    print(args.result_root / "comparison.csv")


if __name__ == "__main__":
    main()
