#!/usr/bin/env python3
"""Expand, validate, run and summarize the versioned AgentX experiment matrix."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base, override):
    result = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_matrix(path):
    document = yaml.safe_load(Path(path).read_text()) or {}
    defaults = document.get("defaults") or {}
    experiments = []
    for point in document.get("experiments") or []:
        resolved = deep_merge(defaults, point)
        resolved["id"] = point.get("id")
        resolved["groups"] = list(point.get("groups") or [])
        experiments.append(resolved)
    return document, experiments


def select(experiments, *, ids=None, group=None, hardware=None):
    ids = set(ids or [])
    result = []
    for point in experiments:
        if ids and point["id"] not in ids:
            continue
        if group and group not in point.get("groups", []):
            continue
        if hardware and point.get("hardware") != hardware:
            continue
        result.append(point)
    return result


def check(points):
    errors = []
    required = ("id", "hardware", "cluster", "concurrency", "max_num_seqs")
    for point in points:
        for key in required:
            if not point.get(key):
                errors.append(f"{point.get('id')}: missing {key}")
        for key in ("cluster", "model_config_path"):
            value = point.get(key)
            if value and not (ROOT / value).is_file():
                errors.append(f"{point['id']}: missing file {value}")
        if point.get("concurrency", 0) <= 0 or point.get("max_num_seqs", 0) <= 0:
            errors.append(f"{point.get('id')}: concurrency/max_num_seqs must be positive")
        if point.get("trace_count") != 393:
            errors.append(f"{point['id']}: trace_count must be 393 for full AgentX runs")
        if point.get("profiling_window_s") != 3600:
            errors.append(f"{point['id']}: profiling_window_s must be 3600")
    if errors:
        raise SystemExit("\n".join(errors))
    return {"points": len(points), "valid": True}


def run_point(point, result_root, backend, seed=0):
    target = Path(result_root) / f"seed{seed}" / point["id"]
    existing = target / point["hardware"] / "run_status.json"
    if existing.is_file():
        try:
            status = json.loads(existing.read_text())
            if status.get("exit_code") == 0:
                return {"id": point["id"], "exit_code": 0, "skipped": True}
        except (OSError, ValueError):
            pass
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_agentx_native.py"),
        "--case",
        point["id"],
        "--result-root",
        str(target),
        "--backend",
        backend,
        "--random-seed",
        str(seed),
    ]
    # run_agentx_native appends the hardware subdirectory below result-root.
    process = subprocess.run(command, cwd=ROOT)
    return {"id": point["id"], "exit_code": process.returncode}


def summarize(result_root, output, baseline_root=None):
    rows = []
    roots = [Path(result_root)]
    if baseline_root:
        roots.append(Path(baseline_root))
    candidates = sorted(
        result
        for root in roots
        for result in root.rglob("agentx_c*.json")
        if result.parent.name in {"b200", "b300"}
    )
    point_rows = []
    for result in candidates:
        point_dir = result.parent.parent
        data = json.loads(result.read_text())
        metrics = data["agentx_metrics"]
        manifest_path = result.parent / "run_manifest.json"
        point_id = point_dir.name
        seed = next(
            (part[4:] for part in result.parts if part.startswith("seed")),
            "0",
        )
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                point_id = manifest.get("point", {}).get("id", point_id)
                seed = str(manifest.get("simulation_seed", seed))
            except (OSError, ValueError):
                pass
        reference_path = ROOT / "data/reference" / point_id / "key_point.json"
        reference_metrics = {}
        if reference_path.is_file():
            reference_metrics = json.loads(reference_path.read_text()).get("metrics", {})
        reference_output = reference_metrics.get("output_tput_tps")
        output_tps = metrics["output_token_throughput_tps"]
        row = {
            "row_type": "seed",
            "seed": seed,
            "point_id": point_id,
            **{
                "run": point_dir.name,
                "hardware": result.parent.name,
                "concurrency": metrics.get("concurrency"),
                "output_token_throughput_tps": output_tps,
                "reference_output_tps": reference_output,
                "output_tps_error_percent": (
                    100 * (output_tps / reference_output - 1)
                    if reference_output
                    else None
                ),
                "input_token_throughput_tps": metrics["input_token_throughput_tps"],
                "p90_ttft_s": metrics["ttft_s"]["p90"],
                "reference_p90_ttft_s": reference_metrics.get("p90_ttft"),
                "p90_itl_s": metrics["tpot_s"]["p90"],
                "gpu_cache_hit_rate": metrics["gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": metrics["cpu_cache_hit_rate"],
                "request_count": metrics["request_count"],
            },
        }
        point_rows.append(row)
    rows.extend(point_rows)
    grouped = {}
    for row in point_rows:
        grouped.setdefault((row["point_id"], row["hardware"]), []).append(row)
    for (point_id, hardware), values in grouped.items():
        if len(values) < 2:
            continue
        numeric = "output_token_throughput_tps"
        samples = [float(v[numeric]) for v in values]
        aggregate = dict(values[0])
        aggregate.update(
            row_type="aggregate",
            seed="mean",
            output_token_throughput_tps=statistics.mean(samples),
            output_tps_min=min(samples),
            output_tps_max=max(samples),
            output_tps_stdev=statistics.pstdev(samples),
        )
        rows.append(aggregate)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        fieldnames = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["run"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "output": str(output)}


def plot_summary(csv_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(csv_path).open()))
    aggregate = [row for row in rows if row.get("row_type") == "aggregate"]
    if not aggregate:
        raise SystemExit("summary has no aggregate seed rows to plot")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for hardware in sorted({row["hardware"] for row in aggregate}):
        values = sorted(
            [row for row in aggregate if row["hardware"] == hardware],
            key=lambda row: int(row["concurrency"] or 0),
        )
        x = [int(row["concurrency"]) for row in values]
        axes[0].plot(
            x,
            [float(row["output_token_throughput_tps"]) for row in values],
            marker="o",
            label=hardware,
        )
        axes[1].plot(
            x,
            [float(row["p90_ttft_s"]) for row in values],
            marker="o",
            label=hardware,
        )
    axes[0].set(xlabel="concurrency", ylabel="output token/s", title="Native throughput")
    axes[1].set(xlabel="concurrency", ylabel="p90 TTFT (s)", title="Native p90 TTFT")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {"output": str(output), "rows": len(aggregate)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "data/experiments/agentx.yaml")
    parser.add_argument("--action", choices=("list", "check", "expand", "run", "summary", "plot"), default="list")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--group")
    parser.add_argument("--hardware")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Existing seed0 checkpoint root to include in a multi-seed summary.",
    )
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/agentx-matrix")
    parser.add_argument(
        "--backend",
        choices=("simple_cpu", "mooncake", "hbm"),
        default="simple_cpu",
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated simulation seeds; use 0,1,2 for the native random MTP matrix.",
    )
    args = parser.parse_args(argv)
    _, experiments = load_matrix(args.matrix)
    points = select(experiments, ids=args.ids, group=args.group, hardware=args.hardware)
    if not points:
        raise SystemExit("no experiments matched the selection")
    if args.action == "list":
        print(json.dumps([{"id": p["id"], "hardware": p["hardware"], "concurrency": p["concurrency"], "groups": p.get("groups", [])} for p in points], indent=2))
    elif args.action == "check":
        print(json.dumps(check(points), indent=2))
    elif args.action == "expand":
        target = args.output or Path("expanded-agentx-matrix.json")
        target.write_text(json.dumps(points, indent=2) + "\n")
        print(target)
    elif args.action == "run":
        check(points)
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
            futures = [
                pool.submit(run_point, p, args.result_root, args.backend, seed)
                for p in points
                for seed in seeds
            ]
            results = [future.result() for future in as_completed(futures)]
        print(json.dumps(sorted(results, key=lambda item: item["id"]), indent=2))
    elif args.action == "summary":
        print(
            json.dumps(
                summarize(
                    args.result_root,
                    args.output or args.result_root / "summary.csv",
                    args.baseline_root,
                ),
                indent=2,
            )
        )
    else:
        summary_path = args.result_root / "summary.csv"
        summarize(args.result_root, summary_path, args.baseline_root)
        print(json.dumps(plot_summary(summary_path, args.output or args.result_root / "summary.png"), indent=2))


if __name__ == "__main__":
    main()
