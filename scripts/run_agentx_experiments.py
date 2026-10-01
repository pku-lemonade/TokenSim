#!/usr/bin/env python3
"""Expand, validate, run and summarize the versioned AgentX experiment matrix."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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


def run_point(point, result_root, backend):
    target = Path(result_root) / point["id"]
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
    ]
    # run_agentx_native appends the hardware subdirectory below result-root.
    process = subprocess.run(command, cwd=ROOT)
    return {"id": point["id"], "exit_code": process.returncode}


def summarize(result_root, output):
    rows = []
    root = Path(result_root)
    candidates = (
        sorted(root.glob("*/b200/agentx_c196.json"))
        + sorted(root.glob("*/b300/agentx_c384.json"))
        + sorted(root.glob("b200/agentx_c196.json"))
        + sorted(root.glob("b300/agentx_c384.json"))
    )
    for result in candidates:
        point_dir = result.parent.parent
        data = json.loads(result.read_text())
        metrics = data["agentx_metrics"]
        rows.append(
            {
                "run": point_dir.name,
                "hardware": result.parent.name,
                "output_token_throughput_tps": metrics["output_token_throughput_tps"],
                "input_token_throughput_tps": metrics["input_token_throughput_tps"],
                "p90_ttft_s": metrics["ttft_s"]["p90"],
                "p90_itl_s": metrics["tpot_s"]["p90"],
                "gpu_cache_hit_rate": metrics["gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": metrics["cpu_cache_hit_rate"],
                "request_count": metrics["request_count"],
            }
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "data/experiments/agentx.yaml")
    parser.add_argument("--action", choices=("list", "check", "expand", "run", "summary"), default="list")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--group")
    parser.add_argument("--hardware")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/agentx-matrix")
    parser.add_argument("--backend", choices=("simple_cpu", "hbm"), default="simple_cpu")
    parser.add_argument("--max-parallel", type=int, default=1)
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
        with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
            futures = [pool.submit(run_point, p, args.result_root, args.backend) for p in points]
            results = [future.result() for future in as_completed(futures)]
        print(json.dumps(sorted(results, key=lambda item: item["id"]), indent=2))
    else:
        print(json.dumps(summarize(args.result_root, args.output or args.result_root / "summary.csv"), indent=2))


if __name__ == "__main__":
    main()
