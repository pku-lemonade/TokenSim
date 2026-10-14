"""Run the LPX figure's workload through TokenSim analytical operators and SimPy."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.audit import run_audit
from scripts.lpx.config import load_cases
from scripts.lpx.simulator import pareto
from scripts.lpx.native_simulation import point


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/sweep.json"
    )
    result.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-native-reproduction"
    )
    result.add_argument("--batches", type=int, nargs="+")
    result.add_argument("--output-tokens", type=int)
    result.add_argument("--new-tokens", type=int)
    result.add_argument("--context-samples", type=int)
    result.add_argument("--acceptance", type=float)
    result.add_argument("--rubin-watts", type=float)
    result.add_argument("--lp30-watts", type=float)
    result.add_argument("--facility-watts", type=float, default=0.0)
    result.add_argument("--trace", action="store_true")
    result.add_argument("--plot", action="store_true")
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--skip-audit", action="store_true")
    result.add_argument("--model", default="gpt-oss-2t-proxy")
    result.add_argument("--micro-batch", type=int, default=1)
    result.add_argument("--a-replicas", type=int, default=1)
    result.add_argument("--warmup-steps", type=int, default=2)
    result.add_argument("--measurement-seconds", type=float, default=0.05)
    result.add_argument(
        "--execution-mode", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    result.add_argument("--graph-kernel-us", type=float, default=0.2)
    result.add_argument(
        "--handoff-mode", choices=("full", "incremental"), default="incremental"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        raw, workload, cases = load_cases(args.config)
        updates = {
            name: value
            for name, value in (
                ("output_tokens", args.output_tokens),
                ("new_tokens", args.new_tokens),
                ("context_samples", args.context_samples),
                ("acceptance_probability", args.acceptance),
            )
            if value is not None
        }
        workload = replace(workload, **updates)
        from TokenSim.config.validation import integer, number

        for batch in args.batches or raw["batches"]:
            integer(batch, "batch", minimum=1)
        number(args.facility_watts, "facility watts")
        power = {
            device: watts
            for device, watts in (
                ("rubin", args.rubin_watts),
                ("groq_lp30", args.lp30_watts),
            )
            if watts is not None
        }
        for device, watts in power.items():
            number(watts, f"{device} power", positive=True)
        if args.audit_only and args.skip_audit:
            raise ConfigurationError("audit-only and skip-audit are mutually exclusive")
        hardware = HardwareContext.load(ROOT / "data")
        target = hardware.model(args.model)
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.skip_audit:
            print("Run production AF layer audit", flush=True)
            write_json(args.output / "af-audit.json", run_audit(hardware, target))
        if args.audit_only:
            return 0
        rows = []
        for case in cases:
            for batch in args.batches or raw["batches"]:
                print(f"Run {case.name}, batch={batch}", flush=True)
                try:
                    row = point(
                        hardware,
                        target,
                        workload,
                        case,
                        batch,
                        power_overrides=power,
                        facility_watts=args.facility_watts,
                        trace=args.trace and batch == 1,
                        micro_batch=args.micro_batch,
                        a_dp=args.a_replicas,
                        warmup_steps=args.warmup_steps,
                        measurement_seconds=args.measurement_seconds,
                        execution_mode=args.execution_mode,
                        graph_kernel_us=args.graph_kernel_us,
                        handoff_mode=args.handoff_mode,
                    )
                except ConfigurationError as exc:
                    # Capacity failures remain visible and are excluded from the frontier.
                    row = {
                        "case": case.name,
                        "mode": case.mode,
                        "batch": batch,
                        "status": "infeasible",
                        "reason": str(exc),
                    }
                rows.append(row)
        if not any(row["status"] == "ok" for row in rows):
            raise ConfigurationError("no feasible experiment points")
        if not any(row["status"] == "ok" and row["mode"] == "rubin" for row in rows):
            raise ConfigurationError("normalization requires a feasible Rubin baseline")
        baseline = max(
            row["tokens_per_s_per_mw"]
            for row in rows
            if row["status"] == "ok" and row["mode"] == "rubin"
        )
        for row in rows:
            if row["status"] == "ok":
                row["normalized_throughput"] = row["tokens_per_s_per_mw"] / baseline
        modes = ("rubin", "draft", "af", "pd")
        frontiers = {
            mode: [
                {
                    key: row[key]
                    for key in (
                        "case",
                        "batch",
                        "tokens_per_s_per_user",
                        "tokens_per_s_per_mw",
                        "normalized_throughput",
                    )
                }
                for row in pareto([row for row in rows if row["mode"] == mode])
            ]
            for mode in modes
        }
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        payload = {
            "schema": "tokensim.lpx_projection",
            "version": 2,
            "executor": "production_ModelAFExecutor",
            "git_revision": revision,
            "git_worktree_dirty": dirty,
            "workload": asdict(workload),
            "model": target.to_dict(),
            "cases": [asdict(case) for case in cases],
            "hardware": {
                device: hardware.device(device).to_dict()
                for device in ("rubin", "groq_lp30")
            },
            "power_overrides": power,
            "facility_watts": args.facility_watts,
            "normalization_tokens_per_s_per_mw": baseline,
            "frontiers": frontiers,
            "results": rows,
            "limitations": [
                "Production whole-model AF DAGs, physical rank ownership, independent serving cohorts and endpoint injection reservations.",
                "Independent model/topology/precision/power assumptions; NVIDIA's undisclosed projection cannot be numerically reproduced.",
                "Native PP and combined expert/tensor groups are validated; unimplemented context-parallel layouts fail configuration.",
                "Warm-up and drain are excluded from steady-state throughput; queue delays and per-resource utilization are exported.",
                "Layer windows and packed expert weights are executed with native per-request routing and KV byte admission.",
                "Warm prefixes are resident; prefix creation, checkpoint loading and model quality are outside the simulation.",
                "Output throughput includes new-suffix prefill, cache handoff and draft service; per-user interactivity excludes TTFT.",
                "Assumed processor TDP plus supplied facility watts; no automatic rack power or cooling estimate.",
                "External draft acceptance uses a seeded Bernoulli prefix; full gamma+1 verification positions and rollback are charged.",
                "Graph host/device costs are explicit analytical assumptions; no measured graph or compiler-vector-overlap calibration is supplied.",
            ],
        }
        write_json(args.output / "summary.json", payload)
        fields = [
            "case",
            "mode",
            "batch",
            "status",
            "tokens_per_s_per_user",
            "tokens_per_s_per_mw",
            "normalized_throughput",
            "power_watts",
            "reason",
        ]
        with (args.output / "points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} points to {args.output}", flush=True)
        if args.plot:
            from scripts.lpx.plotting import generate_plots

            generate_plots(args.output / "summary.json")
        return 0
    except (ConfigurationError, ValueError, TypeError) as exc:
        print(f"LPX experiment: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
