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
from scripts.lpx.simulator import pareto, point


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/sweep.json"
    )
    result.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-reproduction"
    )
    result.add_argument("--batches", type=int, nargs="+")
    result.add_argument("--output-tokens", type=int)
    result.add_argument("--context-samples", type=int)
    result.add_argument("--acceptance", type=float)
    result.add_argument("--rubin-watts", type=float)
    result.add_argument("--lp30-watts", type=float)
    result.add_argument("--facility-watts", type=float, default=0.0)
    result.add_argument("--trace", action="store_true")
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--skip-audit", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        raw, workload, cases = load_cases(args.config)
        updates = {
            name: value
            for name, value in (
                ("output_tokens", args.output_tokens),
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
        target = hardware.model("gpt-oss-2t-proxy")
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
            "version": 1,
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
                "Analytical projection using TokenSim AF operator composition, exact local expert workloads, collective models and SimPy resources.",
                "Independent model/topology/precision/power assumptions; NVIDIA's undisclosed projection cannot be numerically reproduced.",
                "Layer pipeline and context partitions are experiment compositions; production AF supports PP=1 and EP=1 or TP*DP.",
                "Fixed synchronous batches; no overlap between model passes or compiler vector transfers; pool-wide reservations are conservative.",
                "Decode latency integrates sampled contexts and routing histograms; prefill uses a balanced conserved histogram.",
                "Warm prefixes are resident; prefix creation, checkpoint loading and model quality are outside the simulation.",
                "Output throughput includes new-suffix prefill, cache handoff and draft service; per-user interactivity excludes TTFT.",
                "Assumed processor TDP plus supplied facility watts; no automatic rack power or cooling estimate.",
                "External draft acceptance is an independent expectation; full gamma+1 verification width is always charged.",
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
        return 0
    except (ConfigurationError, ValueError, TypeError) as exc:
        print(f"LPX experiment: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
