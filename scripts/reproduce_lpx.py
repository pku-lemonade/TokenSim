"""Run the LPX figure's workload through TokenSim analytical operators and SimPy."""

import argparse
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
from scripts.lpx.frontier import pareto
from scripts.lpx.runner import evaluate_jobs
from scripts.lpx.architecture import describe_model
from scripts.lpx.validation import power_snapshot, validate_summary
from scripts.lpx.export import write_points_csv
from scripts.lpx.hardware_overrides import reserve_memory


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def repository_state():
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
    return revision, dirty


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/sweep.json"
    )
    result.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-native-reproduction"
    )
    result.add_argument(
        "--users",
        "--batches",
        dest="batches",
        type=int,
        nargs="+",
        help="concurrent clients; --micro-batch independently controls cohort size",
    )
    result.add_argument(
        "--execution-backend", choices=("native", "sampled-replay"), default="native"
    )
    result.add_argument("--replay-samples", type=int, default=4)
    result.add_argument("--replay-prefill-samples", type=int, default=1)
    result.add_argument(
        "--replay-cache", type=Path, default=ROOT / "outputs/lpx-replay-profile-cache"
    )
    result.add_argument("--seed", type=int)
    result.add_argument("--output-tokens", type=int)
    result.add_argument("--new-tokens", type=int)
    result.add_argument("--acceptance", type=float)
    result.add_argument("--rubin-watts", type=float)
    result.add_argument("--lp30-watts", type=float)
    result.add_argument("--facility-watts", type=float, default=0.0)
    result.add_argument("--trace", action="store_true")
    result.add_argument("--retain-timing-history", action="store_true")
    result.add_argument("--lp30-reserve-bytes", type=int, default=0)
    result.add_argument("--rubin-reserve-bytes", type=int, default=0)
    result.add_argument("--plot", action="store_true")
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--skip-audit", action="store_true")
    result.add_argument("--model", default="gpt-oss-2t-wide-proxy")
    result.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="maximum independent simulation processes; does not change simulated users",
    )
    result.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed points whose code, inputs and settings still match",
    )
    result.add_argument(
        "--routing-selection", choices=("cyclic", "random_without_replacement")
    )
    result.add_argument("--draft-model", default="gpt-oss-draft-proxy")
    result.add_argument("--cases", nargs="+")
    result.add_argument("--cached-tokens", type=int)
    result.add_argument("--prefill-chunk", type=int)
    result.add_argument("--micro-batch", type=int, default=1)
    result.add_argument("--a-replicas", type=int, default=1)
    result.add_argument("--warmup-steps", type=int, default=2)
    result.add_argument("--warmup-seconds", type=float, default=0.0)
    result.add_argument(
        "--warmup-mode",
        choices=("resident_decode", "prefill_and_decode"),
        default="resident_decode",
    )
    result.add_argument("--measurement-seconds", type=float, default=0.05)
    result.add_argument("--measurement-turns", type=int, default=1)
    result.add_argument(
        "--qps",
        type=float,
        help="open arrival rate; users bound simultaneously active requests",
    )
    result.add_argument(
        "--arrival-pattern", choices=("uniform", "poisson"), default="uniform"
    )
    result.add_argument(
        "--dag-template-cache", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument(
        "--execution-mode", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    result.add_argument("--graph-kernel-us", type=float, default=0.2)
    result.add_argument(
        "--endpoint-transfer-occupancy", choices=("wire", "service"), default="wire"
    )
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
                ("seed", args.seed),
                ("output_tokens", args.output_tokens),
                ("new_tokens", args.new_tokens),
                ("cached_tokens", args.cached_tokens),
                ("prefill_chunk", args.prefill_chunk),
                ("acceptance_probability", args.acceptance),
            )
            if value is not None
        }
        workload = replace(workload, **updates)
        from TokenSim.config.validation import integer, number

        for batch in args.batches or raw["batches"]:
            integer(batch, "batch", minimum=1)
        batches = args.batches or raw["batches"]
        if len(set(batches)) != len(batches):
            raise ConfigurationError("batch/client counts must be unique")
        for value, name in (
            (args.micro_batch, "micro-batch"),
            (args.a_replicas, "A replicas"),
            (args.warmup_steps, "warm-up steps"),
            (args.jobs, "jobs"),
        ):
            integer(value, name, minimum=1)
        number(args.measurement_seconds, "measurement interval", positive=True)
        integer(args.measurement_turns, "measurement turns")
        number(args.graph_kernel_us, "graph kernel microseconds")
        if args.cases:
            unknown = set(args.cases) - {case.name for case in cases}
            if unknown:
                raise ConfigurationError(f"unknown case names: {sorted(unknown)}")
            cases = [case for case in cases if case.name in args.cases]
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
        reserves = {
            "rubin": args.rubin_reserve_bytes,
            "groq_lp30": args.lp30_reserve_bytes,
        }
        hardware = HardwareContext.load(ROOT / "data")
        target = hardware.models.resolve(args.model)
        draft = hardware.models.resolve(args.draft_model)
        hardware = reserve_memory(hardware, reserves)
        if args.routing_selection:
            target = replace(
                target,
                moe=replace(
                    target.moe,
                    routing=replace(
                        target.moe.routing, uniform_selection=args.routing_selection
                    ),
                ),
            )
            draft = replace(
                draft,
                moe=replace(
                    draft.moe,
                    routing=replace(
                        draft.moe.routing, uniform_selection=args.routing_selection
                    ),
                ),
            )
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.skip_audit:
            print("Run production AF layer audit", flush=True)
            write_json(args.output / "af-audit.json", run_audit(hardware, target))
        if args.audit_only:
            return 0
        revision, dirty = repository_state()
        jobs = []
        for case in cases:
            for batch in args.batches or raw["batches"]:
                print(f"Queue {case.name}, clients={batch}", flush=True)
                jobs.append(
                    (
                        target,
                        workload,
                        case,
                        batch,
                        dict(
                            power_overrides=power,
                            memory_reserve_overrides=reserves,
                            facility_watts=args.facility_watts,
                            trace=args.trace and batch == 1,
                            stream_metrics=not args.retain_timing_history,
                            checkpoint_output=str(args.output / ".experiments"),
                            resume=args.resume,
                            micro_batch=args.micro_batch,
                            a_dp=args.a_replicas,
                            warmup_steps=args.warmup_steps,
                            warmup_seconds=args.warmup_seconds,
                            warmup_mode=args.warmup_mode,
                            measurement_seconds=args.measurement_seconds,
                            measurement_turns=args.measurement_turns,
                            execution_mode=args.execution_mode,
                            graph_kernel_us=args.graph_kernel_us,
                            endpoint_transfer_occupancy=args.endpoint_transfer_occupancy,
                            execution_backend=args.execution_backend,
                            replay_samples=args.replay_samples,
                            replay_prefill_samples=args.replay_prefill_samples,
                            replay_cache=str(args.replay_cache),
                            handoff_mode=args.handoff_mode,
                            qps=args.qps,
                            arrival_pattern=args.arrival_pattern,
                            dag_template_cache=args.dag_template_cache,
                            draft_spec=draft,
                        ),
                    )
                )
        completed = {}
        for index, row in evaluate_jobs(jobs, processes=args.jobs):
            completed[index] = row
            rows = [completed[key] for key in sorted(completed)]
            write_json(args.output / "partial-results.json", rows)
            if row["status"] == "ok":
                print(
                    f"Done {row['case']}, clients={row['batch']}: "
                    f"{row['tokens_per_s_per_user']:.1f} TPS/user; "
                    f"{row['output_tokens_per_s']:.1f} output TPS; "
                    f"{row['power_watts'] / 1000:.1f} kW",
                    flush=True,
                )
            else:
                print(
                    f"Infeasible {row['case']}, clients={row['batch']}: {row['reason']}",
                    flush=True,
                )
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
        end_revision, end_dirty = repository_state()
        payload = {
            "schema": "tokensim.lpx_projection",
            "version": 3 if args.execution_backend == "sampled-replay" else 2,
            "executor": (
                "sampled_native_dag_replay"
                if args.execution_backend == "sampled-replay"
                else "production_ModelAFExecutor"
            ),
            "git_revision": revision,
            "git_worktree_dirty": dirty,
            "end_git_revision": end_revision,
            "end_git_worktree_dirty": end_dirty,
            "workload": asdict(workload),
            "model": target.to_dict(),
            "draft_model": draft.to_dict(),
            "architecture": describe_model(target),
            "run_settings": {
                "client_counts": list(batches),
                "stream_metrics": not args.retain_timing_history,
                "event_accounting_required": True,
                "execution_backend": args.execution_backend,
                "replay_samples": (
                    args.replay_samples
                    if args.execution_backend == "sampled-replay"
                    else None
                ),
                "result_scope": "finite_measurement_window",
                "additional_memory_reserve_bytes": reserves,
                "micro_batch": args.micro_batch,
                "simulation_processes": args.jobs,
                "resume": args.resume,
                "interactivity_statistic": "per_request_generation_tps_p50",
                "routing_selection": target.moe.routing.uniform_selection,
                "a_replicas": args.a_replicas,
                "warmup_steps": args.warmup_steps,
                "warmup_seconds": args.warmup_seconds,
                "warmup_mode": args.warmup_mode,
                "measurement_seconds": args.measurement_seconds,
                "measurement_turns": args.measurement_turns,
                "target_qps": args.qps,
                "arrival_pattern": args.arrival_pattern,
                "dag_template_cache": args.dag_template_cache,
                "execution_mode": args.execution_mode,
                "graph_device_kernel_us": args.graph_kernel_us,
                "endpoint_transfer_occupancy": args.endpoint_transfer_occupancy,
                "handoff_mode": args.handoff_mode,
            },
            "cases": [asdict(case) for case in cases],
            "hardware": {
                device: hardware.device(device).to_dict()
                for device in sorted(
                    {name for case in cases for name in case.inventory()}
                )
            },
            "power_overrides": power,
            "power_model": power_snapshot(
                hardware,
                power,
                sorted({name for case in cases for name in case.inventory()}),
            ),
            "facility_watts": args.facility_watts,
            "normalization_tokens_per_s_per_mw": baseline,
            "frontiers": frontiers,
            "results": rows,
            "limitations": [
                "Production whole-model AF DAGs, physical rank ownership, independent serving cohorts and endpoint injection reservations.",
                "Independent model/topology/precision/power assumptions; NVIDIA's undisclosed projection cannot be numerically reproduced.",
                "Native PP and combined expert/tensor groups are validated; unimplemented context-parallel layouts fail configuration.",
                "Warm-up and drain are excluded from the finite measurement window; steady-state convergence requires a separate window-sensitivity test.",
                "Layer windows and packed expert weights are executed with native per-request routing and KV byte admission.",
                "Warm prefixes are resident; prefix creation, checkpoint loading and model quality are outside the simulation.",
                "Output throughput includes new-suffix prefill, cache handoff and draft service; per-user interactivity excludes TTFT.",
                "Assumed processor TDP plus supplied facility watts; no automatic rack power or cooling estimate.",
                "External draft acceptance uses a seeded Bernoulli prefix; full gamma+1 verification positions and rollback are charged.",
                "Graph host/device costs are explicit analytical assumptions; no measured graph or compiler-vector-overlap calibration is supplied.",
            ],
        }
        payload["validation"] = validate_summary(payload)
        write_json(args.output / "summary.json", payload)
        write_points_csv(args.output / "points.csv", rows)
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
