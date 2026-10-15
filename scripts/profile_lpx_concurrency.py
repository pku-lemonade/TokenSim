"""Check LPX resident capacity and sweep decode concurrency independently of turns."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.concurrency import decode_probe, resident_capacity
from scripts.lpx.config import load_cases
from scripts.lpx.native_config import runtime_configurations
from scripts.lpx.hardware_overrides import reserve_memory
from scripts.reproduce_lpx import repository_state, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/pipeline-sweep.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-concurrency-profile"
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--model", default="gpt-oss-2t-wide-proxy")
    parser.add_argument("--draft-model", default="gpt-oss-draft-proxy")
    parser.add_argument(
        "--execution-backend", choices=("native", "sampled-replay"), default="native"
    )
    parser.add_argument("--replay-samples", type=int, default=4)
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--lp30-reserve-bytes", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        for values, label in (
            (args.batches, "clients"),
            (args.micro_batches, "micro-batch"),
        ):
            for value in values:
                integer(value, label, minimum=1)
            if len(set(values)) != len(values):
                raise ConfigurationError(f"duplicate {label} values")
        integer(args.steps, "steps", minimum=1)
        integer(args.warmup_steps, "warm-up steps", minimum=1)
        if not args.capacity_only and not any(
            micro <= clients for micro in args.micro_batches for clients in args.batches
        ):
            raise ConfigurationError("no micro-batch fits a requested client count")
        _, workload, cases = load_cases(args.config)
        if args.cases:
            unknown = set(args.cases) - {case.name for case in cases}
            if unknown:
                raise ConfigurationError(f"unknown case names: {sorted(unknown)}")
            cases = [case for case in cases if case.name in args.cases]
        elif not args.capacity_only:
            cases = [case for case in cases if case.mode in {"rubin", "af"}]
        if not args.capacity_only and any(
            case.mode not in {"rubin", "af"} for case in cases
        ):
            raise ConfigurationError(
                "decode sweeps require explicit Rubin/AF cases; PD/draft support capacity-only checks"
            )
        hardware = reserve_memory(
            HardwareContext.load(ROOT / "data"), {"groq_lp30": args.lp30_reserve_bytes}
        )
        execution_factory = None
        if args.execution_backend == "sampled-replay":
            from scripts.lpx.replay_driver import ReplayConfig

            settings = dict(samples=args.replay_samples)
            if args.replay_cache is not None:
                settings["cache"] = str(args.replay_cache)
            execution_factory = ReplayConfig(workload, **settings)
        target, draft = hardware.model(args.model), hardware.model(args.draft_model)
        revision, dirty = repository_state()
        args.output.mkdir(parents=True, exist_ok=True)
        payload = dict(
            schema="tokensim.lpx_concurrency_profile",
            version=1,
            status="running",
            capacity_only=args.capacity_only,
            git_revision=revision,
            git_worktree_dirty=dirty,
            workload=asdict(workload),
            model=target.to_dict(),
            draft_model=draft.to_dict(),
            cases=[asdict(case) for case in cases],
            run_settings=dict(
                client_counts=args.batches,
                micro_batches=(
                    args.micro_batches[:1] if args.capacity_only else args.micro_batches
                ),
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                execution_backend=args.execution_backend,
                replay_samples=args.replay_samples if execution_factory else None,
                lp30_additional_reserve_bytes=args.lp30_reserve_bytes,
            ),
            capacity=[],
            decode_probes=[],
            limitations=[
                "Decode probes exclude suffix prefill, KV handoff and draft verification; they do not measure complete-turn factory throughput or efficiency.",
                "Streams stay active until every cohort completes --steps measured passes; drain is excluded. Increase --steps and check stability before claiming saturation.",
                "Capacity uses the production physical rank ledger and fixed resident cohorts; no paging, eviction or context parallelism.",
            ],
        )
        for case in cases:
            for micro in (
                args.micro_batches
                if not args.capacity_only
                else [args.micro_batches[0]]
            ):
                runtimes = runtime_configurations(
                    hardware, target, case, micro_batch=micro, draft_spec=draft
                )
                peak = (
                    workload.cached_tokens
                    + workload.new_tokens
                    + workload.output_tokens
                    + (workload.draft_tokens if case.mode == "draft" else 0)
                )
                services = {
                    name: resident_capacity(runtime, peak, micro_batch=micro)
                    for name, runtime in runtimes.items()
                }
                maximum = min(
                    value["maximum_resident_clients"] for value in services.values()
                )
                capacity = dict(
                    case=case.name,
                    micro_batch=micro,
                    maximum_resident_clients=maximum,
                    services=services,
                )
                payload["capacity"].append(capacity)
                print(
                    f"Capacity {case.name}, micro={micro}: {maximum} resident clients",
                    flush=True,
                )
                write_json(args.output / "profile.json", payload)
                if args.capacity_only:
                    continue
                for clients in sorted(args.batches):
                    if micro > clients:
                        continue
                    if clients > maximum:
                        row = dict(
                            case=case.name,
                            clients=clients,
                            micro_batch=micro,
                            status="infeasible",
                            reason="resident capacity",
                        )
                    else:
                        print(
                            f"Probe {case.name}, clients={clients}, micro={micro}",
                            flush=True,
                        )
                        started = perf_counter()
                        row = decode_probe(
                            runtimes["target"],
                            clients,
                            micro_batch=micro,
                            context=workload.cached_tokens + workload.new_tokens,
                            steps=args.steps,
                            warmup_steps=args.warmup_steps,
                            seed=workload.seed,
                            maximum_context=peak,
                            execution_factory=execution_factory,
                        )
                        row.update(
                            case=case.name,
                            status="ok",
                            host_execution_s=perf_counter() - started,
                        )
                        print(
                            f"Done {case.name}, clients={clients}, micro={micro}: {row['decode_tokens_per_s']:.1f} decode TPS; {row['tokens_per_s_per_user']:.1f} TPS/user",
                            flush=True,
                        )
                    payload["decode_probes"].append(row)
                    write_json(args.output / "profile.json", payload)
        payload["end_git_revision"], payload["end_git_worktree_dirty"] = (
            repository_state()
        )
        payload["status"] = "complete"
        write_json(args.output / "profile.json", payload)
        return 0
    except (ConfigurationError, OSError, ValueError, TypeError) as exc:
        print(f"LPX concurrency profile: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
