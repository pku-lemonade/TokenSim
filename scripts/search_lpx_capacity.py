"""Search independent native LPX layouts for near-zero admission-wait QPS."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from multiprocessing import get_context

from TokenSim.hardware import HardwareContext
from TokenSim.config.validation import integer, number
from scripts.lpx.config import load_cases, with_collective_algorithms
from scripts.lpx.hardware_overrides import communication_hypothesis
from scripts.lpx.capacity import search
from scripts.lpx.runner import evaluate
from scripts.reproduce_lpx import parser, write_json, repository_state


def run_search(job):
    target, draft, workload, case, users, options = job
    settings = dict(
        communication_overrides={
            key: value
            for key, value in (
                ("latency_us", options.get("c2c_latency_us")),
                ("launch_us", options.get("collective_launch_us")),
            )
            if value is not None
        },
        result_format="reference",
        execution_backend=options["execution_backend"],
        replay_samples=options["replay_samples"],
        replay_prefill_samples=options["replay_prefill_samples"],
        replay_cache=str(options["replay_cache"]),
        draft_spec=draft,
        stream_metrics=not options["retain_timing_history"],
        micro_batch=options["micro_batch"],
        ffn_batching=options["ffn_batching"],
        ffn_batch_wait_us=options["ffn_batch_wait_us"],
        a_dp=options["a_replicas"],
        warmup_steps=options["warmup_steps"],
        warmup_seconds=options["warmup_seconds"],
        execution_mode=options["execution_mode"],
        graph_kernel_us=options["graph_kernel_us"],
        endpoint_transfer_occupancy=options["endpoint_transfer_occupancy"],
        handoff_mode=options["handoff_mode"],
        checkpoint_output=options["checkpoint_output"],
        resume=options["resume"],
        dag_template_cache=options["dag_template_cache"],
        warmup_mode=options["warmup_mode"],
        facility_watts=options["facility_watts"],
        power_overrides={
            k: v
            for k, v in (
                ("rubin", options["rubin_watts"]),
                ("groq_lp30", options["lp30_watts"]),
            )
            if v is not None
        },
        memory_reserve_overrides={
            "rubin": options["rubin_reserve_bytes"],
            "groq_lp30": options["lp30_reserve_bytes"],
        },
    )
    isolated = evaluate(
        (
            target,
            workload,
            case,
            1,
            dict(settings, measurement_turns=1, measurement_seconds=1e-9),
        )
    )
    if isolated["status"] != "ok":
        return dict(case=case.name, users=users, status="infeasible", isolated=isolated)
    latency = isolated["request_statistics"]["latency_s"]["mean"]
    settings["warmup_seconds"] = max(settings["warmup_seconds"], 1.0, 10 * latency)

    def trial(qps, multiplier):
        result = evaluate(
            (
                target,
                workload,
                case,
                users,
                dict(
                    settings,
                    qps=qps,
                    arrival_pattern=options["arrival_pattern"],
                    measurement_turns=0,
                    measurement_seconds=max(
                        options["measurement_seconds"],
                        options["duration_multiple"] * latency,
                    )
                    * multiplier,
                ),
            )
        )
        print(
            f"Probe {case.name} users={users} qps={qps:.6g} window={multiplier} status={result['status']}",
            flush=True,
        )
        return result

    result = search(
        trial,
        latency,
        resolution=options["resolution"],
        max_probes=options["max_probes"],
        maximum_multiple=options["maximum_multiple"],
    )
    return dict(case=case.name, users=users, status="ok", isolated=isolated, **result)


def main(argv=None):
    cli = parser()
    cli.description = __doc__
    cli.add_argument("--resolution", type=float, default=0.02)
    cli.add_argument("--max-probes", type=int, default=24)
    cli.add_argument("--duration-multiple", type=float, default=100)
    cli.add_argument("--maximum-multiple", type=float, default=64)
    args = cli.parse_args(argv)
    if args.qps is not None or args.audit_only or args.plot or args.trace:
        cli.error(
            "capacity search selects QPS itself and does not support audit-only, plot or trace"
        )
    if args.micro_batch != 1:
        cli.error("open arrival capacity search requires micro-batch 1")
    integer(args.jobs, "jobs", minimum=1)
    number(args.duration_multiple, "duration multiple", positive=True)
    raw, workload, cases = load_cases(args.config)
    cases = with_collective_algorithms(
        cases, gpu=args.gpu_collective_algorithm, lpu=args.lpu_collective_algorithm
    )
    overrides = {
        name: value
        for name, value in (
            ("seed", args.seed),
            ("cached_tokens", args.cached_tokens),
            ("new_tokens", args.new_tokens),
            ("output_tokens", args.output_tokens),
            ("prefill_chunk", args.prefill_chunk),
            ("acceptance_probability", args.acceptance),
        )
        if value is not None
    }
    workload = replace(workload, **overrides)
    if args.cases:
        unknown = set(args.cases) - {c.name for c in cases}
        if unknown:
            cli.error(f"unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c.name in args.cases]
    hardware = HardwareContext.load(ROOT / "data")
    communication_hypothesis(hardware, args.c2c_latency_us, args.collective_launch_us)
    target, draft = hardware.models.resolve(args.model), hardware.models.resolve(
        args.draft_model
    )
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
    args.output.mkdir(parents=True, exist_ok=True)
    options = dict(vars(args), checkpoint_output=str(args.output / ".experiments"))
    jobs = [
        (target, draft, workload, case, users, options)
        for case in cases
        for users in args.batches or raw["batches"]
    ]
    revision, dirty = repository_state()
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = {pool.submit(run_search, job): i for i, job in enumerate(jobs)}
        completed = {}
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
            rows = [completed[i] for i in sorted(completed)]
            write_json(
                args.output / "capacity.json",
                dict(
                    git_revision=revision,
                    git_worktree_dirty=dirty,
                    workload=asdict(workload),
                    model=target.to_dict(),
                    draft_model=draft.to_dict(),
                    cases=[asdict(case) for case in cases],
                    run_settings={
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in options.items()
                    },
                    results=rows,
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
