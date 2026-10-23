"""Find conditional minimum LP30 counts with native decode DAG samples."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.fidelity import load as load_fidelity, with_fidelity
from scripts.lpx.token_units import TokenUnits
from scripts.lpx.partition import capacity_partition
from scripts.lpx.native_search import (
    WORKLOADS,
    layouts,
    screen,
    summarize,
)
from scripts.lpx.replay_profiles import source_identity


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verification_candidates(rows, summary, limit):
    choices = []

    def add(row):
        if row and row["layout"] not in [r["layout"] for r in choices]:
            choices.append(row)

    add(summary["common_3400"])
    add(summary["published_targets"])
    threshold = summary["common_3400"]
    smaller = [
        r
        for r in rows
        if r["status"] == "ok" and threshold and r["lp30"] < threshold["lp30"]
    ]
    add(max(smaller, key=lambda r: r["acceptance"]["common_margin"], default=None))
    add(summary["fastest"])
    add(
        next(
            (
                r
                for r in rows
                if r["status"] == "ok"
                and r["layout"]["a_tp"] == 16
                and r["layout"]["a_pp"] == 16
            ),
            None,
        )
    )
    return choices[:limit]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-native-minimum"
    )
    parser.add_argument("--model", default="gemma-4-31b-text")
    parser.add_argument("--tp", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--pp", nargs="+", type=int)
    parser.add_argument(
        "--kvp",
        nargs="+",
        type=int,
        default=[1],
        help="KV-parallel sizes that factor each stage TP (Helix-style)",
    )
    parser.add_argument(
        "--kv-parallel-layers", choices=["full_attention", "all"], default=None
    )
    parser.add_argument("--kv-append-block", type=int, default=None)
    parser.add_argument("--max-chips", type=int, default=256)
    parser.add_argument("--weight-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--kv-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--kernel-launch-us", type=float, default=0.2)
    parser.add_argument(
        "--execution-model", choices=["logical", "compiled"], default="logical"
    )
    parser.add_argument("--instruction-floor-us", type=float, default=0.0)
    parser.add_argument("--invocation-us", type=float, default=0.0)
    parser.add_argument("--c2c-hop-us", type=float)
    parser.add_argument("--fidelity-config", type=Path)
    parser.add_argument(
        "--pipeline-partition", choices=["balanced", "capacity"], default="balanced"
    )
    parser.add_argument("--collective-launch-us", type=float, default=0.5)
    parser.add_argument(
        "--colocated-dense-reuse", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verify-limit", type=int, default=4)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        integer(args.max_chips, "maximum LP30 chips", minimum=1)
        if args.max_chips > 256:
            raise ConfigurationError(
                "native LPX search is limited to one 256-chip rack"
            )
        integer(args.verify_limit, "verification candidate limit")
        integer(args.requests, "measured requests", minimum=1)
        integer(args.warmup_requests, "warm-up requests")
        from TokenSim.comm.streaming import StreamingConfig

        StreamingConfig(collective_launch_us=args.collective_launch_us)
        execution = ExecutionAssumptions(
            mode=args.execution_model,
            kernel_launch_us=args.kernel_launch_us,
            instruction_floor_us=args.instruction_floor_us,
            invocation_us=args.invocation_us,
            c2c_hop_us=args.c2c_hop_us,
        )
        hardware = with_execution(HardwareContext.load(ROOT / "data"), execution)
        fidelity = load_fidelity(args.fidelity_config) if args.fidelity_config else {}
        if fidelity:
            hardware = with_fidelity(hardware, fidelity)
        units = TokenUnits(**fidelity.get("token_units", {}))
        model = replace(
            hardware.models.resolve(args.model),
            dtype=args.weight_dtype,
            kv_cache_dtype=args.kv_dtype,
        )
        candidates = list(layouts(model, args.tp, args.max_chips, args.pp, args.kvp))
        kv_settings = {
            name: value
            for name, value in (
                ("kv_parallel_layers", args.kv_parallel_layers),
                ("kv_append_block", args.kv_append_block),
            )
            if value is not None
        }
        if not candidates:
            raise ConfigurationError("search contains no candidate layouts")
        runner = ExperimentRunner(
            args.output / "points",
            inputs=(ROOT / "data",),
            code_paths=(
                ROOT / "TokenSim",
                ROOT / "scripts/lpx",
                Path(__file__),
                ROOT / "scripts/lpx/replay_engine.cpp",
            ),
            resume=args.resume,
        )
        settings = dict(
            kernel_launch_us=args.kernel_launch_us,
            collective_launch_us=args.collective_launch_us,
            colocated_dense_reuse=args.colocated_dense_reuse,
            execution=execution.report(),
            pipeline_partition=args.pipeline_partition,
            fidelity=fidelity,
        )
        if args.kvp != [1] or kv_settings:
            settings["kv_parallel"] = dict(sizes=sorted(set(args.kvp)), **kv_settings)
        started, rows, receipts = perf_counter(), [], []
        partition_cache = {}

        def run(layout, verify=False):
            parameters = dict(
                model=model.to_dict(),
                layout=asdict(layout),
                settings=settings,
                verify=verify,
            )
            if verify:
                parameters.update(requests=args.requests, warmup=args.warmup_requests)

            def execute(directory):
                partition = None
                if args.pipeline_partition == "capacity":
                    try:
                        partition = capacity_partition(
                            hardware,
                            model,
                            layout,
                            max(
                                sum(units.native_lengths(i, o)) for i, o, _ in WORKLOADS
                            ),
                            partition_cache,
                            reuse=args.colocated_dense_reuse,
                            kv_settings=kv_settings,
                        )
                    except ConfigurationError as exc:
                        return dict(
                            layout=asdict(layout),
                            lp30=layout.a_ranks,
                            status="infeasible",
                            reason=str(exc),
                        )
                result = screen(
                    hardware,
                    model,
                    layout,
                    launch_us=args.collective_launch_us,
                    reuse=args.colocated_dense_reuse,
                    verify=verify,
                    requests=args.requests,
                    warmup=args.warmup_requests,
                    invocation_us=args.invocation_us,
                    layer_counts=partition["layer_counts"] if partition else None,
                    token_units=fidelity.get("token_units"),
                    kv_settings=kv_settings,
                )
                if partition is not None:
                    result["partition_evidence"] = partition
                return result

            receipt = runner.run(parameters, execute)
            receipts.append(
                dict(
                    key=receipt.key,
                    result_file=str(receipt.directory / "result.json"),
                    reused=receipt.reused,
                    verified=verify,
                )
            )
            result = receipt.result
            print(
                json.dumps(
                    dict(
                        tp=layout.a_stage_ranks,
                        **({"kvp": layout.a_cp} if layout.a_cp > 1 else {}),
                        pp=layout.a_pp,
                        lp30=layout.a_ranks,
                        status=result["status"],
                        verified=verify,
                        reused=receipt.reused,
                        tps=[
                            round(r["generation_tps"], 3)
                            for r in result.get("workloads", [])
                        ],
                    )
                ),
                flush=True,
            )
            return result

        for layout in candidates:
            rows.append(run(layout))
            write_json(
                args.output / "progress.json",
                dict(completed=len(rows), total=len(candidates), rows=rows),
            )
        summary = summarize(rows)
        verified = [
            run(Layout(**row["layout"]), verify=True)
            for row in verification_candidates(rows, summary, args.verify_limit)
        ]
        result = dict(
            schema_version=2,
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=source_identity(),
            model=model.to_dict(),
            settings=settings,
            search_space=dict(
                tp=sorted(set(args.tp)),
                **({"kvp": sorted(set(args.kvp))} if args.kvp != [1] else {}),
                pp=args.pp or "all nonempty stage counts",
                pipeline_partition=args.pipeline_partition,
                max_chips=args.max_chips,
                max_pp=model.num_layers,
            ),
            workloads=[
                dict(
                    input_tokens=i,
                    output_tokens=o,
                    common_target_tps=3400,
                    published_target_tps=t,
                    target_scope=(
                        "user target" if i == 1000 else "published generation TPS"
                    ),
                )
                for i, o, t in WORKLOADS
            ],
            target_source="https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/",
            method="native_DAG_refinement_adaptive_context_samples",
            scope=[
                "Pure LPX, one serial equal-length client; token ratios are recorded per workload.",
                "TTFT is excluded; serial replay begins with resident prompt KV and the first output.",
                "Memory scope is reported per point; complete prefill is validated separately.",
                "One rack; one balanced or capacity-minimizing contiguous partition per PP count.",
                "Capacity partitioning minimizes peak memory, not speed over all possible partitions.",
                "TP above 16 uses whole-buffer collectives; configured fabric refinement still applies.",
                "Minimum counts depend on unmeasured startup, efficiency and network assumptions.",
            ],
            rows=rows,
            summary=summary,
            verified=verified,
            receipts=receipts,
            host_seconds=perf_counter() - started,
        )
        write_json(args.output / "results.json", result)
        print(
            json.dumps(
                dict(
                    summary={
                        key: (
                            dict(
                                lp30=value["lp30"],
                                tp=value["layout"]["a_tp"] * value["layout"]["a_cp"],
                                kvp=value["layout"]["a_cp"],
                                pp=value["layout"]["a_pp"],
                            )
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in summary.items()
                    },
                    host_seconds=result["host_seconds"],
                )
            ),
            flush=True,
        )
        return 0
    except (ConfigurationError, OSError) as exc:
        print(f"LPX native search: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
