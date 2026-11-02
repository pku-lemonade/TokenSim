"""Search native 100K Gemma decode capacity and useful matrix utilization."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from scripts.lpx import replay_engine
from scripts.lpx.fidelity import load, validate
from scripts.lpx.mfu_capacity import ResidentCapacity
from scripts.lpx.mfu_probe import MFUCase, prepare, probe, runtime_for
from scripts.lpx.partition import native_footprint
from scripts.lpx.replay_profiles import source_identity
from scripts.run_lpx_article_sweep import article_config
from scripts.search_lpx_native import write_json


def build_cache():
    source = Path(replay_engine.__file__).with_suffix(".cpp").read_bytes()
    identity = hashlib.sha256(source).hexdigest()
    folder = (
        Path(tempfile.gettempdir()) / "tokensim-mfu-engine" / identity / "scripts/lpx"
    )
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "replay_engine.cpp"
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
            stream.write(source)
            temporary = Path(stream.name)
        os.replace(temporary, destination)
    replay_engine.__file__ = str(folder / "replay_engine.py")


def execute_case(job):
    config, raw, output, warmup, measured, bounds_only = job
    build_cache()
    case = MFUCase(**raw)
    runner = ExperimentRunner(
        output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    parameters = dict(
        case=asdict(case),
        config=config,
        warmup_rounds=warmup,
        measured_rounds=measured,
        bounds_only=bounds_only,
        metric="Useful decode FLOPs over all allocated LP30 FP8 peak FLOPs.",
    )

    def run(directory):
        began = perf_counter()
        try:
            result = probe(
                config,
                case,
                output / "profiles",
                warmup_rounds=warmup,
                measured_rounds=measured,
                bounds_only=bounds_only,
            )
        except ConfigurationError as exc:
            result = dict(status="infeasible", case=asdict(case), reason=str(exc))
        result["host_seconds"] = perf_counter() - began
        return result

    receipt = runner.run(parameters, run)
    result = receipt.result
    print(
        json.dumps(
            dict(
                case=asdict(case),
                status=result["status"],
                mfu_percent=result.get("mfu_percent"),
                aggregate_tps=result.get("replay", {}).get("aggregate_tps"),
                bound_mfu_percent=result.get("bounds", {}).get("mfu_percent"),
                reused=receipt.reused,
            )
        ),
        flush=True,
    )
    return dict(
        result=result,
        receipt=dict(
            key=receipt.key,
            reused=receipt.reused,
            result_file=str(receipt.directory / "result.json"),
        ),
    )


def capacity_grid(config, args):
    rows = []
    for tp in args.tp_values:
        for kvp in (k for k in args.kvp_values if k <= tp and tp % k == 0):
            modes = args.layer_modes if kvp > 1 else ("full_attention",)
            for mode in modes:
                case = MFUCase(tp, kvp, 1, kv_parallel_layers=mode)
                try:
                    _, hardware, model = prepare(config, case)
                    seed = runtime_for(hardware, model, case, model.num_layers)
                    footprint = native_footprint(seed, 102000)
                    maximum_pp = min(model.num_layers, args.max_chips // tp)
                    if maximum_pp < 1:
                        continue
                    frontier = ResidentCapacity(footprint).frontier(
                        args.request_limit,
                        maximum_pp,
                    )
                    row = dict(
                        tp=tp,
                        kvp=kvp,
                        kv_parallel_layers=mode,
                        status="ok",
                        **frontier,
                    )
                except ConfigurationError as exc:
                    row = dict(
                        tp=tp,
                        kvp=kvp,
                        kv_parallel_layers=mode,
                        status="infeasible",
                        reason=str(exc),
                    )
                rows.append(row)
                write_json(
                    args.output / "capacity.json",
                    dict(
                        source_sha256=source_identity(),
                        config=config,
                        max_chips=args.max_chips,
                        request_limit=args.request_limit,
                        rows=rows,
                        scope="Exact contiguous-partition memory screening; timing and global optimality unverified.",
                    ),
                )
                print(json.dumps(row), flush=True)


def summarize(results):
    feasible = [row["result"] for row in results if row["result"]["status"] == "ok"]
    best = max(feasible, key=lambda row: row["mfu_percent"], default=None)
    single_rack = [row for row in feasible if row["chips"] <= 256]
    fast = [
        row
        for row in feasible
        if min(row["replay"]["per_cohort_per_request_tps"].values()) >= 3400
    ]
    return dict(
        points=len(results),
        feasible=len(feasible),
        bounded=sum(row["result"]["status"] == "bounded" for row in results),
        best_measured_mfu=best,
        best_single_rack_mfu=max(
            single_rack, key=lambda row: row["mfu_percent"], default=None
        ),
        best_at_3400_per_request=max(
            fast, key=lambda row: row["mfu_percent"], default=None
        ),
        scope="Best observed among supplied cases; heterogeneous batches, other partitions, kernels and hardware remain outside this comparison.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("probe", "bound", "capacity"), default="probe"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-maximum-mfu")
    parser.add_argument("--tp", type=int, default=32)
    parser.add_argument("--kvp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--resident-requests", type=int)
    parser.add_argument("--request-limit", type=int, default=256)
    parser.add_argument("--layer-counts", nargs="+", type=int)
    parser.add_argument(
        "--kv-parallel-layers",
        choices=("full_attention", "all"),
        default="full_attention",
    )
    parser.add_argument("--initial-spacing-us", type=float, default=0)
    parser.add_argument("--issue-interval-us", type=float, default=0)
    parser.add_argument("--attention-mfu", type=float, default=0.5)
    parser.add_argument("--inter-rack-us", type=float)
    parser.add_argument("--warmup-rounds", type=int, default=8)
    parser.add_argument("--measured-rounds", type=int, default=32)
    parser.add_argument("--cases", type=Path, help="JSON list of MFUCase objects")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--tp-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--kvp-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--layer-modes",
        nargs="+",
        choices=("full_attention", "all"),
        default=["full_attention", "all"],
    )
    parser.add_argument("--max-chips", type=int, default=1920)
    args = parser.parse_args(argv)
    try:
        for name in ("workers", "request_limit", "max_chips", "warmup_rounds"):
            integer(getattr(args, name), name, minimum=1)
        integer(args.measured_rounds, "measured rounds", minimum=2)
        config = article_config(load(args.config))
        config["compute"]["attention_mfu"] = args.attention_mfu
        if args.inter_rack_us is not None:
            config["fabric"]["inter_rack_us"] = args.inter_rack_us
        config = validate(config)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.mode == "capacity":
            capacity_grid(config, args)
            return 0
        raw_cases = (
            json.loads(args.cases.read_text())
            if args.cases
            else [
                asdict(
                    MFUCase(
                        args.tp,
                        args.kvp,
                        args.pp,
                        args.batch,
                        args.resident_requests,
                        args.request_limit,
                        args.kv_parallel_layers,
                        tuple(args.layer_counts) if args.layer_counts else None,
                        args.initial_spacing_us,
                        args.issue_interval_us,
                    )
                )
            ]
        )
        if (
            not isinstance(raw_cases, list)
            or not raw_cases
            or any(not isinstance(raw, dict) for raw in raw_cases)
        ):
            raise ConfigurationError("MFU cases must be a nonempty JSON object list")
        try:
            cases = [asdict(MFUCase(**raw)) for raw in raw_cases]
        except TypeError as exc:
            raise ConfigurationError(f"invalid MFU case fields: {exc}") from exc
        if any(case["tp"] * case["pp"] > args.max_chips for case in cases):
            raise ConfigurationError("MFU case exceeds the declared chip search limit")
        jobs = [
            (
                config,
                case,
                args.output,
                args.warmup_rounds,
                args.measured_rounds,
                args.mode == "bound",
            )
            for case in cases
        ]
        results = []
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(execute_case, jobs):
                    results.append(result)
                    write_json(
                        args.output / "results.json",
                        dict(results=results, summary=summarize(results)),
                    )
        else:
            for job in jobs:
                results.append(execute_case(job))
                write_json(
                    args.output / "results.json",
                    dict(results=results, summary=summarize(results)),
                )
        return 0
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"LPX MFU search: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
