"""Explore Gemma/AA-shaped hardware hypotheses without changing AF defaults."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.lpx.aa_cost import Costs
from scripts.lpx.aa_model import GemmaText
from scripts.lpx.aa_search import replay, search


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/aa-gemma4.json"
    )
    result.add_argument("--workloads", nargs="+", help="Workload names from the config")
    result.add_argument(
        "--weight-dtypes", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"]
    )
    result.add_argument(
        "--collectives",
        nargs="+",
        choices=("native-ring", "scheduled-direct"),
        default=["native-ring", "scheduled-direct"],
    )
    result.add_argument("--kv-dtype", choices=("bf16", "fp8"), default="bf16")
    result.add_argument("--rho-in", type=float, default=1.0)
    result.add_argument("--rho-out", type=float, default=1.0)
    result.add_argument("--kernel-us", type=float, default=0.2)
    result.add_argument("--timing-window", choices=("full", "tail80"), default="full")
    result.add_argument("--reserve-fraction", type=float, default=0.05)
    result.add_argument("--cross-latency-us", type=float)
    result.add_argument("--prefill-chunk", type=int, default=256)
    result.add_argument("--max-lpu", type=int, default=1024)
    result.add_argument("--max-gpu", type=int, default=144)
    result.add_argument("--requests", type=int, default=50)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-aa-20261019")
    result.add_argument("--no-resume", action="store_true")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("rho_in", "rho_out"):
        number(getattr(args, name), name, positive=True)
    number(args.kernel_us, "kernel overhead")
    number(args.reserve_fraction, "memory reserve", minimum=0, maximum=0.99)
    if args.cross_latency_us is not None:
        number(args.cross_latency_us, "cross latency")
    for name in ("requests", "prefill_chunk", "max_lpu", "max_gpu"):
        integer(getattr(args, name), name, minimum=1)
    integer(args.warmup, "warmup")
    raw = json.loads(args.config.read_text())
    if raw["schema_version"] != 1:
        raise ConfigurationError("unsupported AA exploration schema")
    model = GemmaText(**raw["model"])
    for workload in raw["workloads"]:
        integer(workload["input_tokens"], "input tokens", minimum=1)
        integer(workload["output_tokens"], "output tokens", minimum=2)
        number(workload["target_tps"], "target TPS", positive=True)
    if args.workloads:
        unknown = set(args.workloads) - {w["name"] for w in raw["workloads"]}
        if unknown:
            raise ConfigurationError(f"unknown AA workloads: {sorted(unknown)}")
        raw["workloads"] = [w for w in raw["workloads"] if w["name"] in args.workloads]
    args.output.mkdir(parents=True, exist_ok=True)
    hardware = HardwareContext.load(ROOT / "data")
    results = []
    started = time.perf_counter()
    for dtype in args.weight_dtypes:
        for collective in args.collectives:
            name = f"{dtype}-{collective}"
            parameters = dict(
                model=asdict(model),
                workloads=raw["workloads"],
                weight_dtype=dtype,
                collective=collective,
                settings={
                    key: value
                    for key, value in vars(args).items()
                    if key
                    not in {
                        "output",
                        "no_resume",
                        "config",
                        "weight_dtypes",
                        "collectives",
                    }
                },
            )
            runner = ExperimentRunner(
                args.output / "checkpoints" / name,
                inputs=(ROOT / "data/devices", ROOT / "data/topologies", args.config),
                code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
                resume=not args.no_resume,
            )

            def execute(directory):
                costs = Costs(
                    hardware,
                    model,
                    weight_dtype=dtype,
                    collective=collective,
                    kv_dtype=args.kv_dtype,
                    kernel_us=args.kernel_us,
                    rho_in=args.rho_in,
                    rho_out=args.rho_out,
                    chunk=args.prefill_chunk,
                    cross_latency_us=args.cross_latency_us,
                    timing_window=args.timing_window,
                )
                result, cases = search(
                    costs,
                    raw["workloads"],
                    max_lpu=args.max_lpu,
                    max_gpu=args.max_gpu,
                    reserve=args.reserve_fraction,
                )
                result["replays"] = {
                    case_name: replay(
                        costs,
                        case,
                        raw["workloads"],
                        requests=args.requests,
                        warmup=args.warmup,
                    )
                    for case_name, case in cases.items()
                }
                return result

            result = runner.run(parameters, execute).result
            write_json(args.output / f"{name}.json", result)
            results.append(
                dict(
                    profile=name,
                    selections=result["selections"],
                    feasible_points=len(result["points"]),
                    rejected=len(result["rejected"]),
                )
            )
            for mode, selection in result["selections"].items():
                best = selection["best"]
                minimum = selection["minimum_cards"]
                print(
                    json.dumps(
                        dict(
                            profile=name,
                            mode=mode,
                            minimum=minimum["inventory"] if minimum else None,
                            best_tps=(
                                [
                                    round(row["predicted_tps"], 1)
                                    for row in best["benchmarks"]
                                ]
                                if best
                                else None
                            ),
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    manifest = dict(
        result_class="published-workload-hypothesis",
        strict_published_configuration_reproduction=False,
        acceptance_metric={
            "name": "generation_output_tokens_per_second",
            "full_window_formula": "(output_tokens - 1) / (last_output_s - first_output_s)",
            "includes_ttft": False,
            "includes_prefill_or_initial_kv_handoff": False,
            "ttft_limit_s": None,
            "timing_window": args.timing_window,
        },
        model=asdict(model),
        workloads=raw["workloads"],
        parameter_elements_without_norm=model.parameter_elements(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        sources=raw["sources"],
        profiles=results,
        settings=parameters["settings"],
        wall_seconds=time.perf_counter() - started,
        assumptions=[
            "Rubin companion is a user-selected hypothesis; actual benchmark inventory is undisclosed",
            "Original 50 prompts, output lengths, native/o200k ratios and chunk timestamps are unavailable",
            "1K 3400 TPS is a user target; no published 1K LPX number was found",
            "All workloads use the user-requested 3400 generation TPS threshold; published 3382/3431 remain reference values",
            "Output lengths are AA methodology minima, not original request records",
            "Token ratios are explicit sensitivity assumptions; no original text tokenizer calibration",
            "BF16 is the public text checkpoint precision; service precision is undisclosed",
            "FP8 GEMMs assume fused conversion and do not establish equal model quality",
            "No speculation, hidden reasoning, API chunking or client network costs are simulated",
            "Tail80 approximates AA's hidden-reasoning window with native token intervals; streamed answer chunks are unavailable",
            "Serial deterministic repeats do not represent 50 independent accuracy samples",
            "Scheduled direct collectives are a prototype hypothesis; native LPX still uses flat ring",
            "Single-client PP increases storage capacity; pipeline stages do not multiply token speed",
            "Minimum PP uses contiguous whole-layer partitions and a shared tied head on the first stage; final hidden states return to that stage",
            "Weights and KV must remain in SRAM; DDR5 streaming is unsupported",
            "Prefill executes chunks serially; chunk/PP overlap could reduce the reported TTFT",
            "Shared host/NIC/switch implementation costs and placement feasibility remain uncalibrated",
            "GPU/LPU efficiencies, fixed latency, graph and kernel overheads are unmeasured",
            "No latency/TTFT constraint is published, so one Rubin prefill GPU minimizes PD chip count",
        ],
    )
    write_json(args.output / "summary.json", manifest)
    print(f"Saved {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
