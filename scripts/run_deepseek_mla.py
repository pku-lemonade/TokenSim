"""DeepSeek-V3 MLA/AF simulation on the current operator-table backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.deepseek_v3_runtime import (
    MODEL_ID,
    configuration,
    simulate,
    write_profile,
)


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def run(args):
    config = configuration(
        full_model=args.full_model,
        a_tp=args.a_tp,
        f_tp=args.f_tp,
        af_batch_size=args.af_batch_size,
        data_root=args.data_root,
        device_id=args.device,
        network=args.network,
        hypothetical_f_memory_gib=args.hypothetical_f_memory_gib,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model-spec.json", config.model.to_dict())
    write_json(output / "cluster.json", config.doc)
    write_json(output / "runtime-plan.json", config.plan.summary())
    assumptions = {
        "devices": {
            worker.hardware: config.hardware.device(worker.hardware).to_dict()
            for worker in config.workers
        },
        "network": args.network,
        "link": config.hardware.link(args.network).__dict__,
        "fallback": args.fallback,
        "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
        "virtual_capacity_only": args.hypothetical_f_memory_gib is not None,
        "note": "F capacity override changes storage only; compute and memory bandwidth remain those of the base device.",
    }
    write_json(output / "hardware-assumptions.json", assumptions)
    rows = []
    for batch in args.batches:
        print(
            f"Run operator-table V3 decode: layers={config.plan.total_layers}, B={batch}",
            flush=True,
        )
        executor, measurements, compute, transfer = simulate(
            config,
            batch,
            args.context,
            steps=args.steps,
            trace=True,
            fallback=args.fallback,
        )
        step_us = sum(row["makespan_us"] for row in measurements) / args.steps
        profile = write_profile(
            output / f"decode-layers{config.plan.total_layers}-b{batch}.trace.json",
            executor,
            metadata={
                "model": MODEL_ID,
                "backend": "operator_table",
                "timing": "simulated service and queue durations; no GPU execution",
                "hardware_assumptions_file": "hardware-assumptions.json",
                "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
            },
        )
        rows.append(
            {
                "batch_size": batch,
                "context_len": args.context,
                "steps": args.steps,
                "avg_step_us": step_us,
                "tokens_per_s": batch / (step_us / 1e6),
                "tokens_per_s_per_user": 1e6 / step_us,
                "profile": str(profile.name),
                "operator_stats": compute.stats_dict(),
                "transfer": transfer.describe(),
            }
        )
    write_json(
        output / "summary.json",
        {
            "status": "operator_table_simulation",
            "model": config.model.to_dict(),
            "layers": config.plan.total_layers,
            "af_batch_size": args.af_batch_size,
            "hardware_assumptions_file": "hardware-assumptions.json",
            "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
            "fallback": args.fallback,
            "results": rows,
            "limitations": [
                "CPU event simulation; no checkpoint loading or GPU execution",
                "BF16 architectural timing; official checkpoint FP8 is not modeled",
                "Decode phase only; prefill/MTP are not a serving benchmark",
            ],
        },
    )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/deepseek-v3-operator-table"
    )
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--a-tp", type=int, default=1)
    parser.add_argument("--f-tp", type=int)
    parser.add_argument("--af-batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="a100_sxm_80g")
    parser.add_argument("--network", default="nvlink3")
    parser.add_argument("--hypothetical-f-memory-gib", type=int)
    parser.add_argument(
        "--fallback",
        choices=("table_first", "table_only", "analytical_only"),
        default="table_first",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        min(
            *args.batches,
            args.context,
            args.steps,
            args.a_tp,
            args.f_tp or 1,
            args.af_batch_size,
        )
        < 1
    ):
        raise SystemExit("batch, context, steps and parallel sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
