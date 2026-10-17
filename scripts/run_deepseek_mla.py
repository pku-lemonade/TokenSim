"""DeepSeek-V3 MLA/AF simulation on the current operator-table backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.deepseek_v3_runtime import (
    MODEL_ID,
    configuration,
    simulate,
    write_profile,
)
from scripts.af_trace_options import add_trace_arguments, trace_configuration
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.errors import ConfigurationError


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def publish_trace(source, target):
    """Publish the cached trace without copying a potentially large file."""
    temporary = None
    try:
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        temporary.unlink()
        os.link(source, temporary)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args):
    trace_config = trace_configuration(args)
    trace_kwargs = (
        {"trace_config": trace_config} if trace_config != AFTraceConfig() else {}
    )
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
    runner = AFExperimentRunner(
        config,
        output / ".experiments",
        trace_config=trace_config,
        fallback=args.fallback,
        seed=args.seed,
        resume=args.resume,
        verify=args.verify_checkpoints,
        code_paths=(
            ROOT / "TokenSim",
            Path(__file__),
            ROOT / "scripts/deepseek_v3_runtime.py",
        ),
        profile_metadata={
            "model": MODEL_ID,
            "backend": "operator_table",
            "timing": "simulated service and queue durations; no GPU execution",
            "hardware_assumptions_file": "hardware-assumptions.json",
            "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
        },
        simulate=simulate,
        write_profile=write_profile,
    )
    rows = []
    for batch in args.batches:
        print(
            f"Run operator-table V3 decode: layers={config.plan.total_layers}, B={batch}",
            flush=True,
        )
        completed = runner.run(batch, args.context, steps=args.steps)
        row = completed.result["summary"]
        if row["profile"] is not None:
            publish_trace(completed.directory / row["profile"], output / row["profile"])
        if completed.reused:
            print(f"Reuse completed experiment: B={batch}", flush=True)
        rows.append(row)
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
            **({"seed": args.seed} if args.seed != 42 else {}),
            **({"trace_config": trace_config.describe()} if trace_kwargs else {}),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse matching completed experiment points; interrupted points restart",
    )
    parser.add_argument(
        "--verify-checkpoints",
        choices=("stat", "sha256"),
        default="stat",
        help="verify output size and mtime, or additionally read and hash every output",
    )
    parser.add_argument(
        "--fallback",
        choices=("table_first", "table_only", "analytical_only"),
        default="table_first",
    )
    add_trace_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        trace_configuration(args)
        if args.seed < 0:
            raise ConfigurationError("seed must be nonnegative")
    except ConfigurationError as exc:
        parser.error(str(exc))
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
