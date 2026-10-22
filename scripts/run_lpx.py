"""Run native fixed-batch decode with LP30 or Rubin catalog hardware."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.comm.collective_types import ALGORITHMS
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.af_trace_options import add_trace_arguments, trace_configuration
from scripts.lpx_runtime import Layout, configuration


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma-4-31b-text")
    parser.add_argument("--device", choices=("groq_lp30", "rubin"), default="groq_lp30")
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--pp", type=int, default=16)
    parser.add_argument("--pipeline-layer-counts", type=int, nargs="+")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp8"))
    parser.add_argument("--kv-cache-dtype", choices=("bf16", "fp16", "fp8"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1])
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collective-algorithm", choices=ALGORITHMS, default="ring")
    parser.add_argument(
        "--execution-mode", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    parser.add_argument("--colocated-dense-reuse", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx")
    add_trace_arguments(parser)
    return parser


def run(args):
    for name in ("tp", "pp", "context", "steps"):
        integer(getattr(args, name), name, minimum=1)
    integer(args.seed, "seed", minimum=0)
    for batch in args.batches:
        integer(batch, "batch", minimum=1)
    hardware = HardwareContext.load(args.data_root)
    model = hardware.models.resolve(args.model)
    changes = {
        key: value
        for key, value in (
            ("dtype", args.dtype),
            ("kv_cache_dtype", args.kv_cache_dtype),
        )
        if value is not None
    }
    model = replace(model, **changes)
    runtime = configuration(
        hardware,
        model,
        Layout(
            a_device=args.device,
            f_device=args.device,
            a_tp=args.tp,
            f_tp=args.tp,
            f_ep=1,
            a_pp=args.pp,
            f_pp=args.pp,
            a_collective_algorithm=args.collective_algorithm,
            f_collective_algorithm=args.collective_algorithm,
        ),
        namespace="lpx",
        micro_batch=max(args.batches),
        execution_mode=args.execution_mode,
        colocated_dense_reuse=args.colocated_dense_reuse,
        pipeline_layer_counts=args.pipeline_layer_counts,
    )
    runner = AFExperimentRunner(
        runtime,
        args.output / "points",
        fallback="analytical_only",
        seed=args.seed,
        resume=args.resume,
        trace_config=trace_configuration(args),
        trace_format=args.trace_format,
        code_paths=(ROOT / "TokenSim", Path(__file__), ROOT / "scripts/lpx_runtime.py"),
    )
    results = []
    for batch in args.batches:
        receipt = runner.run(batch, args.context, steps=args.steps)
        results.append(
            dict(
                **receipt.result["summary"],
                checkpoint_key=receipt.key,
                checkpoint_directory=str(receipt.directory.relative_to(args.output)),
            )
        )
    summary = dict(
        status="native_analytical_decode",
        model=model.to_dict(),
        device=hardware.device(args.device).to_dict(),
        runtime_plan=runtime.plan.summary(),
        results=results,
        limitations=[
            "Fixed-batch decode; prefill and end-to-end request latency are excluded.",
            "Analytical catalog assumptions; no measured LP30 operator calibration.",
            "Simulated native tokens; no text generation or model-quality evaluation.",
        ],
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = build_parser()
    try:
        summary = run(parser.parse_args(argv))
    except (ConfigurationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(summary['results'])} native decode points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
