"""Measure each branch in a fresh interpreter, preserving all public outputs."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=ROOT)
parser.add_argument("--kernels", default="")
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--keep-trace", action="store_true")
parser.add_argument("--semantic-digest", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from scripts import run_deepseek_mla as cli
from TokenSim.af import factory

args.output.mkdir(parents=True, exist_ok=True)
metrics = dict(kernels=args.kernels, source=str(args.source), steps=args.steps, rows=[])
original_simulate, original_export = cli.simulate, cli.write_profile
original_run = factory.run_environment


def timed_run(*values, **options):
    start = perf_counter()
    try:
        return original_run(*values, **options)
    finally:
        metrics["rows"].append(
            dict(step=len(metrics["rows"]), wall_seconds=perf_counter() - start)
        )
        (args.output / "progress.json").write_text(json.dumps(metrics["rows"]))


def simulate(*values, **options):
    # Diagnostic kernels compare with the original archive/export path.
    # An empty selection measures the production CLI, including its defaults.
    if args.kernels:
        options.pop("trace_output", None)
    start = perf_counter()
    result = original_simulate(*values, **options)
    metrics["simulation_seconds"] = perf_counter() - start
    executor, rows, compute, transfer = result
    cache = executor.plan_builder.template_cache
    metrics.update(
        events=len(executor.trace.events),
        iteration_rows=rows,
        compute_requests=compute.estimate_requests,
        compute_hits=compute.estimate_cache_hits,
        transfer_requests=transfer.estimate_requests,
        transfer_hits=transfer.estimate_cache_hits,
        template_entries=len(cache.templates),
        template_requests=cache.requests,
        template_hits=cache.hits,
    )
    return result


def export(*values, **options):
    start = perf_counter()
    try:
        return original_export(*values, **options)
    finally:
        metrics["export_seconds"] = perf_counter() - start


with ExitStack() as stack:
    for name in filter(None, args.kernels.split(",")):
        stack.enter_context(
            importlib.import_module("scripts.af_probe_" + name).installed(args.output)
        )
    original_simulate, original_export = cli.simulate, cli.write_profile
    cli.simulate, cli.write_profile = simulate, export
    factory.run_environment = timed_run
    started = perf_counter()
    cli.main(
        [
            "--full-model",
            "--a-tp",
            "4",
            "--f-tp",
            "32",
            "--batches",
            "32",
            "--af-batch-size",
            "32",
            "--context",
            "8192",
            "--steps",
            str(args.steps),
            "--device",
            "a100_sxm_80g",
            "--network",
            "nvlink3",
            "--fallback",
            "analytical_only",
            "--data-root",
            str(ROOT / "data"),
            "--output",
            str(args.output),
        ]
        + (
            ["--no-resume"]
            if hasattr(cli.build_parser().parse_args([]), "resume")
            else []
        )
    )
    metrics["total_seconds"] = perf_counter() - started
metrics["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin":
    metrics["peak_rss_bytes"] *= 1024
metrics["hashes"] = {}
for path in args.output.glob("*.json"):
    if path.name in ("progress.json", "measurement.json"):
        continue
    with path.open("rb") as stream:
        metrics["hashes"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
if args.semantic_digest:
    from scripts.af_probe_digest import trace_digest

    metrics["semantic_digest"] = trace_digest(
        args.output / "decode-layers61-b32.trace.json"
    )
(args.output / "measurement.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in metrics.items()
            if k not in ("rows", "iteration_rows", "hashes")
        }
    ),
    flush=True,
)
if not args.keep_trace:
    (args.output / "decode-layers61-b32.trace.json").unlink()
    shutil.rmtree(args.output / ".experiments", ignore_errors=True)
