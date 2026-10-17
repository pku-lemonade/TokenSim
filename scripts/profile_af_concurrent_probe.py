"""Measure interleaved original-size cohorts in one simulation process."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simpy

from scripts.deepseek_v3_runtime import configuration
from scripts import deepseek_v3_runtime
from TokenSim.af import factory
from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.host import host_collection
from TokenSim.af.trace.pipeline import PipelinedTraceRecorder


def peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def measure(args):
    start = perf_counter()
    config = configuration(
        a_tp=4, f_tp=32, af_batch_size=32, full_model=True, data_root=ROOT / "data"
    )
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(
        hardware=config.hardware,
        model=config.model,
        cluster_config=config.cluster,
        worker_configs=config.workers,
        placement=config.placement,
        dtype=config.model.activation_dtype,
        fallback="analytical_only",
    )
    executor = ModelAFExecutor(
        env,
        config.cluster,
        config.workers,
        compute,
        transfer,
        config.model.model_id,
        config.model.dtype,
        config.plan.total_layers,
        config.model.hidden_size,
        moe_config=config.model.moe,
        random_seed=42,
        runtime_plan=config.plan,
        trace_recorder=(
            PipelinedTraceRecorder(args.output)
            if args.trace_pipeline
            else factory.ArchivedTraceRecorder(True)
        ),
        model=config.model,
    )
    owner = next(item for item in config.plan.request_owners if item.phase == "decode")
    leader = config.plan.binding(owner.domain, owner.leader_rank)
    source = next(
        index
        for index, worker in enumerate(config.workers)
        if worker.physical_worker_ref.identity == leader.physical_identity
    )
    replica = config.plan.replica_for_instance(leader.domain, leader.instance)
    required = args.cohorts * 32 * (-(-(8192 + args.steps) // config.plan.block_size))
    if required > replica.logical_gpu_blocks:
        raise ValueError("cohorts exceed the original physical KV block capacity")
    rows, progress = [], []

    def stream(cohort):
        for step in range(args.steps):
            requests = [
                SimpleNamespace(
                    id=cohort * 32 + index,
                    is_prefill=False,
                    prefill_len=8192,
                    context_len=8192 + step,
                    generation_idx=step,
                )
                for index in range(32)
            ]
            iteration = executor.submit(source, requests, "decode")
            yield from executor.run_iteration(iteration)
            rows.append(
                dict(
                    id=iteration.iteration_id,
                    start=iteration.start_at,
                    end=iteration.end_at,
                    makespan=iteration.makespan,
                )
            )
            progress.append(
                dict(
                    completions=len(rows),
                    elapsed_seconds=perf_counter() - start,
                    peak_rss_bytes=peak_bytes(),
                    templates=len(executor.plan_builder.template_cache.templates),
                )
            )
            (args.output / "progress.json").write_text(json.dumps(progress))
            if peak_bytes() > 4_000_000_000:
                raise MemoryError("per-process 4 GB budget exceeded")
            del iteration

    for cohort in range(args.cohorts):
        env.process(stream(cohort))
    try:
        factory.run_environment(env)
        simulation = perf_counter() - start
        export_start = perf_counter()
        deepseek_v3_runtime.write_profile(args.output / "trace.json", executor)
        export_seconds = perf_counter() - export_start
        if peak_bytes() > 4_000_000_000:
            raise MemoryError("export exceeded the per-process 4 GB budget")
        cache = executor.plan_builder.template_cache
        result = dict(
            kernels=args.kernels,
            trace_pipeline=args.trace_pipeline,
            cohorts=args.cohorts,
            steps=args.steps,
            simulation_seconds=simulation,
            export_seconds=export_seconds,
            total_seconds=perf_counter() - start,
            peak_rss_bytes=peak_bytes(),
            events=len(executor.trace.events),
            retained_events=executor.trace.events.retained_count,
            rows=sorted(rows, key=lambda row: row["id"]),
            compute_requests=compute.estimate_requests,
            compute_hits=compute.estimate_cache_hits,
            transfer_requests=transfer.estimate_requests,
            transfer_hits=transfer.estimate_cache_hits,
            template_entries=len(cache.templates),
            template_requests=cache.requests,
            template_hits=cache.hits,
            required_blocks=required,
            available_blocks=replica.logical_gpu_blocks,
        )
    finally:
        executor.trace.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", default="")
    parser.add_argument(
        "--trace-pipeline",
        action="store_true",
        help="Use the production trace pipeline enabled by the DeepSeek CLI",
    )
    parser.add_argument("--cohorts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cohorts < 1 or args.steps < 1:
        parser.error("cohorts and steps must be positive")
    if "metrics" in args.kernels.split(","):
        parser.error("this diagnostic requires a complete trace")
    if args.trace_pipeline and args.kernels:
        parser.error("choose the production pipeline or diagnostic kernels")
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in filter(None, args.kernels.split(",")):
            stack.enter_context(
                importlib.import_module("scripts.af_probe_" + name).installed(
                    args.output
                )
            )
        with host_collection():
            result = measure(args)
    with (args.output / "trace.json").open("rb") as stream:
        result["trace_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "trace.json").unlink()
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
