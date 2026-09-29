"""Measure the AF cache path without a service-record observer."""

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pstats
from time import perf_counter, process_time
from unittest.mock import patch

from TokenSim.af.executor import AFExecutor
from TokenSim.af import operator_provider
from scripts.deepseek_v3_runtime import configuration, simulate


def run(steps):
    config = configuration(full_model=True, af_batch_size=1)
    iteration_times = []
    original = AFExecutor.run_iteration

    def timed(executor, iteration):
        start, cpu = perf_counter(), process_time()
        result = yield from original(executor, iteration)
        iteration_times.append(dict(wall=perf_counter() - start, cpu=process_time() - cpu))
        return result

    start, cpu = perf_counter(), process_time()
    with patch.object(AFExecutor, "run_iteration", timed):
        executor, rows, compute, transfer = simulate(
            config, 2, 8, steps=steps, trace=False, fallback="analytical_only"
        )
    wall, cpu = perf_counter() - start, process_time() - cpu
    cache = executor.plan_builder.template_cache
    return dict(
        steps=steps, wall=wall, cpu=cpu, iterations=iteration_times, rows=rows,
        template_requests=cache.requests, template_hits=cache.hits,
        providers={name: dict(requests=p.estimate_requests, hits=p.estimate_cache_hits,
                             query_constructions=getattr(p, "query_constructions", p.estimate_requests))
                   for name, p in (("compute", compute), ("transfer", transfer))},
    )


def verify():
    """Hash complete services separately from timing; retain no large row list."""
    results = []
    initialize = operator_provider.initialize_cache
    original = AFExecutor._run_resources_stage
    for mode in ("uncached", "full-query", "fast"):
        config = configuration(full_model=True, af_batch_size=1)
        config.cluster.af.dag_template_cache = mode != "uncached"
        config.cluster.af.service_estimate_cache = mode != "uncached"
        digest = hashlib.sha256()
        count = 0

        def initialize_mode(provider, *args, **kwargs):
            initialize(provider, *args, **kwargs)
            provider.query_fast_path = mode == "fast"

        def observed(executor, block, layer, node, resources, estimate_factory):
            nonlocal count
            result = yield from original(executor, block, layer, node, resources, estimate_factory)
            ready, start, end, estimate = result
            record = dict(iteration=block.iteration_id, block=block.block_id,
                          layer=layer, node=node, ready=ready, start=start, end=end,
                          queue=start-ready, estimate=asdict(estimate))
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
            count += 1
            return result

        with patch.object(operator_provider, "initialize_cache", initialize_mode), patch.object(AFExecutor, "_run_resources_stage", observed):
            _, rows, _, _ = simulate(config, 2, 8, steps=2, trace=False, fallback="analytical_only")
        result = dict(mode=mode, services=count, sha256=digest.hexdigest(), rows=rows)
        if results:
            assert (count, result["sha256"], rows) == (results[0]["services"], results[0]["sha256"], results[0]["rows"]), "service differential failed"
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Run service differential instead of timing")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.verify:
        args.output.write_text(json.dumps(verify(), indent=2), encoding="utf-8")
        return
    records = []
    for steps in args.steps:
        for repeat in range(args.repeats):
            record = dict(repeat=repeat, **run(steps))
            records.append(record)
            args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(run, args.steps[0])
        profiler.dump_stats(str(args.output.with_suffix(".pstats")))
        with args.output.with_suffix(".profile.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(80)


if __name__ == "__main__":
    main()
