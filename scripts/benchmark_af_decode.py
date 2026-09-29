"""不带 profiler/trace/服务记录器的完整 61 层 decode 缓存对照。"""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

from scripts.deepseek_v3_runtime import configuration, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for steps in args.steps:
        reference = None
        for enabled in (False, True):
            config = configuration(full_model=True, af_batch_size=1)
            config.cluster.af.dag_template_cache = enabled
            config.cluster.af.service_estimate_cache = enabled
            start, cpu = perf_counter(), process_time()
            executor, rows, _, _ = simulate(
                config, 2, 8, steps=steps, trace=False, fallback="analytical_only"
            )
            elapsed, cpu = perf_counter() - start, process_time() - cpu
            if reference is None:
                reference = rows
            else:
                assert rows == reference, "cached decode completion differs"
            cache = executor.plan_builder.template_cache
            record = dict(
                steps=steps,
                cache=enabled,
                wall_seconds=elapsed,
                cpu_seconds=cpu,
                template_requests=cache.requests,
                template_hits=cache.hits,
            )
            records.append(record)
            print(record, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
