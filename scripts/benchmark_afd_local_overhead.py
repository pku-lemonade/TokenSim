"""CPU-only implementation overhead; synthetic fixtures, never calibration evidence."""
import argparse
import gc
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from tests.test_af_native_protocol import NativeProtocolTest
from TokenSim.af.factory import build_model_runtime, simulate_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=31)
    args = parser.parse_args()
    NativeProtocolTest.setUpClass()
    fixture = NativeProtocolTest()
    results = []
    for control in (False, True):
        for trace in (False, True):
            runtime = build_model_runtime(fixture.document(communication_timing=dict(
                enabled=True, native_control_enabled=control)), hardware=fixture.base.hardware,
                model=fixture.model)
            def run():
                return simulate_decode(runtime, 8, 129, steps=args.steps, trace=trace, fallback='analytical_only')
            run()  # warm Python imports and catalog paths
            walls = []
            for _ in range(3):
                gc.collect()
                start = time.perf_counter()
                executor, rows, _, _ = run()
                walls.append(time.perf_counter()-start)
            gc.collect()
            tracemalloc.start()
            measured = run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results.append(dict(control=control, trace=trace, steps=args.steps, batch=8,
                wall_seconds=walls, median_wall_seconds=statistics.median(walls),
                peak_python_bytes=peak, recorded_events=len(executor.trace.events),
                mean_simulated_step_us=statistics.mean(row['makespan_us'] for row in rows)))
            del measured
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='local CPU overhead; synthetic uncalibrated workloads',
        memory_scope='separate tracemalloc run; Python allocations, not RSS or GPU memory',
        results=results), indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
