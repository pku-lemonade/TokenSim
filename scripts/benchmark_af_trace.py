"""Trace execution/export baseline: python -m scripts.benchmark_af_trace --output DIR.

Timing excludes observation and memory instrumentation. Memory uses a separate run.
Standard serving uses the catalog's full 61-layer V3 model and 1A/32F cluster.
"""

import argparse
import gc
import hashlib
import json
import tracemalloc
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time

import simpy

from scripts.deepseek_v3_runtime import configuration
from tests.test_af_engine_integration import _run_request
from tests.test_af_trace import _valid_events
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.models import AFIteration
from TokenSim.af.trace import AFTraceRecorder
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import reset_g_time
from util.results import get_af_stats


class AccessList(list):
    visits = 0

    def __iter__(self):
        for item in super().__iter__():
            self.visits += 1
            yield item

    def __getitem__(self, index):
        self.visits += 1
        return super().__getitem__(index)


def lookup_visits(steps):
    recorder = AFTraceRecorder()
    recorder.events = AccessList()
    for step in range(steps):
        for event in _valid_events():
            recorder.record(replace(event, iteration_id=step))
        recorder.events_for(step)
        recorder.validate_invariants(step)
        recorder.summarize(AFIteration(step, "decode", (), (), 0, 4))
    return dict(events=len(recorder.events), visits=recorder.events.visits)


def standard_engine(trace):
    reset_g_time()
    config = configuration(full_model=True, af_batch_size=1)
    doc = json.loads(Path("data/clusters/deepseek_v3/1a32f.json").read_text())
    doc["af"]["trace_enabled"] = trace
    runtime = build_model_runtime(doc, hardware=config.hardware, model=config.model)
    env = simpy.Environment()
    engine = LLMEngine(
        env=env, block_size=16, batching="paged-attn",
        kv_transfer_config=KVTransferConfig.default(),
        psla_config=PSLAConfig.from_file("data/psla/deepseek-v3.json"),
        cluster_config=runtime.cluster, hardware=config.hardware,
        prefill_worker_pool_type="round_robin", decode_worker_pool_type="round_robin",
        max_parallem_sum=8, latency_fallback="analytical_only",
    )
    return env, engine


def run(trace, memory=False):
    gc.collect()
    if memory:
        tracemalloc.start()
    wall, cpu = perf_counter(), process_time()
    env, engine = standard_engine(trace)
    request = _run_request(env, engine, decode_len=2)
    execution = dict(wall_s=perf_counter()-wall, cpu_s=process_time()-cpu)
    if memory:
        execution["retained_bytes"], execution["peak_bytes"] = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    wall, cpu = perf_counter(), process_time()
    payload = json.dumps(dict(
        completed=request.is_done, completion_time=request.total_time,
        ttft=request.prefill_latency, decode_time_sum=request.decode_time_sum,
        stats=get_af_stats(engine),
    ), ensure_ascii=False)
    export = dict(wall_s=perf_counter()-wall, cpu_s=process_time()-cpu)
    if memory:
        export["retained_bytes"], export["peak_bytes"] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return payload, dict(execution=execution, export=export,
                        event_count=len(engine.af_executor.trace.events),
                        sha256=hashlib.sha256(payload.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"lookup": {steps: lookup_visits(steps) for steps in (8, 16, 32)}, "cases": {}}
    for trace in (True, False):
        timings = []
        for repeat in range(args.repeats):
            payload, metrics = run(trace)
            timings.append(metrics)
            (args.output / f"trace-{trace}.json").write_text(payload, encoding="utf-8")
            print(trace, repeat, metrics, flush=True)
        _, memory = run(trace, memory=True)
        report["cases"][str(trace)] = dict(timings=timings, memory=memory)
        (args.output / "measurements.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
