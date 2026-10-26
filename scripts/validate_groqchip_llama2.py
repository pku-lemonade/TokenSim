"""Simulate Llama-2 70B batch-one decode on GroqChip v1 and compare with GroqCloud.

Groq reported more than 300 tokens per second per user for Llama-2 70B in
November 2023, and Artificial Analysis measured a median of 241 tokens per
second on GroqCloud in February 2024.  Neither source states the chip count,
precision, parallel layout or load.  This script prices one native decode step
for several layouts, all-reduce algorithms and hop latencies.  See
docs/groqchip-v1-llama2-validation.md for the recorded results.
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration
from scripts.search_lpx_native import write_json

LAYERS = 80
# Mean decode context of a 100-token prompt with 200 output tokens.
CONTEXT = 200
PUBLISHED_TPS = dict(groq_claim=300, artificial_analysis=241)
LAYOUTS = ((8, 41, "int8"), (8, 72, "int8"), (16, 36, "int8"), (16, 41, "fp16"))
ALGORITHMS = ("ring", "direct")
# Catalog link latency and the 300 ns LPUv1 hop latency reported by SHIP.
HOP_LATENCIES_US = (0.5, 0.3)


def layer_counts(stages):
    """One layer on the first and last stage, which hold the embedding and head."""
    base, extra = divmod(LAYERS - 2, stages - 2)
    return [1] + [base + (i < extra) for i in range(stages - 2)] + [1]


def hardware_with_hop(latency_us):
    hardware = HardwareContext.load(ROOT / "data")
    links = LinkCatalog(
        tuple(
            (
                replace(link, latency_us=latency_us)
                if link.link_id == "groq_c2c_25g"
                else link
            )
            for link in hardware.links
        )
    )
    return replace(hardware, links=links)


def critical_path(events):
    by_id = {event.node_id: event for event in events}
    by_resource = defaultdict(list)
    for event in events:
        for key in event.resource_ids or (event.resource_id,):
            by_resource[key].append(event)
    current = max(events, key=lambda event: event.end_at)
    path = []
    while current is not None:
        path.append(current)
        candidates = [by_id[p] for p in current.predecessor_ids if p in by_id]
        for key in current.resource_ids or (current.resource_id,):
            candidates += by_resource[key]
        candidates = [
            c
            for c in candidates
            if c is not current and c.end_at <= current.start_at + 1e-15
        ]
        current = max(candidates, key=lambda event: event.end_at, default=None)
    return path


def run_case(tp, pp, dtype, algorithm, hop_us):
    hardware = hardware_with_hop(hop_us)
    model = replace(hardware.models.resolve("llama-2-70b"), dtype=dtype)
    layout = Layout(
        a_device="groqchip_v1",
        f_device="groqchip_v1",
        a_tp=tp,
        a_cp=1,
        f_tp=tp,
        a_pp=pp,
        f_pp=pp,
        f_ep=1,
        a_collective_algorithm=algorithm,
        f_collective_algorithm=algorithm,
    )
    runtime = configuration(
        hardware,
        model,
        layout,
        namespace="groqchip-v1",
        dag_template_cache=True,
        colocated_dense_reuse=True,
        pipeline_layer_counts=layer_counts(pp),
        micro_batch=1,
    )
    env = simpy.Environment()
    service = AFService(env, runtime, seed=42, retain_trace=True)
    session = service.admit(
        0, 0, maximum_context=2 * CONTEXT, current_context=CONTEXT - 1
    )
    session.generated = 1
    env.run(
        until=env.process(
            service.forward([session], phase="decode", query_tokens=1, emit_logits=True)
        )
    )
    totals = defaultdict(float)
    for event in critical_path(list(service.trace.events)):
        totals[event.operation or event.event_type] += event.end_at - event.start_at
    step_us = env.now * 1e6
    return dict(
        tp=tp,
        pp=pp,
        chips=tp * pp,
        dtype=dtype,
        algorithm=algorithm,
        hop_us=hop_us,
        step_us=step_us,
        tps=1e6 / step_us,
        ratio_to_published={k: 1e6 / step_us / v for k, v in PUBLISHED_TPS.items()},
        critical_path_us={k: v * 1e6 for k, v in sorted(totals.items())},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    results = []
    for algorithm in ALGORITHMS:
        for hop_us in HOP_LATENCIES_US:
            for tp, pp, dtype in LAYOUTS:
                result = run_case(tp, pp, dtype, algorithm, hop_us)
                results.append(result)
                path = result["critical_path_us"]
                print(
                    f"TP{tp}/PP{pp} {dtype} {algorithm} hop {hop_us}:"
                    f" {result['step_us']:.1f} us, {result['tps']:.0f} tok/s,"
                    f" all-reduce {path.get('all_reduce', 0):.1f} us",
                    flush=True,
                )
    if args.summary:
        write_json(
            args.summary,
            dict(published_tps=PUBLISHED_TPS, context=CONTEXT, cases=results),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
