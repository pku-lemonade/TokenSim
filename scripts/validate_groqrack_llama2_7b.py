"""Simulate batch-one Llama-2-7B decode on a 72-card GroqRack and compare.

Usami, Vishwanath and Bethel (arXiv 2606.17104, HPAI4S'26) measure a time per
output token of 2.98 to 3.05 ms for FP16 Llama-2-7B on a GroqRack of 72
first-generation GroqCards (nine nodes of eight), with one request, one token
per invocation and sampling on the host CPU followed by an MPI broadcast.
This script prices the device-side decode step for plausible layouts with the
method of scripts/validate_groqchip_llama2.py.  See
docs/groqrack-llama2-7b-validation.md for the recorded results.
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
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration
from scripts.search_lpx_native import write_json
from scripts.validate_groqchip_llama2 import critical_path, hardware_with_hop

MEASURED_TPOT_MS = (2.98, 3.05)
# Mean and final decode context for 100 and 1,600 input tokens with equal output.
CONTEXTS = ((150, 200), (2400, 3200))
# FP16 layers of 405 MB fit at most four per 230 MB card at TP8 and two at
# TP4; the first stage also holds the embedding and the last the output head.
LAYOUTS = {
    "tp8-pp9": (8, 9, [3, 4, 4, 4, 4, 4, 3, 3, 3]),
    "tp4-pp18": (4, 18, [1] + [2] * 14 + [1, 1, 1]),
}
ALGORITHMS = ("ring", "direct")
HOP_LATENCIES_US = (0.5, 0.3)


def run_case(name, algorithm, hop_us, context, maximum):
    tp, pp, counts = LAYOUTS[name]
    assert sum(counts) == 32 and len(counts) == pp
    hardware = hardware_with_hop(hop_us)
    model = replace(hardware.models.resolve("llama-2-7b"), dtype="fp16")
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
        namespace="groqrack-7b",
        dag_template_cache=True,
        colocated_dense_reuse=True,
        pipeline_layer_counts=counts,
        micro_batch=1,
    )
    env = simpy.Environment()
    service = AFService(env, runtime, seed=42, retain_trace=True)
    session = service.admit(0, 0, maximum_context=maximum, current_context=context - 1)
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
        layout=name,
        tp=tp,
        pp=pp,
        algorithm=algorithm,
        hop_us=hop_us,
        context=context,
        step_us=step_us,
        missing_ms=[m - step_us / 1e3 for m in MEASURED_TPOT_MS],
        critical_path_us={k: v * 1e6 for k, v in sorted(totals.items())},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    results = []
    for algorithm in ALGORITHMS:
        for hop_us in HOP_LATENCIES_US:
            for name in LAYOUTS:
                for context, maximum in CONTEXTS:
                    result = run_case(name, algorithm, hop_us, context, maximum)
                    results.append(result)
                    print(
                        f"{name} {algorithm} hop {hop_us} context {context}:"
                        f" {result['step_us']:.1f} us, all-reduce"
                        f" {result['critical_path_us'].get('all_reduce', 0):.1f} us",
                        flush=True,
                    )
    if args.summary:
        write_json(
            args.summary,
            dict(measured_tpot_ms=MEASURED_TPOT_MS, contexts=CONTEXTS, cases=results),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
