"""Per-step cost audit of the closest batch-one AA setup on 256 LP30 chips.

Rebuilds TP32/PP8 with DCP 8 at attention efficiency 0.50 under the
article-condition sweep's configuration, lists the operations of one decode
step and its critical path, and measures how the step grows when the
collective launch, the per-operator instruction floor and the per-step
invocation charge, which the sweep sets to zero, take nonzero values.  The
selected token's return from the last pipeline stage to the first is not
modeled, so the audit reports the pipeline handoffs as a size reference.  See
docs/lpx-step-cost-audit.md.
"""

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import simpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.af.serving.service import AFService
from TokenSim.hardware import HardwareContext
from scripts.analyze_lpx_api_outputs import window
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.fidelity import load, with_fidelity
from scripts.lpx.native_config import configuration
from scripts.lpx.native_search import streaming_settings
from scripts.lpx.replay_profiles import capture_profile
from scripts.run_lpx_article_sweep import LAYOUTS, article_config
from scripts.run_lpx_context_parallel import layout_for
from scripts.run_lpx_speculative import AA_CONTEXTS, AA_TPS
from scripts.search_lpx_native import write_json

LAYOUT = "tp32-pp8-kvp8"
EFFICIENCY = 0.5
# Gemma tokens per o200k_base token on AA-like book answers (lpx-api-outputs).
TOKEN_RATIO = 1.006
POINTS = [
    dict(name="sweep", launch_us=0.0, floor_us=0.0, invocation_us=0.0),
    *(
        dict(name=f"launch {v}", launch_us=v, floor_us=0.0, invocation_us=0.0)
        for v in (0.1, 0.25, 0.5)
    ),
    *(
        dict(name=f"floor {v}", launch_us=0.0, floor_us=v, invocation_us=0.0)
        for v in (0.025, 0.05)
    ),
    dict(name="invocation 1", launch_us=0.0, floor_us=0.0, invocation_us=1.0),
    dict(name="robustness", launch_us=0.0, floor_us=0.05, invocation_us=1.0),
    dict(name="robustness + launch", launch_us=0.5, floor_us=0.05, invocation_us=1.0),
]


def build(config, floor_us):
    execution = ExecutionAssumptions(
        mode="compiled", instruction_floor_us=floor_us, c2c_hop_us=0.35
    )
    hardware = with_fidelity(
        with_execution(HardwareContext.load(ROOT / "data"), execution), config
    )
    model = replace(
        hardware.models.resolve("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
    )
    spec = LAYOUTS[LAYOUT]
    layout = layout_for(spec["kvp"], spec["pp"], spec["tp"])
    runtime = configuration(
        hardware,
        model,
        layout,
        namespace="step-audit",
        dag_template_cache=True,
        colocated_dense_reuse=True,
        pipeline_layer_counts=None,
        micro_batch=1,
        kv_parallel_layers="full_attention",
        kv_append_block=16,
    )
    return runtime, layout


def step_us(runtime, layout, point, context, reserve):
    profile = capture_profile(
        runtime,
        "step-audit",
        batch=1,
        queries=1,
        phase="decode",
        emit_logits=True,
        context=context,
        contexts=(context,),
        maximum_contexts=(reserve,),
        dp_rank=0,
        seed=42,
        sample=0,
        streaming=streaming_settings(layout, launch_us=point["launch_us"]),
        invocation_us=point["invocation_us"],
    )
    return profile.metadata["modeled_duration_s"] * 1e6


def inventory(runtime, context, reserve):
    """Operation counts of one native decode step and of its critical path."""
    env = simpy.Environment()
    service = AFService(env, runtime, seed=42, retain_trace=True)
    session = service.admit(0, 0, maximum_context=reserve, current_context=context - 1)
    session.generated = 1
    done = env.process(
        service.forward([session], phase="decode", query_tokens=1, emit_logits=True)
    )
    env.run(until=done)
    events = service.trace.events
    by_id = {event.node_id: event for event in events}
    node, chain = max(events, key=lambda event: event.end_at), []
    while True:
        chain.append(node)
        if not node.predecessor_ids:
            break
        node = max((by_id[p] for p in node.predecessor_ids), key=lambda e: e.end_at)
    gaps = Counter()
    for event in chain:
        parent_end = max((by_id[p].end_at for p in event.predecessor_ids), default=0)
        gaps[event.operation] += (event.ready_at - parent_end) * 1e6
    handoffs = [
        event.service_duration * 1e6
        for event in reversed(chain)
        if event.operation == "af_pipeline_handoff"
    ]
    result = dict(
        events=len(events),
        operations=dict(Counter(event.operation for event in events)),
        path_events=len(chain),
        path_operations=dict(Counter(event.operation for event in chain)),
        path_readiness_gap_us={k: v for k, v in gaps.items() if v > 1e-9},
        first_path_operation=chain[-1].operation,
        last_path_operation=chain[0].operation,
        handoff_service_us=handoffs,
    )
    service.trace.clear()
    return result


def stage_chips(runtime):
    """Global ranks of each pipeline stage."""
    stages = {}
    for worker in runtime.workers:
        info = worker.rank_info
        stages.setdefault(info.pp_rank, []).append(info.global_rank)
    return {k: [min(v), max(v)] for k, v in sorted(stages.items())}


def priced(steps):
    tps = {k: 1e6 / (steps[k] * TOKEN_RATIO) for k in AA_TPS}
    return dict(
        aa_tps=tps,
        error={k: tps[k] / v - 1 for k, v in AA_TPS.items()},
        extra_us=window(steps, {k: 1.0 for k in AA_TPS}, TOKEN_RATIO),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = article_config(load(args.config))
    config = {**config, "compute": {**config["compute"], "attention_mfu": EFFICIENCY}}
    runtimes = {}
    rows = []
    for point in POINTS:
        if point["floor_us"] not in runtimes:
            runtimes[point["floor_us"]] = build(config, point["floor_us"])
        runtime, layout = runtimes[point["floor_us"]]
        steps = {
            name: step_us(runtime, layout, point, context, reserve)
            for name, context, reserve in AA_CONTEXTS
        }
        row = dict(**point, step_us=steps, **priced(steps))
        if rows:
            row["growth_us"] = {k: steps[k] - rows[0]["step_us"][k] for k in steps}
        rows.append(row)
        print(
            f"{point['name']:20s} step {steps['aa_10k']:7.2f} / {steps['aa_100k']:7.2f} µs,"
            f" error {row['error']['aa_10k']:+.1%} / {row['error']['aa_100k']:+.1%},"
            f" extra {row['extra_us']}",
            flush=True,
        )
    runtime, _ = runtimes[0.0]
    _, context, reserve = AA_CONTEXTS[0]
    audit = inventory(runtime, context, reserve)
    print(
        f"inventory: {audit['events']} events, {audit['path_events']} on the path,"
        f" handoffs {[round(v, 3) for v in audit['handoff_service_us']]} µs"
    )
    if args.summary:
        write_json(
            args.summary,
            dict(
                layout=LAYOUT,
                attention_mfu=EFFICIENCY,
                token_ratio=TOKEN_RATIO,
                fabric=config["fabric"],
                stage_ranks=stage_chips(runtime),
                inventory=audit,
                points=rows,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
