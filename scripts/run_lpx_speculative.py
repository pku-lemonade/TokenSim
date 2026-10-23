"""Speculative-decoding controls for the AA Gemma 4 31B pair on LPX.

Each point captures native verification forwards in which the AA request and
any background requests check ``q = K + 1`` query tokens.  The summary applies
chain acceptance arithmetic to the captured step times.  See
docs/lpx-speculative-decoding.md for the hypotheses and recorded results.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from scripts.lpx.fidelity import load
from scripts.lpx.native_config import configuration
from scripts.lpx.native_search import streaming_settings
from scripts.lpx.replay_profiles import capture_profile, source_identity
from scripts.lpx.speculative_fit import acceptance_for, shared_match
from scripts.run_lpx_batched_serving import LAYOUTS
from scripts.run_lpx_context_parallel import environment, layout_for
from scripts.search_lpx_native import write_json

# Workload name, mean decode context and resident KV reserve.
AA_CONTEXTS = (("aa_10k", 10750, 11500), ("aa_100k", 101000, 102000))
CODING_CONTEXT = ("coding_16k", 17000, 18000)
# Published speeds and the 5% bands of the AA pair.
AA_TPS = dict(aa_10k=3382, aa_100k=3431)
CODING_TPS = dict(median=4767, example=10996)
BAND = 0.05
SERIAL_QUERIES = (1, 2, 3, 4, 5, 6, 8)
SERIAL_MFU = (0.1, 0.5, 1.0)
DRAFT_COST_RATIOS = (0.0, 0.05, 0.1, 0.2)
BATCHED_LAYOUTS = ("tp16-pp8", "tp16-pp8-kvp4", "tp16-pp16-kvp4")
BATCHED_QUERIES = (2, 4, 8)
BATCHED_MFU = 0.5
BACKGROUND_COUNTS = (1, 3, 7)
# The study's live scratch and the host-side streaming DAG hold 16 query rows.
MAXIMUM_ROWS = 16
BACKGROUND_CONTEXTS = (1000, 4000, 16000)


def step(runtime, layout, queries, context, reserve, count, background):
    """Native verification step with the AA request at ``context``."""
    profile = capture_profile(
        runtime,
        "speculative",
        batch=1 + count,
        queries=queries,
        phase="decode",
        emit_logits=True,
        context=context,
        contexts=(context,) + (background,) * count,
        maximum_contexts=(reserve,) + (background,) * count,
        dp_rank=0,
        seed=42,
        sample=0,
        streaming=streaming_settings(layout, launch_us=0),
    )
    meta = profile.metadata
    return meta.get("modeled_duration_s", meta["native_duration_s"])


def run_point(runner, config, name, mfu, queries, count, background):
    spec = LAYOUTS[name]
    config = {**config, "compute": {**config["compute"], "attention_mfu": mfu}}
    workloads = AA_CONTEXTS + (() if count else (CODING_CONTEXT,))
    parameters = dict(
        layout_name=name,
        layout=asdict(layout_for(spec["kvp"], spec["pp"])),
        layer_counts=spec["layer_counts"],
        attention_mfu=mfu,
        queries=queries,
        background_count=count,
        background_context=background,
        workloads=[list(w) for w in workloads],
        config=config,
        metric="native verification step time for q query tokens per request",
    )

    def execute(directory):
        hardware, model, _, _ = environment(config)
        layout = layout_for(spec["kvp"], spec["pp"])
        kv = (
            dict(kv_parallel_layers="full_attention", kv_append_block=16)
            if spec["kvp"] > 1
            else {}
        )
        steps = {}
        try:
            runtime = configuration(
                hardware,
                model,
                layout,
                namespace="speculative",
                dag_template_cache=True,
                colocated_dense_reuse=True,
                pipeline_layer_counts=spec["layer_counts"],
                micro_batch=(1 + count) * queries,
                **kv,
            )
            for workload, context, reserve in workloads:
                seconds = step(
                    runtime, layout, queries, context, reserve, count, background
                )
                steps[workload] = seconds * 1e6
        except ConfigurationError as exc:
            return dict(parameters=parameters, status="infeasible", reason=str(exc))
        return dict(
            parameters=parameters,
            status="ok",
            step_us=steps,
            source_sha256=source_identity(),
        )

    receipt = runner.run(parameters, execute)
    result = receipt.result
    print(
        json.dumps(
            dict(
                layout=name,
                mfu=mfu,
                queries=queries,
                background=[count, background],
                status=result["status"],
                step_us={k: round(v, 1) for k, v in result.get("step_us", {}).items()},
                reused=receipt.reused,
            )
        ),
        flush=True,
    )
    return result


def bands(targets):
    return {k: (v * (1 - BAND), v * (1 + BAND)) for k, v in targets.items()}


def aa_fit(steps_us, draft_tokens):
    """Shared-acceptance AA match and the separate acceptance of each workload."""
    aa = bands(AA_TPS)
    match = shared_match(
        [steps_us[k] * 1e-6 for k in AA_TPS], [aa[k] for k in AA_TPS], draft_tokens
    )
    return dict(
        time_ratio_100k_10k=steps_us["aa_100k"] / steps_us["aa_10k"],
        shared_acceptance=list(match) if match else None,
        exact_acceptance={
            k: acceptance_for(AA_TPS[k] * steps_us[k] * 1e-6, draft_tokens)
            for k in AA_TPS
        },
    )


def serial_analysis(points):
    """Batch-one verification with a draft cost proportional to the serial step."""
    base = {
        (p["layout_name"], p["attention_mfu"]): p["step_us"]
        for p in points
        if p["status"] == "ok" and p["queries"] == 1
    }
    rows = []
    for p in points:
        if p["status"] != "ok" or p["background_count"]:
            continue
        draft_tokens = p["queries"] - 1
        one = base[p["layout_name"], p["attention_mfu"]]
        for ratio in DRAFT_COST_RATIOS if draft_tokens else (0.0,):
            steps = {
                k: v + draft_tokens * ratio * one[k] for k, v in p["step_us"].items()
            }
            rows.append(
                dict(
                    layout_name=p["layout_name"],
                    attention_mfu=p["attention_mfu"],
                    draft_tokens=draft_tokens,
                    draft_cost_ratio=ratio,
                    step_us=steps,
                    **aa_fit(steps, draft_tokens),
                    coding_acceptance={
                        k: acceptance_for(v * steps["coding_16k"] * 1e-6, draft_tokens)
                        for k, v in CODING_TPS.items()
                    },
                )
            )
    return rows


def batched_analysis(points):
    """Verification shared with background requests and no draft cost."""
    return [
        dict(
            layout_name=p["layout_name"],
            attention_mfu=p["attention_mfu"],
            draft_tokens=p["queries"] - 1,
            background_count=p["background_count"],
            background_context=p["background_context"],
            step_us=p["step_us"],
            **aa_fit(p["step_us"], p["queries"] - 1),
        )
        for p in points
        if p["status"] == "ok" and p["background_count"]
    ]


def flatten(result):
    keep = (
        "layout_name",
        "attention_mfu",
        "queries",
        "background_count",
        "background_context",
    )
    return {k: v for k, v in result.items() if k != "parameters"} | {
        k: result["parameters"][k] for k in keep
    }


def cases(args):
    if not args.skip_serial:
        for name in args.layouts or LAYOUTS:
            for mfu in SERIAL_MFU:
                for queries in SERIAL_QUERIES:
                    yield name, mfu, queries, 0, None
    if not args.skip_batched:
        for name in args.layouts or BATCHED_LAYOUTS:
            for queries in BATCHED_QUERIES:
                for count in BACKGROUND_COUNTS:
                    if (1 + count) * queries > MAXIMUM_ROWS:
                        continue
                    for background in BACKGROUND_CONTEXTS:
                        yield name, BATCHED_MFU, queries, count, background


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-speculative")
    parser.add_argument("--layouts", nargs="+", choices=sorted(LAYOUTS))
    parser.add_argument("--skip-serial", action="store_true")
    parser.add_argument("--skip-batched", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = load(args.config)
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    points = [flatten(run_point(runner, config, *case)) for case in cases(args)]
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                assumptions=[
                    "Chain drafting with independent per-position acceptance.",
                    "Draft cost per step is K times a ratio of the serial step.",
                    "Batched points carry no draft cost.",
                    "Background requests verify the same number of tokens.",
                    "Rejected positions need no extra KV rollback time.",
                ],
                points=points,
                serial=serial_analysis(points),
                batched=batched_analysis(points),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
