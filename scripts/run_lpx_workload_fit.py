"""Per-workload acceptance fits for the AA pair and the coding values on LPX.

Each point captures native verification forwards for the AA 10K, AA 100K and
16K coding contexts under the same background traffic, so one serving setup
prices all four published speeds.  Each workload receives its own chain
acceptance.  See docs/lpx-workload-acceptance.md for the hypotheses and
recorded results.
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
from scripts.lpx.replay_profiles import source_identity
from scripts.lpx.speculative_fit import acceptance_for
from scripts.run_lpx_attention_group import LAYOUTS as TP32_LAYOUTS
from scripts.run_lpx_batched_serving import LAYOUTS as TP16_LAYOUTS
from scripts.run_lpx_context_parallel import environment, layout_for
from scripts.run_lpx_speculative import (
    AA_CONTEXTS,
    BACKGROUND_CONTEXTS,
    CODING_CONTEXT,
    MAXIMUM_ROWS,
    flatten,
    step,
)
from scripts.search_lpx_native import write_json

LAYOUTS = {
    name: dict(tp=16, **spec)
    for name, spec in TP16_LAYOUTS.items()
    if name in ("tp16-pp8", "tp16-pp8-kvp4", "tp16-pp16-kvp4")
} | {"tp32-pp4-kvp8": dict(layer_counts=None, **TP32_LAYOUTS["tp32-pp4-kvp8"])}
# Layout and attention matrix efficiency pairs.
CASES = (
    ("tp16-pp8-kvp4", 0.5),
    ("tp16-pp8-kvp4", 0.1),
    ("tp16-pp16-kvp4", 0.5),
    ("tp16-pp8", 0.5),
    ("tp32-pp4-kvp8", 0.5),
)
QUERIES = (2, 4, 8)
BACKGROUND_COUNTS = (0, 1, 3, 7)
# Published speeds: AA medians and the Hot Chips coding values.
TARGETS = dict(
    aa_10k=("aa_10k", 3382),
    aa_100k=("aa_100k", 3431),
    coding_median=("coding_16k", 4767),
    coding_example=("coding_16k", 10996),
)


def run_point(runner, config, name, mfu, queries, count, background):
    spec = LAYOUTS[name]
    config = {**config, "compute": {**config["compute"], "attention_mfu": mfu}}
    workloads = AA_CONTEXTS + (CODING_CONTEXT,)
    layout = layout_for(spec["kvp"], spec["pp"], spec["tp"])
    parameters = dict(
        layout_name=name,
        layout=asdict(layout),
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
                namespace="workload-fit",
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


def cases():
    for name, mfu in CASES:
        for queries in QUERIES:
            for count in BACKGROUND_COUNTS:
                if (1 + count) * queries > MAXIMUM_ROWS:
                    continue
                for background in BACKGROUND_CONTEXTS if count else (None,):
                    yield name, mfu, queries, count, background


def fit(point):
    """Exact acceptance of every published speed, or None when unreachable."""
    draft_tokens = point["queries"] - 1
    steps = point["step_us"]
    acceptance = {
        target: acceptance_for(tps * steps[workload] * 1e-6, draft_tokens)
        for target, (workload, tps) in TARGETS.items()
    }
    return dict(
        layout_name=point["layout_name"],
        attention_mfu=point["attention_mfu"],
        draft_tokens=draft_tokens,
        background_count=point["background_count"],
        background_context=point["background_context"],
        step_us=steps,
        committed_tokens={
            target: tps * steps[workload] * 1e-6
            for target, (workload, tps) in TARGETS.items()
        },
        acceptance=acceptance,
        all_reachable=all(value is not None for value in acceptance.values()),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-workload-fit"
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = load(args.config)
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    points = [flatten(run_point(runner, config, *case)) for case in cases()]
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                assumptions=[
                    "Chain drafting with independent per-position acceptance.",
                    "Each published speed receives its own acceptance.",
                    "No draft cost; background requests verify the same tokens.",
                    "The coding request runs under the same background traffic.",
                ],
                points=points,
                fits=[fit(p) for p in points if p["status"] == "ok"],
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
