"""Attention-group size controls for the AA Gemma 4 31B pair on LPX.

TP32 doubles the number of chips that share one global attention layer.  Each
point captures a native decode or verification forward with the AA request and
any background requests, as in the batched-serving and speculative drivers.
See docs/lpx-attention-group.md for the hypotheses and recorded results.
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
from scripts.run_lpx_context_parallel import environment, layout_for
from scripts.run_lpx_speculative import (
    AA_CONTEXTS,
    AA_TPS,
    CODING_CONTEXT,
    MAXIMUM_ROWS,
    aa_fit,
    batched_analysis,
    flatten,
    serial_analysis,
    step,
)
from scripts.search_lpx_native import write_json

# Pipeline layer counts come from the capacity partition of each stage count.
LAYOUTS = {
    "tp32-pp4": dict(tp=32, kvp=1, pp=4),
    "tp32-pp4-kvp2": dict(tp=32, kvp=2, pp=4),
    "tp32-pp4-kvp4": dict(tp=32, kvp=4, pp=4),
    "tp32-pp4-kvp8": dict(tp=32, kvp=8, pp=4),
    "tp32-pp8-kvp8": dict(tp=32, kvp=8, pp=8),
}
ATTENTION_MFU = (0.1, 0.3, 0.5)
BACKGROUND_COUNTS = (0, 1, 2, 3, 7)
BACKGROUND_CONTEXTS = (1000, 4000, 16000)
SPECULATIVE_LAYOUTS = ("tp32-pp4-kvp8",)
SPECULATIVE_MFU = (0.1, 0.5)
SPECULATIVE_QUERIES = (2, 4, 8)
SPECULATIVE_COUNTS = (0, 1, 3, 7)


def run_point(runner, config, name, mfu, queries, count, background):
    spec = LAYOUTS[name]
    config = {**config, "compute": {**config["compute"], "attention_mfu": mfu}}
    workloads = AA_CONTEXTS + (() if count else (CODING_CONTEXT,))
    layout = layout_for(spec["kvp"], spec["pp"], spec["tp"])
    parameters = dict(
        layout_name=name,
        layout=asdict(layout),
        layer_counts=None,
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
                namespace="attention-group",
                dag_template_cache=True,
                colocated_dense_reuse=True,
                pipeline_layer_counts=None,
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


def background_cases(counts, contexts):
    for count in counts:
        for context in contexts if count else (None,):
            yield count, context


def cases(args):
    if not args.skip_single:
        for name in args.layouts or LAYOUTS:
            for mfu in ATTENTION_MFU:
                for count, context in background_cases(
                    BACKGROUND_COUNTS, BACKGROUND_CONTEXTS
                ):
                    yield name, mfu, 1, count, context
    if not args.skip_speculative:
        for name in args.layouts or SPECULATIVE_LAYOUTS:
            for mfu in SPECULATIVE_MFU:
                for queries in SPECULATIVE_QUERIES:
                    for count, context in background_cases(
                        SPECULATIVE_COUNTS, BACKGROUND_CONTEXTS
                    ):
                        if (1 + count) * queries <= MAXIMUM_ROWS:
                            yield name, mfu, queries, count, context


def single_analysis(points):
    """One token per forward: each AA speed is the reciprocal of its step."""
    rows = []
    for p in points:
        if p["status"] != "ok" or p["queries"] != 1:
            continue
        steps = p["step_us"]
        tps = {k: 1e6 / steps[k] for k in AA_TPS}
        rows.append(
            dict(
                layout_name=p["layout_name"],
                attention_mfu=p["attention_mfu"],
                background_count=p["background_count"],
                background_context=p["background_context"],
                step_us=steps,
                increment_us=steps["aa_100k"] - steps["aa_10k"],
                aa_tps=tps,
                error={k: tps[k] / AA_TPS[k] - 1 for k in AA_TPS},
                match=aa_fit(steps, 0)["shared_acceptance"] is not None,
            )
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-attention-group"
    )
    parser.add_argument("--layouts", nargs="+", choices=sorted(LAYOUTS))
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-speculative", action="store_true")
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
        speculative = [
            p for p in points if p["queries"] > 1 or not p["background_count"]
        ]
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                assumptions=[
                    "Pipeline layer counts follow the capacity partition.",
                    "Chain drafting with independent per-position acceptance.",
                    "Draft cost per step is K times a ratio of the serial step.",
                    "Batched verification points carry no draft cost.",
                    "Background requests verify the same number of tokens.",
                ],
                points=points,
                single=single_analysis(points),
                serial=serial_analysis(
                    [p for p in speculative if p["layout_name"] in SPECULATIVE_LAYOUTS]
                ),
                batched=batched_analysis([p for p in points if p["queries"] > 1]),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
