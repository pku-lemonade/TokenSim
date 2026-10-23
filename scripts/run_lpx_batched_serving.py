"""Batched-serving controls for the AA Gemma 4 31B pair on LPX.

The AA request shares each decode forward with background requests at a fixed
context.  Every point prices one native decode step at the AA request's mean
generation context and reports the AA request's speed as one token per step.
See docs/lpx-batched-serving.md for the hypotheses and recorded results.
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
from scripts.run_lpx_context_parallel import LAYER_COUNTS, environment, layout_for
from scripts.search_lpx_native import write_json

# Input tokens, output tokens and published AA median output TPS.
AA = ((10000, 1500, 3382), (100000, 2000, 3431))
LAYOUTS = {
    "tp16-pp8": dict(kvp=1, pp=8, layer_counts=LAYER_COUNTS),
    "tp16-pp8-kvp4": dict(kvp=4, pp=8, layer_counts=LAYER_COUNTS),
    "tp16-pp16": dict(kvp=1, pp=16, layer_counts=None),
    "tp16-pp16-kvp4": dict(kvp=4, pp=16, layer_counts=None),
}
BACKGROUND_COUNTS = (0, 1, 3, 7, 15)
BACKGROUND_CONTEXTS = (1000, 4000, 16000)
ATTENTION_MFU = (0.1, 0.3, 0.5)


def background_cases(counts, contexts):
    yield 0, None
    for count in (c for c in counts if c):
        for context in contexts:
            yield count, context


def step(runtime, layout, aa_input, aa_output, count, context):
    """Native decode step time with the AA request at its mean context."""
    current = aa_input + aa_output // 2
    contexts = (current,) + (context,) * count
    maxima = (aa_input + aa_output,) + (context,) * count
    profile = capture_profile(
        runtime,
        "batched",
        batch=1 + count,
        queries=1,
        phase="decode",
        emit_logits=True,
        context=current,
        contexts=contexts,
        maximum_contexts=maxima,
        dp_rank=0,
        seed=42,
        sample=0,
        streaming=streaming_settings(layout, launch_us=0),
    )
    meta = profile.metadata
    return meta.get("modeled_duration_s", meta["native_duration_s"])


def run_point(runner, config, name, mfu, count, context):
    spec = LAYOUTS[name]
    config = {**config, "compute": {**config["compute"], "attention_mfu": mfu}}
    parameters = dict(
        layout_name=name,
        layout=asdict(layout_for(spec["kvp"], spec["pp"])),
        layer_counts=spec["layer_counts"],
        attention_mfu=mfu,
        background_count=count,
        background_context=context,
        config=config,
        aa_workloads=[list(w) for w in AA],
        metric="AA request output TPS = 1 / native step time at its mean context",
    )

    def execute(directory):
        hardware, model, _, _ = environment(config)
        layout = layout_for(spec["kvp"], spec["pp"])
        kv = (
            dict(kv_parallel_layers="full_attention", kv_append_block=16)
            if spec["kvp"] > 1
            else {}
        )
        rows = []
        try:
            runtime = configuration(
                hardware,
                model,
                layout,
                namespace="batched",
                dag_template_cache=True,
                colocated_dense_reuse=True,
                pipeline_layer_counts=spec["layer_counts"],
                micro_batch=1 + count,
                **kv,
            )
            for aa_input, aa_output, target in AA:
                seconds = step(runtime, layout, aa_input, aa_output, count, context)
                rows.append(
                    dict(
                        input_tokens=aa_input,
                        step_us=seconds * 1e6,
                        aa_tps=1 / seconds,
                        published_tps=target,
                        error=1 / seconds / target - 1,
                    )
                )
        except ConfigurationError as exc:
            return dict(parameters=parameters, status="infeasible", reason=str(exc))
        return dict(
            parameters=parameters,
            status="ok",
            workloads=rows,
            ratio_100k_10k=rows[1]["aa_tps"] / rows[0]["aa_tps"],
            source_sha256=source_identity(),
        )

    receipt = runner.run(parameters, execute)
    result = receipt.result
    print(
        json.dumps(
            dict(
                layout=name,
                mfu=mfu,
                background=[count, context],
                status=result["status"],
                tps=[round(r["aa_tps"], 1) for r in result.get("workloads", [])],
                reused=receipt.reused,
            )
        ),
        flush=True,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-batched-serving"
    )
    parser.add_argument("--layouts", nargs="+", choices=sorted(LAYOUTS))
    parser.add_argument("--attention-mfu", nargs="+", type=float)
    parser.add_argument("--background-counts", nargs="+", type=int)
    parser.add_argument("--background-contexts", nargs="+", type=int)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = load(args.config)
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    results = [
        run_point(runner, config, name, mfu, count, context)
        for name in args.layouts or LAYOUTS
        for mfu in args.attention_mfu or ATTENTION_MFU
        for count, context in background_cases(
            args.background_counts or BACKGROUND_COUNTS,
            args.background_contexts or BACKGROUND_CONTEXTS,
        )
    ]
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                points=[
                    {k: v for k, v in r.items() if k != "parameters"}
                    | {
                        k: r["parameters"][k]
                        for k in (
                            "layout_name",
                            "attention_mfu",
                            "background_count",
                            "background_context",
                        )
                    }
                    for r in results
                ],
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
