"""Batch-one AA Gemma 4 31B sweep under the published LPX test conditions.

The AA footnotes establish one request at a time on one named deployment, and
press reports give FP8 weights on LP30 chips only.  This sweep keeps every
point at batch one without background requests or speculative decoding, uses
one 256-chip rack, prices C2C injection at the published 96 links of 112 Gb/s,
and reports the constant per-step time that would place both AA speeds within
5%.  See docs/lpx-article-sweep.md for the expectations and results.
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
from scripts.run_lpx_speculative import AA_CONTEXTS, AA_TPS, BAND, CODING_CONTEXT
from scripts.run_lpx_speculative import step
from scripts.search_lpx_native import write_json

# 96 C2C links of 112 Gb/s (14 GB/s) per direction, from NVIDIA's articles.
LINKS, LINK_BYTES_PER_S = 96, 14e9
CHIPS = 256
# Pipeline layer counts come from the capacity partition of each stage count.
LAYOUTS = {
    "tp32-pp8": dict(tp=32, kvp=1, pp=8),
    "tp32-pp8-kvp2": dict(tp=32, kvp=2, pp=8),
    "tp32-pp8-kvp4": dict(tp=32, kvp=4, pp=8),
    "tp32-pp8-kvp8": dict(tp=32, kvp=8, pp=8),
    "tp16-pp16": dict(tp=16, kvp=1, pp=16),
    "tp16-pp16-kvp2": dict(tp=16, kvp=2, pp=16),
    "tp16-pp16-kvp4": dict(tp=16, kvp=4, pp=16),
}
ATTENTION_MFU = (0.1, 0.3, 0.5)


def article_config(config):
    fabric = dict(
        config["fabric"],
        ports_per_chip=LINKS,
        lane_bytes_per_s=LINK_BYTES_PER_S,
        injection_bytes_per_s=LINKS * LINK_BYTES_PER_S,
    )
    return {**config, "fabric": fabric}


def run_point(runner, config, name, mfu):
    spec = LAYOUTS[name]
    assert spec["tp"] * spec["pp"] == CHIPS
    config = {**config, "compute": {**config["compute"], "attention_mfu": mfu}}
    workloads = AA_CONTEXTS + (CODING_CONTEXT,)
    layout = layout_for(spec["kvp"], spec["pp"], spec["tp"])
    parameters = dict(
        layout_name=name,
        layout=asdict(layout),
        layer_counts=None,
        attention_mfu=mfu,
        workloads=[list(w) for w in workloads],
        config=config,
        metric="native batch-one decode step time",
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
                namespace="article-sweep",
                dag_template_cache=True,
                colocated_dense_reuse=True,
                pipeline_layer_counts=None,
                micro_batch=1,
                **kv,
            )
            for workload, context, reserve in workloads:
                steps[workload] = step(runtime, layout, 1, context, reserve, 0, 0) * 1e6
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
                status=result["status"],
                step_us={k: round(v, 1) for k, v in result.get("step_us", {}).items()},
                reused=receipt.reused,
            )
        ),
        flush=True,
    )
    return result


def analysis(result):
    """AA errors and the constant per-step time window that matches both values."""
    parameters = result["parameters"]
    row = dict(
        layout_name=parameters["layout_name"],
        attention_mfu=parameters["attention_mfu"],
        status=result["status"],
    )
    if result["status"] != "ok":
        return row | dict(reason=result["reason"])
    steps = result["step_us"]
    shortest = {k: 1e6 / (v * (1 + BAND)) for k, v in AA_TPS.items()}
    longest = {k: 1e6 / (v * (1 - BAND)) for k, v in AA_TPS.items()}
    low = max(0.0, *(shortest[k] - steps[k] for k in AA_TPS))
    high = min(longest[k] - steps[k] for k in AA_TPS)
    tps = {k: 1e6 / steps[k] for k in AA_TPS}
    return row | dict(
        step_us=steps,
        increment_us=steps["aa_100k"] - steps["aa_10k"],
        aa_tps=tps,
        error={k: tps[k] / AA_TPS[k] - 1 for k in AA_TPS},
        match_without_overhead=low == 0.0 and high >= 0.0,
        overhead_window_us=[low, high] if low <= high else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-article")
    parser.add_argument("--layouts", nargs="+", choices=sorted(LAYOUTS))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = article_config(load(args.config))
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    results = [
        run_point(runner, config, name, mfu)
        for name in args.layouts or LAYOUTS
        for mfu in ATTENTION_MFU
    ]
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                assumptions=[
                    "One AA request per forward, without background requests.",
                    "No speculative decoding.",
                    "FP8 weights and FP8 KV cache converted to BF16 for attention.",
                    "Pipeline layer counts follow the capacity partition.",
                    "C2C injection is 96 links of 14 GB/s per direction.",
                ],
                fabric=config["fabric"],
                rows=[analysis(r) for r in results],
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
