"""Verification steps on the closest AA setup under alternative LP30 unit rates.

Captures native batch-one steps of 1 to 8 query rows on TP32/PP8 with DCP 8,
under the published conditions of scripts/run_lpx_article_sweep.py, for the
study's LP30 attention costs and three alternatives.  See
docs/lpx-attention-sensitivity.md for the expectations and results.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.experiments import ExperimentRunner
from scripts.lpx.fidelity import load
from scripts.lpx.replay_profiles import source_identity
from scripts.run_lpx_article_sweep import article_config, run_point
from scripts.search_lpx_native import write_json

LAYOUT = "tp32-pp8-kvp8"
ROWS = tuple(range(1, 9))
# Attention matrix efficiency and LP30 pipeline overrides of each variant.
VARIANTS = {
    "baseline": dict(attention_mfu=0.5, operator={}),
    "matrix": dict(attention_mfu=1.0, operator={}),
    "exponential": dict(attention_mfu=0.5, operator=dict(exp_ops_per_s=4e12)),
    "fast-units": dict(
        attention_mfu=1.0,
        operator=dict(
            exp_ops_per_s=4e12, vector_efficiency=1.0, shuffle_efficiency=1.0
        ),
    ),
}


def variant_config(config, variant):
    spec = VARIANTS[variant]
    return {**config, "operator": {**config["operator"], **spec["operator"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-attention-sensitivity"
    )
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = article_config(load(args.config))
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(
            ROOT / "TokenSim",
            ROOT / "scripts/lpx",
            ROOT / "scripts/run_lpx_article_sweep.py",
            Path(__file__),
        ),
    )
    rows = []
    for variant in args.variants or VARIANTS:
        spec = VARIANTS[variant]
        for queries in ROWS:
            result = run_point(
                runner,
                variant_config(config, variant),
                LAYOUT,
                spec["attention_mfu"],
                queries,
                "attention-sensitivity",
            )
            rows.append(
                dict(
                    variant=variant,
                    attention_mfu=spec["attention_mfu"],
                    operator=spec["operator"],
                    rows=queries,
                    status=result["status"],
                    step_us=result.get("step_us"),
                )
            )
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                layout=LAYOUT,
                variants=VARIANTS,
                assumptions=[
                    "One request per forward with q verified rows, no background.",
                    "Published-conditions fabric and FP8 weights and KV cache.",
                    "Variant rates apply to every operator on the affected unit.",
                ],
                rows=rows,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
