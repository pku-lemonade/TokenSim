"""Speculative decoding under the published AA LPX test conditions.

The AA footnotes, NVIDIA's articles and press reports do not say whether the
endpoint drafts tokens.  This driver keeps the conditions of
scripts/run_lpx_article_sweep.py (one request, FP8, 256 chips, published link
bandwidth) and prices chain verification of K draft tokens with three drafter
models that differ in drafting cost and in how that cost grows with context.
See docs/lpx-article-speculative.md for the expectations and results.
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
from scripts.lpx.speculative_fit import acceptance_for, committed_tokens
from scripts.run_lpx_article_sweep import ATTENTION_MFU, LAYOUTS, article_config
from scripts.run_lpx_article_sweep import run_point
from scripts.run_lpx_speculative import AA_TPS, CODING_TPS, aa_fit
from scripts.search_lpx_native import write_json

DRAFT_TOKENS = (1, 2, 3, 4, 5, 7)
GLOBAL_LAYERS = 10
# Each draft token costs ``ratio`` of the one-token 10K step plus the attention
# increment above 10K of ``global_layers`` target global layers.  Lookup
# drafting matches n-grams in the prompt and output without a model.  Gemma 4's
# 500M MTP drafter has four layers, one of which attends to the target KV cache.
DRAFTERS = {
    "lookup": dict(ratio=0.0, global_layers=0),
    "mtp-0.05": dict(ratio=0.05, global_layers=1),
    "mtp-0.10": dict(ratio=0.10, global_layers=1),
}
# Published per-position acceptance ranges (docs/lpx-acceptance-evidence.md).
PRIORS = {
    "lookup": dict(prose=(0.13, 0.73), code=(0.36, 0.73)),
    "mtp-0.05": dict(prose=(0.57, 0.68), code=(0.85, 0.90)),
    "mtp-0.10": dict(prose=(0.57, 0.68), code=(0.85, 0.90)),
}


def draft_us(one, drafter, draft_tokens):
    """Drafting time per verification step for each workload."""
    spec = DRAFTERS[drafter]
    fixed = spec["ratio"] * one["aa_10k"]
    return {
        k: draft_tokens
        * (fixed + spec["global_layers"] * (v - one["aa_10k"]) / GLOBAL_LAYERS)
        for k, v in one.items()
    }


def extra_at_prior(steps, draft_tokens, interval):
    """Constant time per step that the AA speeds need at the prior acceptances."""
    extra = {}
    for k, tps in AA_TPS.items():
        needed = [committed_tokens(a, draft_tokens) / tps * 1e6 for a in interval]
        extra[k] = [needed[0] - steps[k], needed[1] - steps[k]]
    return extra


def analysis(results):
    one = {
        (r["parameters"]["layout_name"], r["parameters"]["attention_mfu"]): r["step_us"]
        for r in results
        if r["status"] == "ok" and r["parameters"]["queries"] == 1
    }
    rows = []
    for r in results:
        p = r["parameters"]
        draft_tokens = p["queries"] - 1
        if r["status"] != "ok" or not draft_tokens:
            continue
        base = one[p["layout_name"], p["attention_mfu"]]
        for drafter, priors in PRIORS.items():
            drafting = draft_us(base, drafter, draft_tokens)
            steps = {k: v + drafting[k] for k, v in r["step_us"].items()}
            fit = aa_fit(steps, draft_tokens)
            shared = fit["shared_acceptance"]
            rows.append(
                dict(
                    layout_name=p["layout_name"],
                    attention_mfu=p["attention_mfu"],
                    draft_tokens=draft_tokens,
                    drafter=drafter,
                    verify_us=r["step_us"],
                    draft_us=drafting,
                    step_us=steps,
                    **fit,
                    shared_in_prose_prior=bool(
                        shared
                        and shared[1] >= priors["prose"][0]
                        and shared[0] <= priors["prose"][1]
                    ),
                    coding_acceptance={
                        k: acceptance_for(v * steps["coding_16k"] * 1e-6, draft_tokens)
                        for k, v in CODING_TPS.items()
                    },
                    extra_us_at_prose_prior=extra_at_prior(
                        steps, draft_tokens, priors["prose"]
                    ),
                )
            )
    return rows


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
        args.output / "speculative-points",
        inputs=(ROOT / "data",),
        code_paths=(
            ROOT / "TokenSim",
            ROOT / "scripts/lpx",
            ROOT / "scripts/run_lpx_article_sweep.py",
            Path(__file__),
        ),
    )
    results = [
        run_point(runner, config, name, mfu, queries, "article-speculative")
        for name in args.layouts or LAYOUTS
        for mfu in ATTENTION_MFU
        for queries in (1,) + tuple(k + 1 for k in DRAFT_TOKENS)
    ]
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                assumptions=[
                    "One AA request per forward, without background requests.",
                    "Chain drafting with independent per-position acceptance.",
                    "Drafting runs before verification and adds to the step.",
                    "Lookup drafting always proposes K tokens at no cost.",
                    "An MTP draft token costs a fixed share of the one-token 10K "
                    "step plus one target global layer's attention increment.",
                    "No constant time per step beyond the modeled forward.",
                ],
                drafters=DRAFTERS,
                priors=PRIORS,
                fabric=config["fabric"],
                rows=analysis(results),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
