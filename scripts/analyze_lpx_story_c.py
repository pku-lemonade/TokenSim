"""Check one lookup drafter at K = 7 on the AA book answers and the coding spread.

Story C of the AA reproduction uses one lookup drafter at ``K = 7`` on every
run with at most the extra time per step that its coding window allows.  This
script prices each AA-like book answer at that setting and tests whether the
books that copy least fit both AA values, and draws sets of 80 H20 coding turns
to place NVIDIA's P80/median and maximum/median among them.  See
docs/lpx-story-c-checks.md.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_lpx_coding_spread import context_model, lookup_speeds, ratios
from scripts.analyze_lpx_published_conditions import (
    AA_TPS,
    BAND,
    load,
    verification_steps,
)
from scripts.measure_lpx_lookup_acceptance import equivalent_acceptance
from scripts.search_lpx_native import write_json

CONTEXTS = dict(aa_10k="10k", aa_100k="100k")
STYLES = ("summary", "quotes")
DATA_SETS = dict(
    full=("api-lookup-results.json", "api-aa-analysis.json"),
    answer_alone=(
        "api-lookup-answer-alone-results.json",
        "api-aa-answer-alone-analysis.json",
    ),
)
DRAFT_TOKENS = 7
MINIMUM_NGRAMS = (1, 2)
SUBSET_SIZES = (4, 8, 16)
CODING = "speed-coding-plain"
SPREAD_SETTINGS = ((1, 7, 7.0), (2, 7, 6.0), (1, 3, 13.0), (1, 3, 21.0))
NVIDIA_RATIOS = dict(p80=5520 / 4767, maximum=6981 / 4767)
SAMPLE_SIZE, DRAWS, SEED = 80, 2000, 20261101


def aa_speed(row, verify, context, extra, ratio):
    """AA tokens per second of one replayed answer at a fixed AA context."""
    tokens, counts = row
    time = sum(c * (verify[n + 1][context] + extra) for n, c in enumerate(counts))
    return tokens / time * 1e6 / ratio


def in_band(speeds):
    return all(abs(speeds[k] / AA_TPS[k] - 1) <= BAND for k in AA_TPS)


def extra_grid(upper):
    return [x / 2 for x in range(int(upper * 2) + 1)]


def shared_window(speed_at, upper, lower=0.0):
    """Extra times in [lower, upper] at which both AA speeds lie in the band."""
    fits = [e for e in extra_grid(upper) if e >= lower and in_band(speed_at(e))]
    return [min(fits), max(fits)] if fits else None


def tau(row):
    tokens, counts = row
    return tokens / sum(counts)


def books(per_context, verify, ratio, upper, lower=0.0, draft_tokens=DRAFT_TOKENS):
    """Per-book windows, ordered from the book that copies least."""
    results = []
    for i, (low, high) in enumerate(zip(per_context["aa_10k"], per_context["aa_100k"])):
        pair = dict(aa_10k=low, aa_100k=high)
        results.append(
            dict(
                book=i,
                copy_share=statistics.mean(
                    equivalent_acceptance(tau(r), draft_tokens) for r in pair.values()
                ),
                tau={k: tau(r) for k, r in pair.items()},
                aa_tps_at_zero={
                    k: aa_speed(r, verify, k, 0.0, ratio) for k, r in pair.items()
                },
                extra_us=shared_window(
                    lambda e: {
                        k: aa_speed(r, verify, k, e, ratio) for k, r in pair.items()
                    },
                    upper,
                    lower,
                ),
            )
        )
    return sorted(results, key=lambda b: b["copy_share"])


def subsets(per_context, ordered, verify, ratio, upper, lower=0.0):
    results = []
    for size in SUBSET_SIZES:
        chosen = [b["book"] for b in ordered[:size]]

        def medians(e):
            return {
                k: statistics.median(
                    aa_speed(per_context[k][i], verify, k, e, ratio) for i in chosen
                )
                for k in AA_TPS
            }

        results.append(
            dict(
                books=size,
                aa_tps_at_zero=medians(0.0),
                extra_us=shared_window(medians, upper, lower),
            )
        )
    return results


def check_books(lookup, verify, ratio, windows):
    rows = {
        (r["data"], r["minimum_ngram"], r["draft_tokens"]): r["per_row"]
        for r in lookup["results"]
        if r["prompt"] == "full" and "per_row" in r
    }
    results = []
    for minimum in MINIMUM_NGRAMS:
        upper = windows[(minimum, DRAFT_TOKENS)][1]
        for style in STYLES:
            per_context = {
                k: rows[(f"aa-docs-think-{c}-{style}", minimum, DRAFT_TOKENS)]
                for k, c in CONTEXTS.items()
            }
            ordered = books(per_context, verify, ratio, upper)
            results.append(
                dict(
                    minimum_ngram=minimum,
                    style=style,
                    extra_upper_us=upper,
                    fitting_books=sum(b["extra_us"] is not None for b in ordered),
                    books=ordered,
                    subsets=subsets(per_context, ordered, verify, ratio, upper),
                )
            )
    return results


def check_spread(rows, step, extra, rng):
    speeds = lookup_speeds(rows, step, extra)
    median, p80, maximum = ratios(speeds)
    drawn = np.array(
        [
            ratios(rng.choice(speeds, SAMPLE_SIZE, replace=True))[1:]
            for _ in range(DRAWS)
        ]
    )
    result = dict(median_tps=float(median), p80_ratio=p80, maximum_ratio=maximum)
    for name, column in zip(("p80", "maximum"), drawn.T):
        low, high = (float(np.percentile(column, q)) for q in (5, 95))
        result[f"{name}_ratio_5_95"] = [low, high]
        result[f"nvidia_{name}_inside"] = bool(low <= NVIDIA_RATIOS[name] <= high)
        result[f"{name}_ratio_share_at_or_below_nvidia"] = float(
            np.mean(column <= NVIDIA_RATIOS[name])
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    verify = verification_steps(
        load("article-sweep-results.json"), load("article-speculative-results.json")
    )
    windows = {
        (r["minimum_ngram"], r["draft_tokens"]): r["coding_extra_us"]
        for r in load("gpu-cc302-peak.json")["results"]
    }
    book_results = {}
    for name, (lookup_file, analysis_file) in DATA_SETS.items():
        ratio = load(analysis_file)["token_ratio"]
        book_results[name] = dict(
            token_ratio=ratio,
            settings=check_books(load(lookup_file), verify, ratio, windows),
        )
        for s in book_results[name]["settings"]:
            print(
                f"{name:12s} n_min={s['minimum_ngram']} {s['style']:7s}"
                f" books fitting {s['fitting_books']}/16;"
                f" subsets {[x['extra_us'] for x in s['subsets']]}"
            )
    step = context_model(verify)
    lookup = {
        (r["minimum_ngram"], r["draft_tokens"]): r["per_row"]
        for r in load("gpu-cc302-lookup-results.json")["results"]
        if r["data"] == CODING and "per_row" in r
    }
    rng = np.random.default_rng(SEED)
    spread_results = []
    for minimum, draft_tokens, extra in SPREAD_SETTINGS:
        result = dict(
            minimum_ngram=minimum,
            draft_tokens=draft_tokens,
            extra_us=extra,
            turns=len(lookup[(minimum, draft_tokens)]),
            **check_spread(lookup[(minimum, draft_tokens)], step, extra, rng),
        )
        spread_results.append(result)
        print(
            f"coding n_min={minimum} K={draft_tokens} at {extra} µs:"
            f" P80 {result['p80_ratio']:.3f} {result['p80_ratio_5_95']},"
            f" max {result['maximum_ratio']:.3f} {result['maximum_ratio_5_95']}"
        )
    write_json(
        args.summary,
        dict(
            band=BAND,
            aa_tps=AA_TPS,
            nvidia_ratios=NVIDIA_RATIOS,
            sample_size=SAMPLE_SIZE,
            draws=DRAWS,
            seed=SEED,
            books=book_results,
            coding_spread=spread_results,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
