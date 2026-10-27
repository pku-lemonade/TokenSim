"""Price the AA workloads at the lookup acceptance measured on AA-like outputs.

Reads the lookup replay of the generated document answers, written by
measure_lpx_lookup_acceptance.py with ``--per-row`` and data names
``aa-docs-think-<context>-<style>``, and their token ratio from
measure_lpx_token_ratio.py with an ``aa-output`` category.  Each AA workload
is priced on TP32/PP8 with DCP 8 at the proposal lengths measured at its own
context.  It also finds the extra time per step that places the median of
the SPEED-Bench coding requests within 5% of 4,767 TPS, alone and together
with both AA values.  See docs/lpx-api-outputs.md.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_lpx_published_conditions import (
    AA_TPS,
    BAND,
    EFFICIENCY,
    LAYOUT,
    load,
    verification_steps,
)
from scripts.analyze_lpx_coding_spread import context_model, lookup_speeds
from scripts.measure_lpx_lookup_acceptance import equivalent_acceptance
from scripts.search_lpx_native import write_json

CONTEXTS = dict(aa_10k="10k", aa_100k="100k")
STYLES = ("summary", "quotes")
CODING_MEDIAN = 4767
CODING_SETS = ("speed-coding-think", "speed-coding-plain")
EXTRA_GRID_US = [x / 2 for x in range(401)]


def window(steps, tau, ratio):
    """Constant extra time per step that places both AA speeds in the band."""
    low = max(
        0.0,
        *(
            tau[k] * 1e6 / (ratio * v * (1 + BAND)) - steps[k]
            for k, v in AA_TPS.items()
        ),
    )
    high = min(
        tau[k] * 1e6 / (ratio * v * (1 - BAND)) - steps[k] for k, v in AA_TPS.items()
    )
    return [low, high] if low <= high else None


def priced(steps, tau, ratio):
    tps = {k: tau[k] / steps[k] * 1e6 / ratio for k in AA_TPS}
    return dict(
        tau=tau,
        step_us=steps,
        aa_tps=tps,
        error={k: tps[k] / v - 1 for k, v in AA_TPS.items()},
        extra_us=window(steps, tau, ratio),
    )


def lookup_pricing(lookup, verify, ratio):
    rows = {
        (r["data"], r["minimum_ngram"], r["draft_tokens"]): r
        for r in lookup["results"]
        if r["prompt"] == "full"
    }
    results = []
    for style in STYLES:
        for (data, minimum, draft_tokens), _ in sorted(rows.items()):
            if data != f"aa-docs-think-10k-{style}":
                continue
            measured = {
                k: rows[(f"aa-docs-think-{c}-{style}", minimum, draft_tokens)]
                for k, c in CONTEXTS.items()
            }
            steps = {
                k: sum(
                    s * verify[n + 1][k]
                    for n, s in enumerate(measured[k]["proposal_length_shares"])
                )
                for k in AA_TPS
            }
            results.append(
                dict(
                    style=style,
                    minimum_ngram=minimum,
                    draft_tokens=draft_tokens,
                    acceptance={
                        k: measured[k]["equivalent_acceptance"] for k in AA_TPS
                    },
                    **priced(steps, {k: measured[k]["tau"] for k in AA_TPS}, ratio),
                )
            )
    return results


def paired_acceptance(lookup, minimum=1, draft_tokens=7):
    """Per-book acceptance at 10K and 100K from the per-row replay."""
    results = {}
    for style in STYLES:
        per_context = {}
        for k, c in CONTEXTS.items():
            row = next(
                r
                for r in lookup["results"]
                if r["data"] == f"aa-docs-think-{c}-{style}"
                and r["prompt"] == "full"
                and r["minimum_ngram"] == minimum
                and r["draft_tokens"] == draft_tokens
            )
            per_context[k] = [
                equivalent_acceptance(tokens / sum(counts), draft_tokens)
                for tokens, counts in row["per_row"]
            ]
        differences = [
            b - a for a, b in zip(per_context["aa_10k"], per_context["aa_100k"])
        ]
        results[style] = dict(
            acceptance=per_context,
            difference=differences,
            higher_at_100k=sum(d > 0 for d in differences),
        )
    return results


def coding_windows(lookup, verify, aa_rows):
    """Extra time that fits the SPEED-Bench coding median, alone and with AA."""
    step = context_model(verify)
    aa = {(r["style"], r["minimum_ngram"], r["draft_tokens"]): r for r in aa_rows}
    results = []
    for row in lookup["results"]:
        if row["data"] not in CODING_SETS or "per_row" not in row:
            continue
        fits = []
        for extra in EXTRA_GRID_US:
            speeds = sorted(lookup_speeds(row["per_row"], step, extra))
            median = speeds[len(speeds) // 2]
            if len(speeds) % 2 == 0:
                median = (median + speeds[len(speeds) // 2 - 1]) / 2
            if abs(median / CODING_MEDIAN - 1) <= BAND:
                fits.append(extra)
        coding = [min(fits), max(fits)] if fits else None
        shared = {}
        for style in STYLES:
            window_aa = aa[(style, row["minimum_ngram"], row["draft_tokens"])][
                "extra_us"
            ]
            if coding and window_aa:
                low, high = max(coding[0], window_aa[0]), min(coding[1], window_aa[1])
                shared[style] = [low, high] if low <= high else None
            else:
                shared[style] = None
        results.append(
            dict(
                data=row["data"],
                minimum_ngram=row["minimum_ngram"],
                draft_tokens=row["draft_tokens"],
                coding_extra_us=coding,
                shared_extra_us=shared,
            )
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--ratio", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    sweep = load("article-sweep-results.json")
    verify = verification_steps(sweep, load("article-speculative-results.json"))
    lookup = json.loads(args.lookup.read_text())
    ratio = json.loads(args.ratio.read_text())["categories"]["aa-output"][
        "gemma_per_o200k"
    ]
    plain_steps = next(
        r["step_us"]
        for r in sweep["rows"]
        if r["layout_name"] == LAYOUT and r["attention_mfu"] == EFFICIENCY
    )
    result = dict(
        token_ratio=ratio,
        plain=priced(
            {k: plain_steps[k] for k in AA_TPS}, {k: 1.0 for k in AA_TPS}, ratio
        ),
        lookup=lookup_pricing(lookup, verify, ratio),
        paired=paired_acceptance(lookup),
    )
    result["coding"] = coding_windows(lookup, verify, result["lookup"])
    print(f"token ratio {ratio:.4f}; plain extra {result['plain']['extra_us']}")
    for row in result["lookup"]:
        print(
            f"{row['style']:7s} n_min={row['minimum_ngram']} K={row['draft_tokens']}:"
            f" alpha {row['acceptance']['aa_10k']:.3f}/{row['acceptance']['aa_100k']:.3f},"
            f" error {row['error']['aa_10k']:+.1%}/{row['error']['aa_100k']:+.1%},"
            f" extra {row['extra_us']}"
        )
    for style, paired in result["paired"].items():
        print(
            f"{style}: 100K higher in {paired['higher_at_100k']} of"
            f" {len(paired['difference'])} books"
        )
    for row in result["coding"]:
        print(
            f"{row['data']} n_min={row['minimum_ngram']} K={row['draft_tokens']}:"
            f" coding extra {row['coding_extra_us']}, shared {row['shared_extra_us']}"
        )
    if args.summary:
        write_json(args.summary, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
