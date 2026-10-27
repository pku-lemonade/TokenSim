"""Derived tables of the published-condition AA records.

Reads committed results only and recomputes the values that the records
derive from them: extra-time windows at measured token ratios, their
cross-tray latency equivalents, speculative matches at other ratios, the
per-workload acceptances of the TP32 layout, the prompt-lookup pricing and
the GroqRack-adjusted Llama-2 70B speeds.  See docs/lpx-token-ratio.md,
docs/lpx-article-speculative.md, docs/lpx-lookup-acceptance.md and
docs/groqrack-llama2-7b-validation.md.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lpx.speculative_fit import acceptance_for, shared_match
from scripts.search_lpx_native import write_json

RESULTS = ROOT / "experiments/lpx"
AA_TPS = dict(aa_10k=3382, aa_100k=3431)
BAND = 0.05
LAYOUT, EFFICIENCY = "tp32-pp8-kvp8", 0.5
TOKEN_RATIOS = (1.0, 1.031, 1.035, 1.036, 1.048, 1.053, 1.069, 1.073, 1.08, 1.09)
SPECULATIVE_RATIOS = (1.0, 1.035, 1.036, 1.073)
# Step growth per microsecond of middle-tier latency (docs/lpx-tray-latency.md).
LATENCY_SLOPE_US = 249.0
MIDDLE_US = 0.75


def load(name):
    return json.loads((RESULTS / name).read_text())


def window(steps, ratio, tokens=1.0):
    """Constant extra time per step that places both AA speeds in the band."""
    low = max(
        0.0,
        *(
            tokens * 1e6 / (ratio * v * (1 + BAND)) - steps[k]
            for k, v in AA_TPS.items()
        ),
    )
    high = min(
        tokens * 1e6 / (ratio * v * (1 - BAND)) - steps[k] for k, v in AA_TPS.items()
    )
    return [low, high] if low <= high else None


def ratio_windows(sweep):
    rows = []
    for row in sweep["rows"]:
        if row["status"] != "ok":
            continue
        for ratio in TOKEN_RATIOS:
            steps = row["step_us"]
            extra = window(steps, ratio)
            rows.append(
                dict(
                    layout_name=row["layout_name"],
                    attention_mfu=row["attention_mfu"],
                    token_ratio=ratio,
                    error={
                        k: 1e6 / steps[k] / ratio / v - 1 for k, v in AA_TPS.items()
                    },
                    extra_us=extra,
                    middle_us=(
                        [MIDDLE_US + x / LATENCY_SLOPE_US for x in extra]
                        if extra
                        else None
                    ),
                )
            )
    return rows


def speculative_matches(speculative):
    rows = []
    for ratio in SPECULATIVE_RATIOS:
        bands = [
            (v * ratio * (1 - BAND), v * ratio * (1 + BAND)) for v in AA_TPS.values()
        ]
        for row in speculative["rows"]:
            steps = [row["step_us"][k] * 1e-6 for k in AA_TPS]
            match = shared_match(steps, bands, row["draft_tokens"])
            if match:
                rows.append(
                    dict(
                        token_ratio=ratio,
                        layout_name=row["layout_name"],
                        attention_mfu=row["attention_mfu"],
                        draft_tokens=row["draft_tokens"],
                        drafter=row["drafter"],
                        shared_acceptance=list(match),
                    )
                )
    return rows


def workload_acceptances(speculative, ratio=1.036):
    return [
        dict(
            draft_tokens=row["draft_tokens"],
            drafter=row["drafter"],
            token_ratio=ratio,
            acceptance={
                k: acceptance_for(
                    v * ratio * row["step_us"][k] * 1e-6, row["draft_tokens"]
                )
                for k, v in AA_TPS.items()
            },
        )
        for row in speculative["rows"]
        if row["layout_name"] == LAYOUT and row["attention_mfu"] == EFFICIENCY
    ]


def verification_steps(sweep, speculative):
    steps = {
        1: next(
            r["step_us"]
            for r in sweep["rows"]
            if r["layout_name"] == LAYOUT and r["attention_mfu"] == EFFICIENCY
        )
    }
    for row in speculative["rows"]:
        if (
            row["layout_name"] == LAYOUT
            and row["attention_mfu"] == EFFICIENCY
            and row["drafter"] == "lookup"
        ):
            steps[row["draft_tokens"] + 1] = row["verify_us"]
    steps[7] = {k: (steps[6][k] + steps[8][k]) / 2 for k in steps[1]}
    return steps


def lookup_pricing(lookup, verify, ratio=1.036):
    rows = []
    for row in lookup["results"]:
        if row["prompt"] != "full":
            continue
        shares = row["proposal_length_shares"]
        steps = {
            k: sum(s * verify[n + 1][k] for n, s in enumerate(shares)) for k in AA_TPS
        }
        tps = {k: row["tau"] / steps[k] * 1e6 / ratio for k in AA_TPS}
        rows.append(
            dict(
                data=row["data"],
                minimum_ngram=row["minimum_ngram"],
                draft_tokens=row["draft_tokens"],
                token_ratio=ratio,
                tau=row["tau"],
                step_us=steps,
                aa_tps=tps,
                error={k: tps[k] / v - 1 for k, v in AA_TPS.items()},
                extra_us=window(steps, ratio, row["tau"]),
            )
        )
    return rows


def groqrack_adjusted(groqrack, llama70b):
    direct = [c["step_us"] for c in groqrack["cases"] if c["algorithm"] == "direct"]
    low, high = groqrack["measured_tpot_ms"]
    missing = [low - max(direct) / 1e3, high - min(direct) / 1e3]
    steps = [c["step_us"] for c in llama70b["cases"] if c["algorithm"] == "direct"]
    return dict(
        missing_ms=missing,
        llama70b_direct_step_us=[min(steps), max(steps)],
        llama70b_tps=[
            1e3 / (max(steps) / 1e3 + missing[1]),
            1e3 / (min(steps) / 1e3 + missing[0]),
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    sweep = load("article-sweep-results.json")
    speculative = load("article-speculative-results.json")
    result = dict(
        ratio_windows=ratio_windows(sweep),
        speculative_matches=speculative_matches(speculative),
        workload_acceptances=workload_acceptances(speculative),
        lookup_pricing=lookup_pricing(
            load("lookup-acceptance-results.json"),
            verification_steps(sweep, speculative),
        ),
        groqrack=groqrack_adjusted(
            load("groqrack-llama2-7b-results.json"),
            load("groqchip-v1-llama2-results.json"),
        ),
    )
    for row in result["ratio_windows"]:
        if row["layout_name"] == LAYOUT and row["attention_mfu"] == 0.3:
            print(f"ratio {row['token_ratio']}: extra {row['extra_us']}")
    counts = {}
    for row in result["speculative_matches"]:
        counts[row["token_ratio"]] = counts.get(row["token_ratio"], 0) + 1
    print(f"speculative matches by ratio: {counts}")
    print(f"GroqRack-adjusted 70B: {result['groqrack']}")
    if args.summary:
        write_json(args.summary, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
