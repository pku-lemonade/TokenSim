"""Find one extra time per step that fits AA and SPEED-Bench at measured MTP acceptance.

Reads the combined GPU acceptance analysis written by
analyze_lpx_gpu_acceptance.py.  For each measured MTP draft length, coding
thinking mode and drafter cost it finds the constant extra time per step
that places the median of the SPEED-Bench coding turns within 5% of
4,767 TPS, intersects it with the AA window of each question style and
prices the 16K code-completion turns at zero and at the smallest shared
extra time.  See docs/lpx-gpu-joint-fit.md.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_lpx_api_outputs import STYLES
from scripts.analyze_lpx_coding_spread import context_model, PUBLISHED
from scripts.analyze_lpx_gpu_acceptance import (
    DRAFTERS,
    coding_speed,
    mtp_steps,
    speed_summary,
)
from scripts.analyze_lpx_published_conditions import BAND, load, verification_steps
from scripts.search_lpx_native import write_json

EXTRA_GRID_US = [x / 2 for x in range(2001)]
CODING_SETS = ("speed-coding-plain", "speed-coding-think")
COMPLETION_SET = "speed-cc16k-plain"
EXAMPLE_TPS = 10996


def speeds(row, verify, draft_step, extra):
    step = lambda rows, context: verify(rows, context) + extra
    return [
        coding_speed(turn, step, draft_step, row["draft_tokens"])
        for turn in row["turns"]
    ]


def intersect(a, b):
    if a is None or b is None:
        return None
    low, high = max(a[0], b[0]), min(a[1], b[1])
    return [low, high] if low <= high else None


def completion(row, verify, draft_step, extra):
    values = speeds(row, verify, draft_step, extra)
    return dict(
        extra_us=extra,
        maximum_tps=max(values),
        at_or_above_example=sum(v >= EXAMPLE_TPS for v in values),
        turn_count=len(values),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = json.loads(args.analysis.read_text())
    speculative = load("article-speculative-results.json")
    verify = context_model(
        verification_steps(load("article-sweep-results.json"), speculative)
    )
    aa = {
        (row["config"]["K"], row["style"], row["drafter"]): row["extra_us"]
        for row in analysis["aa"]
    }
    mtp = [row for row in analysis["results"] if row["config"]["method"] == "mtp"]
    coding, completions = [], []
    for drafter in DRAFTERS:
        for row in mtp:
            k = row["draft_tokens"]
            simulation = mtp_steps(speculative, k, drafter)
            draft_step = context_model({k + 1: simulation["draft_us"]})
            if row["data"] in CODING_SETS:
                fits = [
                    extra
                    for extra in EXTRA_GRID_US
                    if abs(
                        speed_summary(speeds(row, verify, draft_step, extra))[
                            "median_tps"
                        ]
                        / PUBLISHED["median"]
                        - 1
                    )
                    <= BAND
                ]
                window = [min(fits), max(fits)] if fits else None
                shared = {s: intersect(window, aa.get((k, s, drafter))) for s in STYLES}
                spread = {
                    s: speed_summary(speeds(row, verify, draft_step, w[0]))
                    for s, w in shared.items()
                    if w
                }
                coding.append(
                    dict(
                        data=row["data"],
                        K=k,
                        drafter=drafter,
                        tau=row["tau"],
                        coding_extra_us=window,
                        aa_extra_us={s: aa.get((k, s, drafter)) for s in STYLES},
                        shared_extra_us=shared,
                        spread_at_shared_low=spread,
                    )
                )
                print(
                    f"K={k} {row['data']:20s} {drafter} coding {window}"
                    f" shared {shared}"
                )
    for drafter in DRAFTERS:
        for row in mtp:
            if row["data"] != COMPLETION_SET:
                continue
            k = row["draft_tokens"]
            simulation = mtp_steps(speculative, k, drafter)
            draft_step = context_model({k + 1: simulation["draft_us"]})
            lows = [
                w[0]
                for c in coding
                if c["K"] == k and c["drafter"] == drafter
                for w in c["shared_extra_us"].values()
                if w
            ]
            entry = dict(
                K=k,
                drafter=drafter,
                tau=row["tau"],
                at_zero=completion(row, verify, draft_step, 0.0),
                at_smallest_shared=(
                    completion(row, verify, draft_step, min(lows)) if lows else None
                ),
            )
            completions.append(entry)
            print(
                f"K={k} cc16k {drafter} max at 0 {entry['at_zero']['maximum_tps']:.0f}"
                f" at shared {entry['at_smallest_shared']}"
            )
    write_json(
        args.summary,
        dict(
            source=str(args.analysis),
            coding_median=PUBLISHED["median"],
            example_tps=EXAMPLE_TPS,
            band=BAND,
            coding=coding,
            completion=completions,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
