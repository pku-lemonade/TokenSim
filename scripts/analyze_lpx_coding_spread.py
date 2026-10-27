"""Per-request coding speed spread under plain decoding and lookup drafting.

Prices each recorded coding request on TP32/PP8 with DCP 8 at its own
proposal-length counts and mean decode context, and compares the ratios of
the 80th percentile and the maximum to the median with NVIDIA's SPEED-Bench
coding statistics.  Reads experiments/lpx/lookup-coding-rows.json, written
by measure_lpx_lookup_acceptance.py with --per-row.  See
docs/lpx-coding-spread.md.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_lpx_published_conditions import load, verification_steps
from scripts.run_lpx_speculative import AA_CONTEXTS, CODING_CONTEXT
from scripts.search_lpx_native import write_json

# NVIDIA's SPEED-Bench coding statistics for the deployment, in output TPS.
PUBLISHED = dict(median=4767, p80=5520, maximum=6981)
INPUT_TOKENS = 16000
EXTRA_US = (0.0, 20.0, 50.0)
SAMPLE_SIZES = (40, 80, 160)
DRAWS = 2000
SEED = 20261027


def context_model(verify):
    """Step in microseconds of an n-row forward at a decode context."""
    (_, low, _), (_, high, _) = AA_CONTEXTS
    coding = CODING_CONTEXT[1]

    def step(rows, context):
        v = verify[rows]
        slope = (v["aa_100k"] - v["aa_10k"]) / (high - low)
        return v["coding_16k"] + slope * (context - coding)

    return step


def plain_speeds(rows, step, extra):
    return np.array(
        [1e6 / (step(1, INPUT_TOKENS + tokens / 2) + extra) for tokens, _ in rows]
    )


def lookup_speeds(rows, step, extra):
    speeds = []
    for tokens, counts in rows:
        context = INPUT_TOKENS + tokens / 2
        time = sum(c * (step(n + 1, context) + extra) for n, c in enumerate(counts))
        speeds.append(tokens / time * 1e6)
    return np.array(speeds)


def ratios(speeds):
    median = np.median(speeds)
    return median, np.percentile(speeds, 80) / median, speeds.max() / median


def spread(speeds, rng):
    median, p80, maximum = ratios(speeds)
    samples = {}
    for size in (n for n in SAMPLE_SIZES if n < len(speeds)):
        drawn = [
            ratios(rng.choice(speeds, size, replace=False))[1:] for _ in range(DRAWS)
        ]
        p80s, maxima = np.array(drawn).T
        samples[str(size)] = dict(
            p80_ratio=[float(np.percentile(p80s, q)) for q in (10, 50, 90)],
            maximum_ratio=[float(np.percentile(maxima, q)) for q in (10, 50, 90)],
        )
    return dict(
        median_tps=float(median),
        p80_ratio=float(p80),
        maximum_ratio=float(maximum),
        samples=samples,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=Path, default=ROOT / "experiments/lpx/lookup-coding-rows.json"
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    step = context_model(
        verification_steps(
            load("article-sweep-results.json"), load("article-speculative-results.json")
        )
    )
    rng = np.random.default_rng(SEED)
    measured = json.loads(args.rows.read_text())["results"]
    first = next(r for r in measured if "per_row" in r)
    results = []
    for extra in EXTRA_US:
        settings = [("plain", None, None, plain_speeds(first["per_row"], step, extra))]
        settings += [
            (
                r["data"],
                r["minimum_ngram"],
                r["draft_tokens"],
                lookup_speeds(r["per_row"], step, extra),
            )
            for r in measured
            if "per_row" in r
        ]
        for data, minimum, draft_tokens, speeds in settings:
            row = dict(
                data=data,
                minimum_ngram=minimum,
                draft_tokens=draft_tokens,
                extra_us=extra,
                **spread(speeds, rng),
            )
            results.append(row)
            print(
                f"extra {extra:4.0f} {data:20s} n_min={minimum} K={draft_tokens}:"
                f" median {row['median_tps']:6.0f},"
                f" P80 x{row['p80_ratio']:.3f},"
                f" max x{row['maximum_ratio']:.3f}",
                flush=True,
            )
    if args.summary:
        write_json(
            args.summary,
            dict(
                published=PUBLISHED,
                input_tokens=INPUT_TOKENS,
                sample_sizes=SAMPLE_SIZES,
                draws=DRAWS,
                seed=SEED,
                results=results,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
