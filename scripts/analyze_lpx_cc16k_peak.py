"""Price the fastest 16K code-completion request under each lookup setting.

Reads a lookup replay written by measure_lpx_lookup_acceptance.py with
``--per-row``, holding the SPEED-Bench coding answers without thinking as
``speed-coding-plain``, the code-completion answers as ``speed-cc16k-plain``
and, optionally, the answers to the 40 earlier code-completion prompts as
``speed-cc16k40-plain``.  For each minimum n-gram and draft length it finds
the extra time per step that places the coding median within 5% of
4,767 TPS, and prices every code-completion request at zero extra time, at
13, 21 and 26 µs and at the upper end of that window, against NVIDIA's
10,996 TPS example.  See docs/lpx-cc16k-peak.md.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_lpx_coding_spread import context_model, lookup_speeds, PUBLISHED
from scripts.analyze_lpx_published_conditions import (
    BAND,
    load,
    verification_steps,
)
from scripts.search_lpx_native import write_json

CODING, COMPLETION, EARLIER = (
    "speed-coding-plain",
    "speed-cc16k-plain",
    "speed-cc16k40-plain",
)
EXAMPLE_TPS = 10996
EXTRA_GRID_US = [x / 2 for x in range(201)]
FIXED_EXTRA_US = (0.0, 13.0, 21.0, 26.0)


def coding_window(rows, step):
    fits = [
        extra
        for extra in EXTRA_GRID_US
        if abs(
            statistics.median(lookup_speeds(rows, step, extra)) / PUBLISHED["median"]
            - 1
        )
        <= BAND
    ]
    return [min(fits), max(fits)] if fits else None


def completion(rows, step, extra):
    speeds = lookup_speeds(rows, step, extra)
    taus = [tokens / sum(counts) for tokens, counts in rows]
    return dict(
        extra_us=extra,
        requests=len(rows),
        median_tps=float(statistics.median(speeds)),
        maximum_tps=float(max(speeds)),
        at_or_above_example=int(sum(speeds >= EXAMPLE_TPS)),
        mean_tau=statistics.mean(taus),
        maximum_tau=max(taus),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    lookup = json.loads(args.lookup.read_text())["results"]
    step = context_model(
        verification_steps(
            load("article-sweep-results.json"), load("article-speculative-results.json")
        )
    )
    rows = {
        (r["data"], r["minimum_ngram"], r["draft_tokens"]): r["per_row"]
        for r in lookup
        if "per_row" in r
    }
    settings = sorted({(n, k) for data, n, k in rows if data == COMPLETION})
    results = []
    for n, k in settings:
        coding = rows.get((CODING, n, k))
        window = coding_window(coding, step) if coding else None
        extras = list(FIXED_EXTRA_US) + ([window[1]] if window else [])
        entry = dict(
            minimum_ngram=n,
            draft_tokens=k,
            coding_extra_us=window,
            completion=[completion(rows[(COMPLETION, n, k)], step, e) for e in extras],
        )
        if (EARLIER, n, k) in rows:
            entry["earlier_40"] = [
                completion(rows[(EARLIER, n, k)], step, e) for e in extras
            ]
        results.append(entry)
        peak = max(c["maximum_tps"] for c in entry["completion"])
        print(
            f"n_min={n} K={k} coding window {window}"
            f" fastest {entry['completion'][0]['maximum_tps']:.0f} TPS at 0 µs,"
            f" {entry['completion'][-1]['maximum_tps']:.0f} at"
            f" {entry['completion'][-1]['extra_us']} µs (peak {peak:.0f})"
        )
    write_json(
        args.summary,
        dict(
            source=str(args.lookup),
            coding_median=PUBLISHED["median"],
            example_tps=EXAMPLE_TPS,
            band=BAND,
            results=results,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
