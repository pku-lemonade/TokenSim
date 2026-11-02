"""Recompute the AA explanations' tests under alternative LP30 attention costs.

Reads the verification steps captured by scripts/run_lpx_attention_sensitivity.py
and, for each variant and for the row-independent bound, recomputes the plain
AA window, the coding window of each lookup setting, the fastest 302-prompt
code-completion request at the window's upper end, the AA-like book answers
that fit both AA values inside the window and the joint condition of one
lookup drafter on every run.  It then scales the growth of the multi-row steps
from 10K to 100K between the row-independent bound and the baseline, a scan
added after the variants' results.  See docs/lpx-attention-sensitivity.md.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_lpx_cc16k_peak import (
    COMPLETION,
    EXAMPLE_TPS,
    coding_window,
    completion,
)
from scripts.analyze_lpx_coding_spread import context_model, lookup_speeds
from scripts.analyze_lpx_published_conditions import (
    AA_TPS,
    load,
    verification_steps,
    window,
)
from scripts.analyze_lpx_story_c import (
    CODING,
    CONTEXTS,
    DATA_SETS,
    STYLES,
    aa_speed,
    books,
    extra_grid,
    in_band,
    subsets,
)
from scripts.search_lpx_native import write_json

BOUND = "row-independent"
SETTINGS = tuple((n, k) for n in (1, 2) for k in (1, 3, 5, 7))
# Story A prices AA with plain decoding and coding with this lookup setting.
STORY_A_SETTING = (1, 3)
SCAN_SHARES = tuple(x / 20 for x in range(21))


def captured_steps(summary):
    """Verification steps by variant and verified rows."""
    steps = {}
    for row in summary["rows"]:
        steps.setdefault(row["variant"], {})[row["rows"]] = row["step_us"]
    return steps


def scaled_growth(verify, share):
    """Steps whose growth from 10K beyond the one-row growth is scaled by share.

    A share of 0 gives every row count the one-row growth with context, the
    row-independent bound, and a share of 1 keeps the steps.
    """
    one = verify[1]
    return {
        rows: {
            k: steps["aa_10k"]
            + one[k]
            - one["aa_10k"]
            + share * (steps[k] - steps["aa_10k"] - one[k] + one["aa_10k"])
            for k in steps
        }
        for rows, steps in verify.items()
    }


def reproduction(verify, committed):
    """Largest step difference from the committed steps and the 7-row change."""
    return dict(
        maximum_difference_us=max(
            abs(verify[rows][k] - committed[rows][k])
            for rows in committed
            if rows != 7
            for k in committed[rows]
        ),
        seven_rows_minus_interpolation_us={
            k: verify[7][k] - committed[7][k] for k in committed[7]
        },
    )


def overlap(a, b):
    if a is None or b is None:
        return None
    low, high = max(a[0], b[0]), min(a[1], b[1])
    return [low, high] if low <= high else None


def book_rows(lookup, minimum, draft_tokens, style):
    rows = {
        r["data"]: r["per_row"]
        for r in lookup["results"]
        if r["prompt"] == "full"
        and r["minimum_ngram"] == minimum
        and r["draft_tokens"] == draft_tokens
        and "per_row" in r
    }
    return {k: rows[f"aa-docs-think-{c}-{style}"] for k, c in CONTEXTS.items()}


def joint_window(per_style, verify, ratio, completions, coding):
    """Extra times in the coding window that reach the example and fit AA with
    the median of all books of one question style."""

    def fits(extra, per_context):
        return in_band(
            {
                k: statistics.median(
                    aa_speed(r, verify, k, extra, ratio) for r in per_context[k]
                )
                for k in AA_TPS
            }
        )

    return [
        extra
        for extra in extra_grid(coding[1])
        if extra >= coding[0]
        and lookup_speeds(completions, context_model(verify), extra).max()
        >= EXAMPLE_TPS
        and any(fits(extra, per_context) for per_context in per_style.values())
    ]


def analyze_variant(verify, coding_rows, completion_rows, book_sets):
    step = context_model(verify)
    plain = {
        name: window(verify[1], data["token_ratio"]) for name, data in book_sets.items()
    }
    settings = []
    for minimum, draft_tokens in SETTINGS:
        coding = coding_window(coding_rows[(minimum, draft_tokens)], step)
        result = dict(
            minimum_ngram=minimum,
            draft_tokens=draft_tokens,
            coding_extra_us=coding,
        )
        if coding is not None:
            result["completion_at_upper"] = completion(
                completion_rows[(minimum, draft_tokens)], step, coding[1]
            )
        result["books"] = {}
        for name, data in book_sets.items():
            per_style = {
                style: book_rows(data["lookup"], minimum, draft_tokens, style)
                for style in STYLES
            }
            styles = {}
            for style, per_context in per_style.items():
                if coding is None:
                    styles[style] = dict(fitting_books=0)
                    continue
                ordered = books(
                    per_context,
                    verify,
                    data["token_ratio"],
                    coding[1],
                    coding[0],
                    draft_tokens,
                )
                styles[style] = dict(
                    fitting_books=sum(b["extra_us"] is not None for b in ordered),
                    fitting_extra_us=[
                        b["extra_us"] for b in ordered if b["extra_us"] is not None
                    ],
                    subsets=subsets(
                        per_context,
                        ordered,
                        verify,
                        data["token_ratio"],
                        coding[1],
                        coding[0],
                    ),
                )
            joint = (
                joint_window(
                    per_style,
                    verify,
                    data["token_ratio"],
                    completion_rows[(minimum, draft_tokens)],
                    coding,
                )
                if coding is not None
                else []
            )
            result["books"][name] = dict(
                fitting_pairs=sum(s["fitting_books"] for s in styles.values()),
                styles=styles,
                joint_extra_us=[min(joint), max(joint)] if joint else None,
            )
        settings.append(result)
    story_a_coding = next(
        s["coding_extra_us"]
        for s in settings
        if (s["minimum_ngram"], s["draft_tokens"]) == STORY_A_SETTING
    )
    return dict(
        steps_us=verify,
        increment_us={
            rows: verify[rows]["aa_100k"] - verify[rows]["aa_10k"] for rows in (1, 8)
        },
        plain_aa_extra_us=plain,
        story_a_extra_us={
            name: overlap(w, story_a_coding) for name, w in plain.items()
        },
        settings=settings,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=Path,
        default=ROOT / "experiments/lpx/attention-sensitivity-steps.json",
    )
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    captured = captured_steps(json.loads(args.steps.read_text()))
    committed = verification_steps(
        load("article-sweep-results.json"), load("article-speculative-results.json")
    )
    variants = {**captured, BOUND: scaled_growth(captured["baseline"], 0.0)}
    replay = {
        (r["data"], r["minimum_ngram"], r["draft_tokens"]): r["per_row"]
        for r in load("gpu-cc302-lookup-results.json")["results"]
        if "per_row" in r
    }
    coding_rows = {s: replay[(CODING, *s)] for s in SETTINGS}
    completion_rows = {s: replay[(COMPLETION, *s)] for s in SETTINGS}
    book_sets = {
        name: dict(
            lookup=load(lookup_file), token_ratio=load(analysis_file)["token_ratio"]
        )
        for name, (lookup_file, analysis_file) in DATA_SETS.items()
    }
    results = {}
    for name, verify in variants.items():
        results[name] = analyze_variant(verify, coding_rows, completion_rows, book_sets)
        r = results[name]
        print(
            f"{name:15s} increments {r['increment_us'][1]:.1f}/{r['increment_us'][8]:.1f}"
            f" plain {r['plain_aa_extra_us']} story A {r['story_a_extra_us']}"
        )
        for s in r["settings"]:
            peak = s.get("completion_at_upper", {})
            print(
                f"  n={s['minimum_ngram']} K={s['draft_tokens']}"
                f" coding {s['coding_extra_us']}"
                f" max {peak.get('maximum_tps', 0):.0f}"
                f" pairs {[b['fitting_pairs'] for b in s['books'].values()]}"
                f" joint {[b['joint_extra_us'] for b in s['books'].values()]}"
            )
    scan = []
    for share in SCAN_SHARES:
        verify = scaled_growth(captured["baseline"], share)
        result = analyze_variant(verify, coding_rows, completion_rows, book_sets)
        scan.append(
            dict(
                share=share,
                increment_us=result["increment_us"],
                settings=[
                    dict(
                        minimum_ngram=s["minimum_ngram"],
                        draft_tokens=s["draft_tokens"],
                        coding_extra_us=s["coding_extra_us"],
                        fitting_pairs={
                            n: b["fitting_pairs"] for n, b in s["books"].items()
                        },
                        joint_extra_us={
                            n: b["joint_extra_us"] for n, b in s["books"].items()
                        },
                    )
                    for s in result["settings"]
                ],
            )
        )
        joint = [
            (s["minimum_ngram"], s["draft_tokens"], n, w)
            for s in scan[-1]["settings"]
            for n, w in s["joint_extra_us"].items()
            if w
        ]
        print(
            f"scan share {share:.2f}: 8-row increment"
            f" {result['increment_us'][8]:.1f} µs, joint {joint}"
        )
    write_json(
        args.summary,
        dict(
            aa_tps=AA_TPS,
            example_tps=EXAMPLE_TPS,
            token_ratios={n: d["token_ratio"] for n, d in book_sets.items()},
            baseline_reproduction=reproduction(captured["baseline"], committed),
            variants=results,
            growth_scan=scan,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
