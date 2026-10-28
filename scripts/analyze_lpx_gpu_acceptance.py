"""Summarize measured vLLM acceptance and price it on LP30 TP32/PP8 with DCP8.

Pass one or more prompt-free files from generate_lpx_gpu_outputs.py.
Configurations and truncated smoke runs remain separate. No acceptance or
GPU speed is inferred for turns whose measurements are missing or invalid.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_api_outputs import CONTEXTS, STYLES, priced
from scripts.analyze_lpx_coding_spread import context_model, PUBLISHED
from scripts.analyze_lpx_published_conditions import (
    EFFICIENCY,
    LAYOUT,
    load,
    verification_steps,
)
from scripts.generate_lpx_gemma_outputs import file_name
from scripts.lpx.gpu_metrics import acceptance, output_quality
from scripts.measure_lpx_lookup_acceptance import equivalent_acceptance
from scripts.search_lpx_native import write_json

DRAFTERS = ("mtp-0.05", "mtp-0.10")


def speed_summary(speeds):
    if not speeds or any(not math.isfinite(value) or value <= 0 for value in speeds):
        raise WorkloadValidationError(
            "Speed summary needs finite positive measurements"
        )
    median = statistics.median(speeds)
    p80 = float(np.percentile(speeds, 80))
    maximum = max(speeds)
    return dict(
        turn_count=len(speeds),
        median_tps=median,
        p80_tps=p80,
        maximum_tps=maximum,
        p80_ratio=p80 / median,
        maximum_ratio=maximum / median,
    )


def summarize(outputs):
    groups = defaultdict(list)
    seen = set()
    for output in outputs:
        config_key = json.dumps(output["config"], sort_keys=True)
        data = file_name(
            dict(
                set=output["set"],
                reasoning=output["thinking"],
                meta=output["meta"],
            )
        )
        for turn in output["turns"]:
            identity = (config_key, output["id"], turn["turn"])
            if identity in seen:
                raise WorkloadValidationError(f"Duplicate turn: {output['id']}")
            seen.add(identity)
            groups[(config_key, data)].append(
                dict(
                    id=output["id"],
                    meta=output["meta"],
                    **turn,
                )
            )
    results = []
    for (config_key, data), turns in sorted(groups.items()):
        turns.sort(key=lambda turn: (turn["id"], turn["turn"]))
        config = json.loads(config_key)
        valid_speeds = [
            turn["output_tps"] for turn in turns if turn["output_tps"] is not None
        ]
        result = dict(
            data=data,
            config=config,
            turn_count=len(turns),
            gpu_speed=speed_summary(valid_speeds) if valid_speeds else None,
            untimed_turns=len(turns) - len(valid_speeds),
            prompt="full",
            draft_tokens=config["K"],
        )
        quality = [output_quality(turn) for turn in turns]
        result["quality"] = dict(
            length_limited_turns=sum(
                row["finish_reason"] == "length" for row in quality
            ),
            empty_answer_turns=sum(row["empty_answer"] is True for row in quality),
            repetition_flagged_turns=sum(
                row["repetition_flag"] is True for row in quality
            ),
            threshold_note="128 identical consecutive native tokens; diagnostic chosen after the plain T0 loop, no turns excluded",
        )
        if config["method"] in ("plain", "ngram"):
            result["turns"] = [
                dict(
                    id=turn["id"],
                    turn=turn["turn"],
                    meta=turn["meta"],
                    prompt_tokens=turn["prompt_tokens"],
                    decode_tokens=turn["completion_tokens"] - 1,
                )
                for turn in turns
            ]
            if config["method"] == "ngram":
                result["acceptance_unavailable"] = (
                    "vLLM draft counters exclude zero-proposal forwards; "
                    "the total verification-step count is not measured"
                )
                for measured, turn in zip(result["turns"], turns):
                    measured.update(
                        {
                            field: turn[field]
                            for field in (
                                "drafts",
                                "draft_tokens",
                                "accepted_tokens",
                                "accepted_per_position",
                            )
                        }
                    )
        else:
            measured = [
                acceptance(turn, config["K"], config["method"] == "mtp")
                for turn in turns
            ]
            drafts = sum(turn["drafts"] for turn in turns)
            positions = sorted(
                set().union(*(turn["accepted_per_position"] for turn in turns)), key=int
            )
            result.update(
                tau=statistics.mean(row["tau"] for row in measured),
                pooled_tau=sum(row["decode_tokens"] for row in measured) / drafts,
                per_position_acceptance={
                    position: sum(
                        turn["accepted_per_position"].get(position, 0) for turn in turns
                    )
                    / drafts
                    for position in positions
                },
                turns=[
                    dict(
                        id=turn["id"],
                        turn=turn["turn"],
                        meta=turn["meta"],
                        prompt_tokens=turn["prompt_tokens"],
                        **row,
                    )
                    for turn, row in zip(turns, measured)
                ],
            )
            if config["method"] == "mtp":
                result["per_row"] = [row["per_row"] for row in measured]
            result["equivalent_acceptance"] = equivalent_acceptance(
                result["tau"], config["K"]
            )
        for measured, diagnostic, turn in zip(result["turns"], quality, turns):
            measured["quality"] = diagnostic
            for field in ("emitted_steps", "preemptions"):
                if field in turn:
                    measured[field] = turn[field]
        results.append(result)
    return results


def mtp_steps(speculative, draft_length, drafter):
    rows = [
        row
        for row in speculative["rows"]
        if (
            row["layout_name"] == LAYOUT
            and row["attention_mfu"] == EFFICIENCY
            and row["draft_tokens"] == draft_length
            and row["drafter"] == drafter
        )
    ]
    if len(rows) != 1:
        raise WorkloadValidationError(
            f"Expected one LP30 step for K={draft_length}, {drafter}"
        )
    row = rows[0]
    if any(not math.isfinite(value) or value <= 0 for value in row["step_us"].values()):
        raise WorkloadValidationError("LP30 steps must be positive and finite")
    return row


def aa_pricing(results, speculative, ratio):
    if not math.isfinite(ratio) or ratio <= 0:
        raise WorkloadValidationError("Token ratio must be positive and finite")
    indexed = {
        (json.dumps(row["config"], sort_keys=True), row["data"]): row for row in results
    }
    prices = []
    for (config_key, data), low in indexed.items():
        if low["config"]["method"] != "mtp":
            continue
        for style in STYLES:
            if data != f"aa-docs-think-10k-{style}":
                continue
            high = indexed.get((config_key, f"aa-docs-think-100k-{style}"))
            if high is None:
                continue
            tau = dict(aa_10k=low["tau"], aa_100k=high["tau"])
            for drafter in DRAFTERS:
                simulation = mtp_steps(speculative, low["draft_tokens"], drafter)
                steps = {
                    workload: simulation["step_us"][workload] for workload in CONTEXTS
                }
                prices.append(
                    dict(
                        config=low["config"],
                        style=style,
                        drafter=drafter,
                        acceptance=dict(
                            aa_10k=low["equivalent_acceptance"],
                            aa_100k=high["equivalent_acceptance"],
                        ),
                        **priced(steps, tau, ratio),
                    )
                )
    return prices


def paired(results):
    indexed = {
        (json.dumps(row["config"], sort_keys=True), row["data"]): row for row in results
    }
    pairs = []
    for (config_key, data), low in indexed.items():
        if low["config"]["method"] != "mtp":
            continue
        for style in STYLES:
            if data != f"aa-docs-think-10k-{style}":
                continue
            high = indexed.get((config_key, f"aa-docs-think-100k-{style}"))
            if high is None:
                continue
            contexts = []
            for row in (low, high):
                books = {}
                for turn in row["turns"]:
                    book = turn["meta"]["book"]
                    if book in books:
                        raise WorkloadValidationError(
                            "Document pairing expects one turn per book"
                        )
                    books[book] = turn["equivalent_acceptance"]
                contexts.append(books)
            shorter, longer = contexts
            books = sorted(shorter.keys() & longer.keys())
            differences = {book: longer[book] - shorter[book] for book in books}
            pairs.append(
                dict(
                    config=low["config"],
                    style=style,
                    paired_books=len(books),
                    differences=differences,
                    mean_difference=(
                        statistics.mean(differences.values()) if differences else None
                    ),
                    higher_at_100k=sum(value > 0 for value in differences.values()),
                    only_10k=sorted(shorter.keys() - longer.keys()),
                    only_100k=sorted(longer.keys() - shorter.keys()),
                    parts="combined thinking and answer; counters do not separate them",
                )
            )
    return pairs


def coding_speed(turn, step, draft_step, draft_length):
    """Price one turn at its observed mean context, including draft time."""
    tokens = turn["decode_tokens"]
    prompt_tokens = turn["prompt_tokens"]
    if tokens < 1 or prompt_tokens < 0:
        raise WorkloadValidationError(
            "Coding pricing needs decode tokens and a valid context"
        )
    context = prompt_tokens + tokens / 2
    rows = draft_length + 1
    elapsed_step = step(rows, context) + draft_step(rows, context)
    if not math.isfinite(elapsed_step) or elapsed_step <= 0:
        raise WorkloadValidationError(
            "Estimated coding step must be positive and finite"
        )
    if draft_length:
        counts = turn["per_row"][1]
        if len(counts) != rows or any(counts[:-1]) or counts[-1] <= 0:
            raise WorkloadValidationError(
                "MTP pricing requires K+1 verification rows per step"
            )
        steps = counts[-1]
    else:
        steps = tokens
    return tokens / (steps * elapsed_step) * 1e6


def coding_pricing(results, sweep, speculative):
    verify = context_model(verification_steps(sweep, speculative))
    prices = []
    for row in results:
        if not row["data"].startswith("speed-coding-"):
            continue
        method = row["config"]["method"]
        if method == "ngram":
            continue
        for drafter in DRAFTERS if method == "mtp" else ("plain",):
            draft_length = row["draft_tokens"]
            if method == "mtp":
                simulation = mtp_steps(speculative, draft_length, drafter)
                draft_step = context_model({draft_length + 1: simulation["draft_us"]})
            else:
                draft_step = lambda rows, context: 0.0
            speeds = [
                coding_speed(turn, verify, draft_step, draft_length)
                for turn in row["turns"]
            ]
            summary = speed_summary(speeds)
            prices.append(
                dict(
                    config=row["config"],
                    data=row["data"],
                    drafter=drafter,
                    **summary,
                    median_error=summary["median_tps"] / PUBLISHED["median"] - 1,
                    turns=[
                        dict(id=turn["id"], turn=turn["turn"], tps=speed)
                        for turn, speed in zip(row["turns"], speeds)
                    ],
                )
            )
    return prices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=1.006)
    args = parser.parse_args(argv)
    outputs = [
        json.loads(line)
        for path in args.outputs
        for line in path.read_text().splitlines()
    ]
    results = summarize(outputs)
    speculative = load("article-speculative-results.json")
    result = dict(
        layout=LAYOUT,
        attention_mfu=EFFICIENCY,
        token_ratio=args.ratio,
        results=results,
        paired=paired(results),
        aa=aa_pricing(results, speculative, args.ratio),
        coding=coding_pricing(results, load("article-sweep-results.json"), speculative),
    )
    write_json(args.summary, result)
    print(
        f"{len(outputs)} requests, {len(results)} groups, {len(result['aa'])} AA prices"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
