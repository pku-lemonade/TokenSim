"""Compare T0 greedy outputs and estimate generation time before T1.

Token differences use streamed model token IDs when present, with a separate
re-tokenized text comparison. Runtime estimates use the plain 10K decode rate and
previous API outputs' lengths; they exclude prefill and server startup.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.lpx.gpu_metrics import acceptance
from scripts.search_lpx_native import write_json


def first_difference(left, right):
    for position, (plain, draft) in enumerate(zip(left, right)):
        if plain != draft:
            return position
    return min(len(left), len(right)) if len(left) != len(right) else None


def compare(plain, mtp, tokenizer):
    plain_by_id = {row["id"]: row for row in plain}
    mtp_by_id = {row["id"]: row for row in mtp}
    if len(plain_by_id) != len(plain) or len(mtp_by_id) != len(mtp):
        raise WorkloadValidationError("Smoke comparison has duplicate request IDs")
    if plain_by_id.keys() != mtp_by_id.keys():
        raise WorkloadValidationError("Smoke comparison requires identical request IDs")
    if not plain:
        raise WorkloadValidationError("Smoke comparison needs completed requests")
    rows, residuals = [], Counter()
    for identity in sorted(plain_by_id):
        baseline, candidate = plain_by_id[identity], mtp_by_id[identity]
        for field in (
            "requests_sha256",
            "environment_sha256",
            "weight_precision",
            "kv_precision",
            "model",
            "max_tokens_override",
        ):
            if baseline["config"][field] != candidate["config"][field]:
                raise WorkloadValidationError(f"Smoke configurations differ in {field}")
        if (
            baseline["config"]["method"] != "plain"
            or candidate["config"]["method"] != "mtp"
        ):
            raise WorkloadValidationError("Expected plain and MTP smoke runs")
        if len(baseline["turns"]) != len(candidate["turns"]):
            raise WorkloadValidationError("Smoke comparison has different turn counts")
        turns = []
        for plain_turn, mtp_turn in zip(baseline["turns"], candidate["turns"]):
            if plain_turn["turn"] != mtp_turn["turn"]:
                raise WorkloadValidationError("Smoke turns are out of order")
            same_text = all(
                plain_turn[part] == mtp_turn[part] for part in ("reasoning", "content")
            )
            plain_ids = tokenizer.encode(
                plain_turn["reasoning"] + plain_turn["content"],
                add_special_tokens=False,
            ).ids
            mtp_ids = tokenizer.encode(
                mtp_turn["reasoning"] + mtp_turn["content"], add_special_tokens=False
            ).ids
            native_plain = plain_turn.get("token_ids")
            native_mtp = mtp_turn.get("token_ids")
            if (native_plain is None) != (native_mtp is None):
                raise WorkloadValidationError(
                    "Only one smoke run includes native token IDs"
                )
            native_difference = (
                first_difference(native_plain, native_mtp)
                if native_plain is not None
                else None
            )
            measured = acceptance(mtp_turn, candidate["config"]["K"])
            residuals[measured["counter_identity_residual"]] += 1
            turns.append(
                dict(
                    turn=plain_turn["turn"],
                    identical_text=same_text,
                    same_finish_reason=plain_turn["finish_reason"]
                    == mtp_turn["finish_reason"],
                    first_different_retokenized_position=first_difference(
                        plain_ids, mtp_ids
                    ),
                    native_token_ids_available=native_plain is not None,
                    first_different_token_position=native_difference,
                    plain_text_tokens=len(plain_ids),
                    mtp_text_tokens=len(mtp_ids),
                    counter_identity_residual=measured["counter_identity_residual"],
                )
            )
        rows.append(
            dict(
                id=identity,
                identical=all(
                    turn["identical_text"]
                    and turn["same_finish_reason"]
                    and turn["first_different_token_position"] is None
                    for turn in turns
                ),
                turns=turns,
            )
        )
    return dict(
        request_pairs=len(rows),
        identical_requests=sum(row["identical"] for row in rows),
        counter_identity_residuals=dict(sorted(residuals.items())),
        requests=rows,
        difference_unit="zero-based native token position; re-tokenized positions reported separately",
    )


def estimate_runtime(requests, reference, smoke):
    speed = smoke["turns"][0]["output_tps"]
    if speed is None or not math.isfinite(speed) or speed <= 0:
        raise WorkloadValidationError("Runtime estimation needs a measured decode rate")
    reference = {row["id"]: row for row in reference}
    selections = [
        ("T1 priority 1", "aa-docs", True, [3, 7]),
        ("T1 priority 1", "speed-coding", False, [3, 7]),
        ("T1 priority 2", "speed-coding", True, [3, 7]),
        ("T1 priority 2", "aa-docs", True, [1, 5]),
        ("T1 priority 3", "speed-cc16k", False, [3, 7]),
        ("T1 priority 3", "speed-coding", False, [1, 5]),
        ("T1 priority 3", "speed-coding", True, [1, 5]),
        ("T3 FP8", "aa-docs", True, [3]),
        ("T4 ngram", "aa-docs", True, [7]),
        ("T4 ngram", "speed-coding", False, [7]),
        ("T5 plain", "speed-coding", False, [0]),
    ]
    estimates = []
    for task, dataset, thinking, lengths in selections:
        selected = [
            row
            for row in requests
            if row["set"] == dataset and row["reasoning"] == thinking
        ]
        if not selected or any(row["id"] not in reference for row in selected):
            raise WorkloadValidationError("Runtime estimation lacks reference outputs")
        tokens = sum(
            max(0, turn["usage"]["completion_tokens"] - 1)
            for row in selected
            for turn in reference[row["id"]]["turns"]
        ) * len(lengths)
        maximum = sum(
            len(row["turns"]) * (row["max_tokens"] - 1) for row in selected
        ) * len(lengths)
        estimates.append(
            dict(
                task=task,
                set=dataset,
                thinking=thinking,
                draft_lengths=lengths,
                requests=len(selected) * len(lengths),
                reference_decode_tokens=tokens,
                estimated_generation_hours=tokens / speed / 3600,
                generation_hours_if_all_outputs_hit_cap=maximum / speed / 3600,
            )
        )
    estimates.append(
        dict(
            task="T6",
            requests=4,
            reference_decode_tokens=4 * 255,
            estimated_generation_hours=4 * 255 / speed / 3600,
        )
    )
    return dict(
        measured_plain_tps=speed,
        smoke_request=smoke["id"],
        method="previous API output lengths divided by plain 10K GPU rate; no assumed speculative speedup",
        excludes=[
            "prefill",
            "server startup",
            "profiling overhead",
            "metrics settling",
        ],
        analysis_only=["T2", "T5 analysis"],
        optional_skipped=["T7: no H20 operator data"],
        tasks=estimates,
    )


def read_outputs(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path)
    parser.add_argument("--mtp", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "experiments/lpx/api-gemma-outputs.jsonl",
    )
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {}
    if args.plain or args.mtp:
        if not all((args.plain, args.mtp, args.tokenizer)):
            parser.error("comparison requires --plain, --mtp and --tokenizer")
        from tokenizers import Tokenizer

        result["comparison"] = compare(
            read_outputs(args.plain),
            read_outputs(args.mtp),
            Tokenizer.from_file(str(args.tokenizer)),
        )
    if args.smoke:
        if not args.requests:
            parser.error("runtime estimation requires --requests")
        result["runtime"] = estimate_runtime(
            read_outputs(args.requests),
            read_outputs(args.reference),
            read_outputs(args.smoke)[0],
        )
    if not result:
        parser.error("select comparison or runtime estimation")
    write_json(args.summary, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
