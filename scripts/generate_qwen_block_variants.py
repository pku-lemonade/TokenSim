#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO


DEFAULT_TRACE_NAMES = (
    "qwen_traceA_blksz_16.jsonl",
    "qwen_traceB_blksz_16.jsonl",
    "qwen_thinking_blksz_16.jsonl",
    "qwen_coder_blksz_16.jsonl",
)


class TraceConversionError(ValueError):
    pass


@dataclass(frozen=True)
class ConversionStats:
    source_path: str
    output_path: str
    source_block_size: int
    target_block_size: int
    threshold_ratio: float
    threshold_blocks: int
    record_count: int
    source_hash_count: int
    output_hash_count: int
    unique_output_hashes: int


@dataclass(frozen=True)
class ValidationStats:
    source_path: str
    variant_path: str
    source_block_size: int
    target_block_size: int
    threshold_ratio: float
    threshold_blocks: int
    record_count: int
    source_hit_tokens: int
    source_eligible_tokens: int
    source_token_hit_rate: float
    variant_hit_tokens: int
    variant_eligible_tokens: int
    variant_token_hit_rate: float
    hit_rate_delta_percentage_points: float
    ideal_threshold_hit_tokens: int
    ideal_threshold_token_hit_rate: float
    realization_gap_tokens: int
    mean_request_hit_fraction_error: float
    max_request_hit_fraction_error: float


class CompositeIdMap:
    def __init__(self) -> None:
        self._ids: dict[tuple[Any, ...], int] = {}

    def get_id(self, signature: tuple[Any, ...]) -> int:
        composite_id = self._ids.get(signature)
        if composite_id is not None:
            return composite_id
        composite_id = len(self._ids)
        self._ids[signature] = composite_id
        return composite_id

    def __len__(self) -> int:
        return len(self._ids)


class PrefixReplay:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._cache: set[bytes] = set()
        self._root = hashlib.sha256(
            f"qwen-block-variant-prefix:{block_size}".encode("utf-8")
        ).digest()

    def replay(self, input_length: int, hash_ids: list[Any]) -> int:
        full_blocks = input_length // self.block_size
        if len(hash_ids) < full_blocks:
            raise TraceConversionError(
                f"record needs {full_blocks} full blocks but has only "
                f"{len(hash_ids)} hash_ids"
            )

        parent = self._root
        keys: list[bytes] = []
        hit_blocks = 0
        checking_hits = True
        for block_hash in hash_ids[:full_blocks]:
            payload = json.dumps(
                [parent.hex(), block_hash],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            key = hashlib.sha256(payload).digest()
            keys.append(key)
            if checking_hits and key in self._cache:
                hit_blocks += 1
            else:
                checking_hits = False
            parent = key

        self._cache.update(keys)
        return hit_blocks


def threshold_block_count(group_size: int, threshold_ratio: float) -> int:
    if group_size <= 0:
        raise TraceConversionError("group_size must be positive")
    if not 0 < threshold_ratio <= 1:
        raise TraceConversionError("threshold_ratio must be in (0, 1]")
    return math.ceil(group_size * threshold_ratio)


def convert_hash_ids(
    hash_ids: list[Any],
    source_block_size: int,
    target_block_size: int,
    threshold_ratio: float,
    composite_ids: CompositeIdMap,
) -> list[int]:
    group_size = _validate_block_sizes(source_block_size, target_block_size)
    threshold_blocks = threshold_block_count(group_size, threshold_ratio)
    converted: list[int] = []

    for group_start in range(0, len(hash_ids), group_size):
        group = hash_ids[group_start : group_start + group_size]
        if len(group) == group_size:
            selected = group[:threshold_blocks]
            signature = (
                "threshold",
                source_block_size,
                target_block_size,
                threshold_blocks,
                *(_typed_hash(block_hash) for block_hash in selected),
            )
        else:
            signature = (
                "partial",
                source_block_size,
                target_block_size,
                len(group),
                *(_typed_hash(block_hash) for block_hash in group),
            )
        converted.append(composite_ids.get_id(signature))
    return converted


def convert_trace(
    source_path: Path,
    output_path: Path,
    source_block_size: int = 16,
    target_block_size: int = 128,
    threshold_ratio: float = 0.75,
    force: bool = False,
) -> ConversionStats:
    group_size = _validate_block_sizes(source_block_size, target_block_size)
    threshold_blocks = threshold_block_count(group_size, threshold_ratio)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --force to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    composite_ids = CompositeIdMap()
    record_count = 0
    source_hash_count = 0
    output_hash_count = 0
    temporary_path: Path | None = None

    try:
        with source_path.open("r", encoding="utf-8") as source_file:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                for line_number, row in _iter_jsonl(source_file, source_path):
                    input_length, hash_ids = _validate_source_row(
                        row,
                        source_path,
                        line_number,
                        source_block_size,
                    )
                    converted_hashes = convert_hash_ids(
                        hash_ids=hash_ids,
                        source_block_size=source_block_size,
                        target_block_size=target_block_size,
                        threshold_ratio=threshold_ratio,
                        composite_ids=composite_ids,
                    )
                    expected_hashes = math.ceil(input_length / target_block_size)
                    if len(converted_hashes) != expected_hashes:
                        raise TraceConversionError(
                            f"{source_path}:{line_number} produced "
                            f"{len(converted_hashes)} hashes, expected {expected_hashes}"
                        )
                    row["hash_ids"] = converted_hashes
                    json.dump(
                        row, output_file, ensure_ascii=False, separators=(",", ":")
                    )
                    output_file.write("\n")
                    record_count += 1
                    source_hash_count += len(hash_ids)
                    output_hash_count += len(converted_hashes)

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ConversionStats(
        source_path=str(source_path),
        output_path=str(output_path),
        source_block_size=source_block_size,
        target_block_size=target_block_size,
        threshold_ratio=threshold_ratio,
        threshold_blocks=threshold_blocks,
        record_count=record_count,
        source_hash_count=source_hash_count,
        output_hash_count=output_hash_count,
        unique_output_hashes=len(composite_ids),
    )


def validate_variant(
    source_path: Path,
    variant_path: Path,
    source_block_size: int = 16,
    target_block_size: int = 128,
    threshold_ratio: float = 0.75,
) -> ValidationStats:
    group_size = _validate_block_sizes(source_block_size, target_block_size)
    threshold_blocks = threshold_block_count(group_size, threshold_ratio)
    source_replay = PrefixReplay(source_block_size)
    variant_replay = PrefixReplay(target_block_size)

    record_count = 0
    source_hit_tokens = 0
    source_eligible_tokens = 0
    variant_hit_tokens = 0
    variant_eligible_tokens = 0
    ideal_threshold_hit_tokens = 0
    request_error_sum = 0.0
    request_error_count = 0
    max_request_error = 0.0

    with source_path.open("r", encoding="utf-8") as source_file:
        with variant_path.open("r", encoding="utf-8") as variant_file:
            source_rows = _iter_jsonl(source_file, source_path)
            variant_rows = _iter_jsonl(variant_file, variant_path)
            for source_item, variant_item in _zip_strict(source_rows, variant_rows):
                source_line, source_row = source_item
                variant_line, variant_row = variant_item
                if source_line != variant_line:
                    raise TraceConversionError(
                        "source and variant line numbers diverged"
                    )

                input_length, source_hashes = _validate_source_row(
                    source_row,
                    source_path,
                    source_line,
                    source_block_size,
                )
                variant_input_length, variant_hashes = _validate_source_row(
                    variant_row,
                    variant_path,
                    variant_line,
                    target_block_size,
                )
                if input_length != variant_input_length:
                    raise TraceConversionError(
                        f"{variant_path}:{variant_line} changed input_length"
                    )
                _validate_preserved_fields(
                    source_row,
                    variant_row,
                    variant_path,
                    variant_line,
                )

                source_hits = source_replay.replay(input_length, source_hashes)
                variant_hits = variant_replay.replay(input_length, variant_hashes)
                source_full_blocks = input_length // source_block_size
                variant_full_blocks = input_length // target_block_size
                ideal_hits = source_hits // group_size
                if source_hits % group_size >= threshold_blocks:
                    ideal_hits += 1
                ideal_hits = min(ideal_hits, variant_full_blocks)

                source_hit_tokens += source_hits * source_block_size
                source_eligible_tokens += source_full_blocks * source_block_size
                variant_hit_tokens += variant_hits * target_block_size
                variant_eligible_tokens += variant_full_blocks * target_block_size
                ideal_threshold_hit_tokens += ideal_hits * target_block_size

                if source_full_blocks and variant_full_blocks:
                    source_fraction = source_hits / source_full_blocks
                    variant_fraction = variant_hits / variant_full_blocks
                    request_error = abs(source_fraction - variant_fraction)
                    request_error_sum += request_error
                    request_error_count += 1
                    max_request_error = max(max_request_error, request_error)
                record_count += 1

    source_rate = _safe_rate(source_hit_tokens, source_eligible_tokens)
    variant_rate = _safe_rate(variant_hit_tokens, variant_eligible_tokens)
    ideal_rate = _safe_rate(ideal_threshold_hit_tokens, variant_eligible_tokens)
    return ValidationStats(
        source_path=str(source_path),
        variant_path=str(variant_path),
        source_block_size=source_block_size,
        target_block_size=target_block_size,
        threshold_ratio=threshold_ratio,
        threshold_blocks=threshold_blocks,
        record_count=record_count,
        source_hit_tokens=source_hit_tokens,
        source_eligible_tokens=source_eligible_tokens,
        source_token_hit_rate=source_rate,
        variant_hit_tokens=variant_hit_tokens,
        variant_eligible_tokens=variant_eligible_tokens,
        variant_token_hit_rate=variant_rate,
        hit_rate_delta_percentage_points=(variant_rate - source_rate) * 100,
        ideal_threshold_hit_tokens=ideal_threshold_hit_tokens,
        ideal_threshold_token_hit_rate=ideal_rate,
        realization_gap_tokens=variant_hit_tokens - ideal_threshold_hit_tokens,
        mean_request_hit_fraction_error=_safe_rate(
            request_error_sum,
            request_error_count,
        ),
        max_request_hit_fraction_error=max_request_error,
    )


def output_name(
    source_path: Path, source_block_size: int, target_block_size: int
) -> str:
    suffix = f"_blksz_{source_block_size}.jsonl"
    if not source_path.name.endswith(suffix):
        raise TraceConversionError(
            f"source filename must end with {suffix!r}: {source_path.name}"
        )
    return f"{source_path.name[:-len(suffix)]}_blksz_{target_block_size}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate larger-block Qwen JSONL variants using a leading-prefix "
            "threshold signature."
        )
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="source JSONL files; defaults to the four bundled Qwen traces",
    )
    parser.add_argument(
        "--source-block-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[128, 512],
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen-bailian-usagetraces-anon"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the sequential prefix-reuse validation pass",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="optional JSON path for conversion and validation statistics",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sources = args.sources or [
        Path("qwen-bailian-usagetraces-anon") / name for name in DEFAULT_TRACE_NAMES
    ]
    report: dict[str, list[dict[str, Any]]] = {
        "conversions": [],
        "validations": [],
    }

    for source_path in sources:
        for target_block_size in args.targets:
            destination = args.output_dir / output_name(
                source_path,
                args.source_block_size,
                target_block_size,
            )
            conversion = convert_trace(
                source_path=source_path,
                output_path=destination,
                source_block_size=args.source_block_size,
                target_block_size=target_block_size,
                threshold_ratio=args.threshold_ratio,
                force=args.force,
            )
            report["conversions"].append(asdict(conversion))
            print(
                f"generated {destination}: records={conversion.record_count}, "
                f"hashes={conversion.output_hash_count}, "
                f"unique={conversion.unique_output_hashes}"
            )

            if not args.skip_validation:
                validation = validate_variant(
                    source_path=source_path,
                    variant_path=destination,
                    source_block_size=args.source_block_size,
                    target_block_size=target_block_size,
                    threshold_ratio=args.threshold_ratio,
                )
                report["validations"].append(asdict(validation))
                print(
                    f"validated {destination}: source_hit_rate="
                    f"{validation.source_token_hit_rate:.6f}, variant_hit_rate="
                    f"{validation.variant_token_hit_rate:.6f}, delta_pp="
                    f"{validation.hit_rate_delta_percentage_points:+.6f}, "
                    f"realization_gap_tokens={validation.realization_gap_tokens}"
                )

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote report {args.report_path}")
    return 0


def _validate_block_sizes(source_block_size: int, target_block_size: int) -> int:
    if source_block_size <= 0:
        raise TraceConversionError("source_block_size must be positive")
    if target_block_size <= source_block_size:
        raise TraceConversionError(
            "target_block_size must be larger than source_block_size"
        )
    if target_block_size % source_block_size != 0:
        raise TraceConversionError(
            "target_block_size must be divisible by source_block_size"
        )
    return target_block_size // source_block_size


def _typed_hash(block_hash: Any) -> tuple[str, Any]:
    if isinstance(block_hash, bool) or not isinstance(block_hash, (int, str)):
        raise TraceConversionError(
            f"hash_ids values must be integers or strings, got {block_hash!r}"
        )
    return type(block_hash).__name__, block_hash


def _iter_jsonl(file: TextIO, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceConversionError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise TraceConversionError(
                f"record at {path}:{line_number} must be a JSON object"
            )
        yield line_number, row


def _validate_source_row(
    row: dict[str, Any],
    path: Path,
    line_number: int,
    block_size: int,
) -> tuple[int, list[Any]]:
    try:
        input_length = int(row["input_length"])
        hash_ids = row["hash_ids"]
    except KeyError as exc:
        raise TraceConversionError(
            f"record at {path}:{line_number} is missing {exc.args[0]!r}"
        ) from exc
    if input_length <= 0:
        raise TraceConversionError(
            f"input_length at {path}:{line_number} must be positive"
        )
    if not isinstance(hash_ids, list):
        raise TraceConversionError(f"hash_ids at {path}:{line_number} must be a list")
    expected_hashes = math.ceil(input_length / block_size)
    if len(hash_ids) != expected_hashes:
        raise TraceConversionError(
            f"{path}:{line_number} has {len(hash_ids)} hash_ids, "
            f"expected {expected_hashes} for input_length={input_length} "
            f"and block_size={block_size}"
        )
    for block_hash in hash_ids:
        _typed_hash(block_hash)
    return input_length, hash_ids


def _validate_preserved_fields(
    source_row: dict[str, Any],
    variant_row: dict[str, Any],
    variant_path: Path,
    line_number: int,
) -> None:
    source_fields = dict(source_row)
    variant_fields = dict(variant_row)
    source_fields.pop("hash_ids", None)
    variant_fields.pop("hash_ids", None)
    if source_fields != variant_fields:
        raise TraceConversionError(
            f"{variant_path}:{line_number} changed fields other than hash_ids"
        )


def _zip_strict(
    source_rows: Iterator[tuple[int, dict[str, Any]]],
    variant_rows: Iterator[tuple[int, dict[str, Any]]],
) -> Iterator[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]]:
    sentinel = object()
    while True:
        source_item = next(source_rows, sentinel)
        variant_item = next(variant_rows, sentinel)
        if source_item is sentinel and variant_item is sentinel:
            return
        if source_item is sentinel or variant_item is sentinel:
            raise TraceConversionError("source and variant record counts differ")
        yield source_item, variant_item


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
