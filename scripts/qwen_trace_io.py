"""Validated JSONL records shared by Qwen trace conversion and replay checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, TextIO


class TraceConversionError(ValueError):
    pass


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
