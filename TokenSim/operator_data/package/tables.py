"""Validate operator rows and read or write Parquet and CSV tables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping
from TokenSim.hardware.device import normalize_dtype
from TokenSim.operator_data.schema import (
    INTEGER_FIELDS,
    TABLE_SPECS,
    TableSpec,
    ELEMENTWISE_OPS,
    normalize_table_key_value,
)
from .metadata import PackageMeta, OperatorDataValidationError


def validate_rows(
    table_name: str,
    rows: list[Mapping[str, Any]],
    meta: PackageMeta,
    *,
    strict_sources: bool = True,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    spec: TableSpec = TABLE_SPECS[table_name]
    where = str(path) if path else table_name
    checked: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    required = spec.required_fields
    required_set = frozenset(required)
    # 0: non-negative integer, 1: dtype name, 2: lowercase string.
    kinds = tuple(
        (f, 0 if f in INTEGER_FIELDS else 1 if f in spec.dtype_fields else 2)
        for f in spec.key_fields
    )
    dtypes: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not row.keys() >= required_set:
            missing = [f for f in required if f not in row]
            raise OperatorDataValidationError(
                f"{where}: row {index} is missing fields {missing}"
            )
        clean: dict[str, Any] = {}
        for f, kind in kinds:
            value = row[f]
            if kind == 0:
                try:
                    number = int(value)
                except (TypeError, ValueError) as exc:
                    raise OperatorDataValidationError(
                        f"{where}: row {index} field {f} must be an integer"
                    ) from exc
                if number < 0:
                    raise OperatorDataValidationError(
                        f"{where}: row {index} field {f} must be non-negative"
                    )
                clean[f] = number
            elif kind == 1:
                text = str(value)
                dtype = dtypes.get(text)
                if dtype is None:
                    dtype = dtypes[text] = normalize_dtype(text)
                clean[f] = dtype
            else:
                clean[f] = str(value).strip().lower()
            try:
                clean[f] = normalize_table_key_value(table_name, f, clean[f])
            except ValueError as exc:
                raise OperatorDataValidationError(
                    f"{where}: row {index} field {f} is invalid: {exc}"
                ) from exc
        positive_fields = {
            "groups",
            "m",
            "n",
            "k",
            "batch_size",
            "query_len",
            "context_lengths",
            "input_seq_len",
            "context_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "top_k",
            "num_experts",
            "tp_size",
            "ep_size",
            "group_size",
            "message_bytes",
        }
        for f in spec.key_fields:
            if (
                table_name
                in {"batched_gemm", "mla_attention", "routed_experts", "elementwise"}
                and f in positive_fields
                and f != "context_lengths"
                and clean[f] <= 0
            ):
                raise OperatorDataValidationError(
                    f"{where}: row {index} field {f} must be positive"
                )
        if table_name == "elementwise" and clean["op_name"] not in ELEMENTWISE_OPS:
            raise OperatorDataValidationError(
                f"{where}: row {index} unsupported elementwise op_name {clean['op_name']!r}"
            )
        if table_name == "mla_attention":
            if clean["phase"] not in {"decode", "prefill"}:
                raise OperatorDataValidationError(
                    f"{where}: row {index} phase must be decode or prefill"
                )
            if clean["execution_mode"] not in {"absorbed", "expanded"}:
                raise OperatorDataValidationError(
                    f"{where}: row {index} execution_mode must be absorbed or expanded"
                )
        try:
            latency = float(row["latency_us"])
        except (TypeError, ValueError) as exc:
            raise OperatorDataValidationError(
                f"{where}: row {index} latency_us must be numeric"
            ) from exc
        if not math.isfinite(latency) or latency < 0:
            raise OperatorDataValidationError(
                f"{where}: row {index} latency_us must be finite and non-negative"
            )
        clean["latency_us"] = latency
        source_id = str(row["source_id"])
        if strict_sources and source_id not in meta.sources:
            raise OperatorDataValidationError(
                f"{where}: row {index} references unknown source_id {source_id!r}"
            )
        clean["source_id"] = source_id
        for extra_field in ("kernel", "grade"):
            if extra_field in row and row[extra_field] is not None:
                clean[extra_field] = row[extra_field]
        for metadata_field in spec.metadata_fields:
            if metadata_field in row:
                clean[metadata_field] = row[metadata_field]
        key = tuple(map(clean.__getitem__, spec.key_fields))
        if key in seen:
            raise OperatorDataValidationError(
                f"{where}: duplicate key {dict(zip(spec.key_fields, key))}"
            )
        seen.add(key)
        checked.append(clean)
    return checked


def _read_table(root: Path, filename: str) -> list[dict[str, Any]] | None:
    parquet_path = root / filename
    csv_path = parquet_path.with_suffix(".csv")
    if parquet_path.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise OperatorDataValidationError(
                "reading parquet operator tables requires pyarrow; install project requirements"
            ) from exc
        try:
            return pq.read_table(parquet_path).to_pylist()
        except Exception as exc:
            raise OperatorDataValidationError(
                f"failed to read {parquet_path}: {exc}"
            ) from exc
    if csv_path.is_file():
        import csv

        with csv_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return None


def _write_table(
    path: Path, rows: Iterable[Mapping[str, Any]], columns: tuple[str, ...] | None
) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return
    if columns is None:
        columns = tuple(dict.fromkeys(k for row in rows for k in row))
    else:
        extra = tuple(dict.fromkeys(k for row in rows for k in row if k not in columns))
        columns = columns + extra
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - fall back to CSV
        import csv

        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c) for c in columns})
        return
    table = pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows])
    pq.write_table(table, path, compression="snappy")
