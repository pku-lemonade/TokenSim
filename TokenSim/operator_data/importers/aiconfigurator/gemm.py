"""Import AIConfigurator gemm measurements while preserving version precedence."""

from __future__ import annotations

from typing import Any
from .context import ImportContext
from .discovery import (
    _map_dtype,
    _read_parquet,
    _has_table,
    _candidates,
    _version_label,
)


def import_gemm(context: ImportContext):
    backend = context.backend
    dropped = context.dropped
    resolved_versions = context.resolved_versions
    system_path = context.system_path
    tables = context.tables
    version = context.version
    source_for = context.source_for
    keep_kernel = context.keep_kernel

    rows: dict[tuple, dict[str, Any]] = {}
    for resolved in _candidates(
        system_path / "gemm", backend, version, ("gemm_perf.parquet",)
    ):
        if not _has_table(resolved, "gemm_perf.parquet"):
            continue
        raw_rows = _read_parquet(resolved / "gemm_perf.parquet")
        label = _version_label(resolved, raw_rows, version or "unknown")
        source_id = source_for("gemm", resolved, label)
        resolved_versions.setdefault("gemm", []).append(label)
        seen_here: set[tuple] = set()
        for row in raw_rows:
            dtype = _map_dtype(row.get("gemm_dtype"))
            if dtype is None or not keep_kernel(row):
                dropped["gemm"] = dropped.get("gemm", 0) + 1
                continue
            key = (dtype, int(row["m"]), int(row["n"]), int(row["k"]))
            # Newer versions win on conflicts; older versions only backward-fill.
            if key in rows and key not in seen_here:
                continue
            seen_here.add(key)
            latency_us = float(row["latency"]) * 1000.0
            # Several kernel sources may cover the same shape; keep the fastest,
            # which is what the serving framework's autotuner selects.
            current = rows.get(key)
            if current is None or latency_us < current["latency_us"]:
                rows[key] = {
                    "dtype": dtype,
                    "m": key[1],
                    "n": key[2],
                    "k": key[3],
                    "latency_us": latency_us,
                    "source_id": source_id,
                    "kernel": str(row.get("kernel_source", "")),
                }
    if rows:
        tables["gemm"] = list(rows.values())
