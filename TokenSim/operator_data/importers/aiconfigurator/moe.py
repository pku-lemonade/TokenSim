"""Import AIConfigurator moe measurements while preserving version precedence."""

from __future__ import annotations


from .context import ImportContext
from .discovery import (
    _map_dtype,
    _read_parquet,
    _has_table,
    _version_key,
    _latest_version,
    _select,
    _candidates,
    _version_label,
    _system_gpus_per_node,
    _A2A_MODE_MAP,
    _OPERATION_MAP,
)


def import_moe(context: ImportContext):
    backend = context.backend
    dropped = context.dropped
    dropped_dtypes = context.dropped_dtypes
    resolved_versions = context.resolved_versions
    system_path = context.system_path
    tables = context.tables
    version = context.version
    source_for = context.source_for
    keep_kernel = context.keep_kernel

    rows = {}
    for resolved in _candidates(
        system_path / "moe", backend, version, ("moe_perf.parquet",)
    ):
        if not _has_table(resolved, "moe_perf.parquet"):
            continue
        raw_rows = _read_parquet(resolved / "moe_perf.parquet")
        label = _version_label(resolved, raw_rows, version or "unknown")
        source_id = source_for("moe", resolved, label)
        resolved_versions.setdefault("moe", []).append(label)
        seen_here = set()
        for row in raw_rows:
            dtype = _map_dtype(row.get("moe_dtype"))
            if dtype is None or not keep_kernel(row):
                dropped["moe"] = dropped.get("moe", 0) + 1
                dtype_label = str(row.get("moe_dtype"))
                dropped_dtypes[dtype_label] = dropped_dtypes.get(dtype_label, 0) + 1
                continue
            key = (
                dtype,
                str(row.get("distribution", "uniform")).lower(),
                int(row["num_tokens"]),
                int(row["hidden_size"]),
                int(row["inter_size"]),
                int(row["topk"]),
                int(row["num_experts"]),
                int(row.get("moe_tp_size", 1)),
                int(row.get("moe_ep_size", 1)),
            )
            if key in rows and key not in seen_here:
                continue
            seen_here.add(key)
            latency_us = float(row["latency"]) * 1000.0
            current = rows.get(key)
            if current is None or latency_us < current["latency_us"]:
                rows[key] = {
                    "dtype": key[0],
                    "distribution": key[1],
                    "num_tokens": key[2],
                    "hidden_size": key[3],
                    "inter_size": key[4],
                    "top_k": key[5],
                    "num_experts": key[6],
                    "tp_size": key[7],
                    "ep_size": key[8],
                    "latency_us": latency_us,
                    "source_id": source_id,
                    "kernel": str(row.get("kernel_source", "")),
                }
    # Wide-EP expert kernels share the moe schema (plus extra columns); they only
    # backfill keys the standard table does not have. Generation rows are
    # preferred over context rows because the runtime queries MoE per step.
    for wide_file in (
        "wideep_moe_perf.parquet",
        "wideep_generation_moe_perf.parquet",
        "wideep_context_moe_perf.parquet",
    ):
        for resolved in _candidates(
            system_path / "moe", backend, version, (wide_file,)
        ):
            if not _has_table(resolved, wide_file):
                continue
            raw_rows = _read_parquet(resolved / wide_file)
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("moe", resolved, label)
            resolved_versions.setdefault("moe_wideep", []).append(
                f"{wide_file}:{label}"
            )
            for row in raw_rows:
                dtype = _map_dtype(row.get("moe_dtype"))
                if dtype is None or not keep_kernel(row):
                    dropped["moe"] = dropped.get("moe", 0) + 1
                    continue
                key = (
                    dtype,
                    str(row.get("distribution", "uniform")).lower(),
                    int(row["num_tokens"]),
                    int(row["hidden_size"]),
                    int(row["inter_size"]),
                    int(row["topk"]),
                    int(row["num_experts"]),
                    int(row.get("moe_tp_size", 1)),
                    int(row.get("moe_ep_size", 1)),
                )
                if key in rows:
                    continue
                rows[key] = {
                    "dtype": key[0],
                    "distribution": key[1],
                    "num_tokens": key[2],
                    "hidden_size": key[3],
                    "inter_size": key[4],
                    "top_k": key[5],
                    "num_experts": key[6],
                    "tp_size": key[7],
                    "ep_size": key[8],
                    "latency_us": float(row["latency"]) * 1000.0,
                    "source_id": source_id,
                    "kernel": f"{wide_file.replace('_perf.parquet', '')}:{row.get('kernel_source', '')}",
                }
    if rows:
        tables["moe"] = list(rows.values())
