"""Import AIConfigurator attention measurements while preserving version precedence."""

from __future__ import annotations


from .context import ImportContext
from .discovery import (
    _map_dtype,
    _read_parquet,
    _has_table,
    _candidates,
    _version_label,
)


def import_attention(context: ImportContext):
    backend = context.backend
    dropped = context.dropped
    resolved_versions = context.resolved_versions
    system_path = context.system_path
    tables = context.tables
    version = context.version
    source_for = context.source_for
    keep_kernel = context.keep_kernel

    attention_candidates = _candidates(
        system_path / "attention",
        backend,
        version,
        ("context_attention_perf.parquet", "generation_attention_perf.parquet"),
    )
    for filename, table, length_field in (
        ("context_attention_perf.parquet", "context_attention", "input_seq_len"),
        ("generation_attention_perf.parquet", "generation_attention", "context_len"),
    ):
        rows = {}
        for resolved in attention_candidates:
            path = resolved / filename
            if not _has_table(resolved, filename):
                continue
            raw_rows = _read_parquet(path)
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("attention", resolved, label)
            resolved_versions.setdefault(table, []).append(label)
            seen_here = set()
            for row in raw_rows:
                attn_dtype = _map_dtype(row.get("attn_dtype"))
                kv_dtype = _map_dtype(row.get("kv_cache_dtype"))
                if attn_dtype is None or kv_dtype is None or not keep_kernel(row):
                    dropped[table] = dropped.get(table, 0) + 1
                    continue
                if int(row.get("beam_width", 1) or 1) != 1:
                    dropped[table] = dropped.get(table, 0) + 1
                    continue
                isl = int(row["isl"])
                step = int(row.get("step", 0) or 0)
                length = isl if table == "context_attention" else isl + step
                heads = int(row["num_heads"])
                kv_heads = int(row["num_key_value_heads"])
                kv_heads = heads if kv_heads == 0 else kv_heads
                key = (
                    attn_dtype,
                    kv_dtype,
                    int(row["batch_size"]),
                    length,
                    heads,
                    kv_heads,
                    int(row["head_dim"]),
                    int(row.get("window_size", 0) or 0),
                )
                if key in rows and key not in seen_here:
                    continue
                seen_here.add(key)
                latency_us = float(row["latency"]) * 1000.0
                current = rows.get(key)
                if current is None or latency_us < current["latency_us"]:
                    rows[key] = {
                        "attn_dtype": key[0],
                        "kv_cache_dtype": key[1],
                        "batch_size": key[2],
                        length_field: key[3],
                        "num_heads": key[4],
                        "num_kv_heads": key[5],
                        "head_dim": key[6],
                        "window_size": key[7],
                        "latency_us": latency_us,
                        "source_id": source_id,
                        "kernel": str(row.get("kernel_source", "")),
                    }
        if rows:
            tables[table] = list(rows.values())
