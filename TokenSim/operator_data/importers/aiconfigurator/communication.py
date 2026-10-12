"""Import AIConfigurator communication measurements while preserving version precedence."""

from __future__ import annotations

from typing import Any
from TokenSim.errors import ConfigurationError
from .context import ImportContext
from .discovery import (
    _map_dtype,
    _read_parquet,
    _has_table,
    _select,
    _candidates,
    _version_label,
    _A2A_MODE_MAP,
    _OPERATION_MAP,
)


def import_expert_all_to_all(context: ImportContext):
    backend = context.backend
    dropped = context.dropped
    gpus_per_node = context.gpus_per_node
    resolved_versions = context.resolved_versions
    system_path = context.system_path
    tables = context.tables
    version = context.version
    source_for = context.source_for

    a2a_rows: dict[tuple, dict[str, Any]] = {}

    def add_a2a(key: tuple, latency_us: float, source_id: str, kernel: str) -> None:
        if latency_us < 0:
            return
        current = a2a_rows.get(key)
        if current is None or latency_us < current["latency_us"]:
            a2a_rows[key] = {
                "dtype": key[0],
                "phase": key[1],
                "mode": key[2],
                "ep_size": key[3],
                "nodes": key[4],
                "hidden_size": key[5],
                "top_k": key[6],
                "num_experts": key[7],
                "num_tokens": key[8],
                "latency_us": latency_us,
                "source_id": source_id,
                "kernel": kernel,
            }

    for resolved in _candidates(
        system_path / "comm", backend, version, ("moe_a2a_perf.parquet",)
    ):
        if not _has_table(resolved, "moe_a2a_perf.parquet"):
            continue
        raw_rows = _read_parquet(resolved / "moe_a2a_perf.parquet")
        label = _version_label(resolved, raw_rows, version or "unknown")
        source_id = source_for("comm", resolved, label)
        resolved_versions.setdefault("ep_all2all", []).append(f"moe_a2a:{label}")
        seen_here = set()
        for row in raw_rows:
            dtype = _map_dtype(row.get("comm_dtype"))
            mode = _A2A_MODE_MAP.get(str(row.get("comm_backend", "")).lower())
            phase = str(row.get("phase", "")).lower()
            if dtype is None or mode is None or phase not in ("dispatch", "combine"):
                dropped["ep_all2all"] = dropped.get("ep_all2all", 0) + 1
                continue
            key = (
                dtype,
                phase,
                mode,
                int(row["ep_size"]),
                int(row.get("node_num", 1) or 1),
                int(row["hidden_size"]),
                int(row["topk"]),
                int(row["num_experts"]),
                int(row["num_tokens"]),
            )
            if key in a2a_rows and key not in seen_here:
                continue
            seen_here.add(key)
            # Unlike the other AIConfigurator tables this one is already in microseconds
            # (``latency == transmit_us + notify_us``).
            add_a2a(
                key,
                float(row["latency"]),
                source_id,
                f"moe_a2a:{row.get('kernel_source', '')}:{row.get('comm_backend', '')}",
            )

    for wide_file, mode in (
        ("wideep_deepep_ll_perf.parquet", "deepep_low_latency"),
        ("wideep_deepep_normal_perf.parquet", "deepep_high_throughput"),
    ):
        for resolved in _candidates(
            system_path / "comm", backend, version, (wide_file,)
        ):
            if not _has_table(resolved, wide_file):
                continue
            if gpus_per_node is None:
                raise ConfigurationError(
                    f"{resolved / wide_file} records node counts, and no system "
                    "description declares num_gpus_per_node; pass --gpus-per-node"
                )
            raw_rows = _read_parquet(resolved / wide_file)
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("comm", resolved, label)
            resolved_versions.setdefault("ep_all2all", []).append(
                f"{wide_file.replace('_perf.parquet', '')}:{label}"
            )
            seen_here = set()
            for row in raw_rows:
                nodes = int(row.get("node_num", 1) or 1)
                base = (
                    "bf16",
                    None,
                    mode,
                    nodes * gpus_per_node,
                    nodes,
                    int(row["hidden_size"]),
                    int(row["num_topk"]),
                    int(row["num_experts"]),
                    int(row["num_token"]),
                )
                for phase in ("dispatch", "combine"):
                    if "dispatch_avg_t_us" in row:
                        latency_us = float(row[f"{phase}_avg_t_us"])
                    else:
                        latency_us = float(
                            row.get(f"{phase}_transmit_us", 0) or 0
                        ) + float(row.get(f"{phase}_notify_us", 0) or 0)
                    key = base[:1] + (phase,) + base[2:]
                    if key in a2a_rows and key not in seen_here:
                        continue
                    seen_here.add(key)
                    add_a2a(
                        key,
                        latency_us,
                        source_id,
                        f"{wide_file.replace('_perf.parquet', '')}:{row.get('kernel_source', '')}",
                    )
    if a2a_rows:
        tables["ep_all2all"] = list(a2a_rows.values())


def import_collectives(context: ImportContext):
    backend = context.backend
    dropped = context.dropped
    include_custom_allreduce = context.include_custom_allreduce
    nccl_version = context.nccl_version
    resolved_versions = context.resolved_versions
    system_path = context.system_path
    tables = context.tables
    version = context.version
    source_for = context.source_for

    collective_rows: dict[tuple, dict[str, Any]] = {}
    for resolved in _candidates(
        system_path / "comm", "nccl", nccl_version, ("nccl_perf.parquet",)
    ):
        if not _has_table(resolved, "nccl_perf.parquet"):
            continue
        raw_rows = _read_parquet(resolved / "nccl_perf.parquet")
        label = _version_label(resolved, raw_rows, nccl_version or "unknown")
        source_id = source_for("comm", resolved, label, backend_label="nccl")
        resolved_versions.setdefault("collective_nccl", []).append(label)
        for row in raw_rows:
            dtype = _map_dtype(row.get("nccl_dtype"))
            operation = _OPERATION_MAP.get(str(row.get("op_name", "")).lower())
            if dtype is None or operation is None:
                dropped["collective"] = dropped.get("collective", 0) + 1
                continue
            key = (dtype, operation, int(row["num_gpus"]), 1, int(row["message_size"]))
            if key in collective_rows:
                continue  # newer NCCL version already provided this point
            collective_rows[key] = {
                "dtype": key[0],
                "operation": key[1],
                "group_size": key[2],
                "nodes": key[3],
                "message_bytes": key[4],
                "latency_us": float(row["latency"]) * 1000.0,
                "source_id": source_id,
                "kernel": "nccl",
            }
    for resolved in _candidates(
        system_path / "comm", "oneccl", nccl_version, ("oneccl_perf.parquet",)
    ):
        # Intel XPU systems ship oneCCL curves with the NCCL column layout.
        if not _has_table(resolved, "oneccl_perf.parquet"):
            continue
        raw_rows = _read_parquet(resolved / "oneccl_perf.parquet")
        label = _version_label(resolved, raw_rows, nccl_version or "unknown")
        source_id = source_for("comm", resolved, label, backend_label="oneccl")
        resolved_versions.setdefault("collective_oneccl", []).append(label)
        for row in raw_rows:
            dtype = _map_dtype(row.get("nccl_dtype"))
            operation = _OPERATION_MAP.get(str(row.get("op_name", "")).lower())
            if dtype is None or operation is None:
                dropped["collective"] = dropped.get("collective", 0) + 1
                continue
            key = (dtype, operation, int(row["num_gpus"]), 1, int(row["message_size"]))
            if key in collective_rows:
                continue
            collective_rows[key] = {
                "dtype": key[0],
                "operation": key[1],
                "group_size": key[2],
                "nodes": key[3],
                "message_bytes": key[4],
                "latency_us": float(row["latency"]) * 1000.0,
                "source_id": source_id,
                "kernel": "oneccl",
            }
    if include_custom_allreduce:
        resolved = _select(
            system_path / "comm", backend, version, ("custom_allreduce_perf.parquet",)
        )
        if resolved is not None and _has_table(
            resolved, "custom_allreduce_perf.parquet"
        ):
            raw_rows = _read_parquet(resolved / "custom_allreduce_perf.parquet")
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("comm", resolved, label)
            resolved_versions["collective_custom"] = [label]
            for row in raw_rows:
                dtype = _map_dtype(row.get("allreduce_dtype"))
                if dtype is None or "_eager" in str(row.get("kernel_source", "")):
                    dropped["collective"] = dropped.get("collective", 0) + 1
                    continue
                key = (
                    dtype,
                    "all_reduce",
                    int(row["num_gpus"]),
                    1,
                    int(row["message_size"]),
                )
                latency_us = float(row["latency"]) * 1000.0
                current = collective_rows.get(key)
                if current is None or latency_us < current["latency_us"]:
                    collective_rows[key] = {
                        "dtype": key[0],
                        "operation": "all_reduce",
                        "group_size": key[2],
                        "nodes": 1,
                        "message_bytes": key[4],
                        "latency_us": latency_us,
                        "source_id": source_id,
                        "kernel": f"custom_allreduce:{row.get('kernel_source', '')}",
                    }
    if collective_rows:
        tables["collective"] = list(collective_rows.values())
