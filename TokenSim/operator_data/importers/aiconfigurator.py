"""Import NVIDIA AIConfigurator perf tables (Apache-2.0) into operator tables.

AIConfigurator stores per-system, per-backend, per-version parquet files under
``aic-core/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/``.
Latencies there are in **milliseconds**; ours are microseconds.

Mapping:

=========================  ==========================  ===============================
AIConfigurator file        our table                   notes
=========================  ==========================  ===============================
gemm_perf                  gemm                        gemm_dtype -> dtype (alias map)
context_attention_perf     context_attention           isl -> input_seq_len
generation_attention_perf  generation_attention        context_len = isl + step
moe_perf                   moe                         topk -> top_k, moe_tp/ep_size
nccl_perf                  collective                  alltoall -> all_to_all, nodes=1
custom_allreduce_perf      collective (kernel=custom)  all_reduce only
=========================  ==========================  ===============================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord

_DTYPE_MAP = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "half": "fp16",
    "fp8": "fp8",
    "fp8_block": "fp8_block",
    "fp8_static": "fp8",
    "nvfp4": "nvfp4",
    "int8": "int8",
    "int8_wo": "int8_wo",
    "int4_wo": "int4_wo",
    "sq": "int8_sq",
    "w4a8_awq": "int4_wo",
    "w4a16_awq": "int4_wo",
    "float32": "fp32",
}

_OPERATION_MAP = {
    "all_reduce": "all_reduce",
    "allreduce": "all_reduce",
    "all_gather": "all_gather",
    "allgather": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "reducescatter": "reduce_scatter",
    "alltoall": "all_to_all",
    "all_to_all": "all_to_all",
}


def _map_dtype(value: Any) -> str | None:
    key = str(value).strip().lower()
    mapped = _DTYPE_MAP.get(key)
    if mapped is None:
        try:
            return normalize_dtype(key)
        except ConfigurationError:
            return None
    return mapped


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("importing AIConfigurator data requires pyarrow") from exc
    return pq.read_table(path).to_pylist()


def _version_key(path: Path) -> tuple:
    parts = []
    for piece in path.name.replace("rc", ".").replace("post", ".").split("."):
        parts.append((0, int(piece)) if piece.isdigit() else (1, piece))
    return tuple(parts)


def _latest_version(directory: Path, required_files: tuple[str, ...] = ()) -> Path | None:
    """Newest version directory that actually contains ``required_files``.

    AIConfigurator sometimes ships a newer version directory holding only a
    ``reuse.yaml`` pointer; those are skipped so the import lands on data.
    """
    if not directory.is_dir():
        return None
    versions = [p for p in directory.iterdir() if p.is_dir()]
    if required_files:
        versions = [p for p in versions if any((p / f).is_file() for f in required_files)]
    if not versions:
        return None
    return sorted(versions, key=_version_key)[-1]


def _select(directory: Path, backend: str, version: str | None, required_files: tuple[str, ...] = ()) -> Path | None:
    backend_dir = directory / backend
    if version:
        candidate = backend_dir / version
        return candidate if candidate.is_dir() else None
    return _latest_version(backend_dir, required_files)


def import_aiconfigurator_system(
    system_dir: str | Path,
    *,
    device_id: str,
    backend: str = "trtllm",
    version: str | None = None,
    nccl_version: str | None = None,
    dataset_version: str | None = None,
    include_custom_allreduce: bool = True,
    kernel_filter: Iterable[str] | None = None,
    upstream_commit: str = "unknown",
) -> OperatorDataPackage:
    """Build an :class:`OperatorDataPackage` from one AIConfigurator system directory."""
    system_path = Path(system_dir)
    if not system_path.is_dir():
        raise ConfigurationError(f"AIConfigurator system directory not found: {system_path}")
    system_name = system_path.name
    tables: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, SourceRecord] = {}
    dropped: dict[str, int] = {}
    dropped_dtypes: dict[str, int] = {}
    resolved_versions: dict[str, str] = {}
    kernels = {k.lower() for k in kernel_filter} if kernel_filter else None

    def source_for(family: str, resolved: Path) -> str:
        source_id = f"aiconfigurator:{system_name}:{family}:{resolved.parent.name}:{resolved.name}"
        if source_id not in sources:
            sources[source_id] = SourceRecord(
                source_id=source_id,
                grade="A",
                method="imported",
                reference=(
                    "https://github.com/ai-dynamo/aiconfigurator/tree/"
                    f"{upstream_commit}/aic-core/src/aiconfigurator_core/systems/data/{system_name}/{family}/{resolved.parent.name}/{resolved.name}"
                ),
                notes="Measured kernel latencies collected by NVIDIA AIConfigurator (Apache-2.0); converted ms -> us.",
                device=device_id,
                backend=f"{resolved.parent.name}:{resolved.name}",
            )
        return source_id

    def keep_kernel(row: Mapping[str, Any]) -> bool:
        if kernels is None:
            return True
        kernel = str(row.get("kernel_source", "")).lower()
        return any(token in kernel for token in kernels)

    # GEMM -------------------------------------------------------------------
    resolved = _select(system_path / "gemm", backend, version, ("gemm_perf.parquet",))
    if resolved is not None and (resolved / "gemm_perf.parquet").is_file():
        source_id = source_for("gemm", resolved)
        resolved_versions["gemm"] = resolved.name
        rows: dict[tuple, dict[str, Any]] = {}
        for row in _read_parquet(resolved / "gemm_perf.parquet"):
            dtype = _map_dtype(row.get("gemm_dtype"))
            if dtype is None or not keep_kernel(row):
                dropped["gemm"] = dropped.get("gemm", 0) + 1
                continue
            key = (dtype, int(row["m"]), int(row["n"]), int(row["k"]))
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
        tables["gemm"] = list(rows.values())

    # Attention ----------------------------------------------------------------
    resolved = _select(system_path / "attention", backend, version, ("context_attention_perf.parquet", "generation_attention_perf.parquet"))
    if resolved is not None:
        for filename, table, length_field in (
            ("context_attention_perf.parquet", "context_attention", "input_seq_len"),
            ("generation_attention_perf.parquet", "generation_attention", "context_len"),
        ):
            path = resolved / filename
            if not path.is_file():
                continue
            source_id = source_for("attention", resolved)
            resolved_versions[table] = resolved.name
            rows = {}
            for row in _read_parquet(path):
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
            tables[table] = list(rows.values())

    # MoE ------------------------------------------------------------------------
    resolved = _select(system_path / "moe", backend, version, ("moe_perf.parquet",))
    if resolved is not None and (resolved / "moe_perf.parquet").is_file():
        source_id = source_for("moe", resolved)
        resolved_versions["moe"] = resolved.name
        rows = {}
        for row in _read_parquet(resolved / "moe_perf.parquet"):
            dtype = _map_dtype(row.get("moe_dtype"))
            if dtype is None or not keep_kernel(row):
                dropped["moe"] = dropped.get("moe", 0) + 1
                label = str(row.get("moe_dtype"))
                dropped_dtypes[label] = dropped_dtypes.get(label, 0) + 1
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
        tables["moe"] = list(rows.values())

    # Collectives -----------------------------------------------------------------
    collective_rows: dict[tuple, dict[str, Any]] = {}
    nccl_dir = system_path / "comm" / "nccl"
    resolved = _select(system_path / "comm", "nccl", nccl_version, ("nccl_perf.parquet",))
    if resolved is not None and (resolved / "nccl_perf.parquet").is_file():
        source_id = source_for("comm", resolved)
        resolved_versions["collective_nccl"] = resolved.name
        for row in _read_parquet(resolved / "nccl_perf.parquet"):
            dtype = _map_dtype(row.get("nccl_dtype"))
            operation = _OPERATION_MAP.get(str(row.get("op_name", "")).lower())
            if dtype is None or operation is None:
                dropped["collective"] = dropped.get("collective", 0) + 1
                continue
            key = (dtype, operation, int(row["num_gpus"]), 1, int(row["message_size"]))
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
    if include_custom_allreduce:
        resolved = _select(system_path / "comm", backend, version, ("custom_allreduce_perf.parquet",))
        if resolved is not None and (resolved / "custom_allreduce_perf.parquet").is_file():
            source_id = source_for("comm", resolved)
            resolved_versions["collective_custom"] = resolved.name
            for row in _read_parquet(resolved / "custom_allreduce_perf.parquet"):
                dtype = _map_dtype(row.get("allreduce_dtype"))
                if dtype is None or "_eager" in str(row.get("kernel_source", "")):
                    dropped["collective"] = dropped.get("collective", 0) + 1
                    continue
                key = (dtype, "all_reduce", int(row["num_gpus"]), 1, int(row["message_size"]))
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

    if not tables:
        raise ConfigurationError(
            f"no AIConfigurator tables found under {system_path} for backend={backend!r} version={version!r}"
        )
    meta = PackageMeta(
        dataset_version=dataset_version or f"aiconfigurator-{system_name}-{backend}-{version or 'latest'}",
        device_id=device_id,
        backend=backend,
        framework_version=version or resolved_versions.get("gemm", ""),
        sources=sources,
        notes=(
            "Imported from NVIDIA AIConfigurator measured kernel database. "
            f"Resolved versions: {resolved_versions}. Rows dropped (unsupported dtype/kernel): {dropped}."
        ),
        extra={
            "aiconfigurator_system": system_name,
            "aiconfigurator_commit": upstream_commit,
            "resolved_versions": resolved_versions,
            "dropped_rows": dropped,
            "dropped_dtypes": dropped_dtypes,
            "license": "Apache-2.0 (NVIDIA AIConfigurator); see data/operator_data/LICENSES.md",
        },
    )
    return OperatorDataPackage.from_rows(meta, tables)
