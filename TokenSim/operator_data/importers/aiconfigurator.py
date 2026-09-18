"""Import NVIDIA AIConfigurator perf tables (Apache-2.0) into operator tables.

Two directory layouts are accepted:

* the upstream database,
  ``aic-core/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/*_perf.parquet``;
* a flat collector run directory, where ``collector/collect.py`` and
  ``network/collect_comm.sh`` leave ``*_perf.parquet`` (or, with ``--keep-csv``,
  the ``*_perf.txt`` CSV staging files) next to each other.

Latencies there are in **milliseconds**; ours are microseconds.

Mapping:

=========================  ==========================  ===============================
AIConfigurator file        our table                   notes
=========================  ==========================  ===============================
gemm_perf                  gemm                        gemm_dtype -> dtype (alias map)
context_attention_perf     context_attention           isl -> input_seq_len
generation_attention_perf  generation_attention        context_len = isl + step
moe_perf                   moe                         topk -> top_k, moe_tp/ep_size
nccl_perf / oneccl_perf    collective                  alltoall -> all_to_all, nodes=1
custom_allreduce_perf      collective (kernel=custom)  all_reduce only
moe_a2a_perf               ep_all2all                  DeepEP dispatch/combine (latency already in us)
wideep_deepep_{ll,normal}  ep_all2all                  SGLang DeepEP (two rows per source row)
wideep_*moe_perf           moe (backfill only)         wide-EP expert kernels
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
    # 4-bit weights, fp8 activations
    "w4a8_awq": "int4_a8",
    "w4afp8": "int4_a8",
    "w4a8_mxfp4_mxfp8": "mxfp4",
    "w4a8_mxfp4_fp8": "mxfp4",
    # 4-bit weights, 16-bit activations (weight-only)
    "w4a16_awq": "int4_wo",
    "w4a16_mxfp4": "mxfp4_wo",
    "float32": "fp32",
}

# comm_backend / op_name labels of the MoE all-to-all tables -> our mode names
_A2A_MODE_MAP = {
    "trtllm_deepep_ht": "deepep_high_throughput",
    "deepep_ht": "deepep_high_throughput",
    "normal": "deepep_high_throughput",
    "trtllm_deepep_ll": "deepep_low_latency",
    "deepep_ll": "deepep_low_latency",
    "ll": "deepep_low_latency",
    "deepep_v2_context": "deepep_v2_context",
    "deepep_v2_generation": "deepep_v2_generation",
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
    if key in {"default", "", "none"}:
        # vLLM's moe_a2a rows do not record the wire dtype; DeepEP moves bf16 activations.
        key = "bfloat16"
    mapped = _DTYPE_MAP.get(key)
    if mapped is None:
        try:
            return normalize_dtype(key)
        except ConfigurationError:
            return None
    return mapped


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read a perf table: ``*.parquet``, or the collector's ``*.txt`` CSV staging file."""
    if path.suffix == ".txt" or (not path.is_file() and path.with_suffix(".txt").is_file()):
        import csv

        csv_path = path if path.suffix == ".txt" else path.with_suffix(".txt")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("importing AIConfigurator data requires pyarrow") from exc
    return pq.read_table(path).to_pylist()


def _has_table(directory: Path, filename: str) -> bool:
    path = directory / filename
    return path.is_file() or path.with_suffix(".txt").is_file()


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
        versions = [p for p in versions if any(_has_table(p, f) for f in required_files)]
    if not versions:
        return None
    return sorted(versions, key=_version_key)[-1]


def _select(directory: Path, backend: str, version: str | None, required_files: tuple[str, ...] = ()) -> Path | None:
    """Locate the newest directory holding ``required_files`` (see :func:`_candidates`)."""
    candidates = _candidates(directory, backend, version, required_files)
    return candidates[0] if candidates else None


def _candidates(directory: Path, backend: str, version: str | None, required_files: tuple[str, ...] = ()) -> list[Path]:
    """Directories holding ``required_files``, newest version first.

    ``directory`` is ``<system>/<family>`` in the upstream layout. In a flat
    collector run directory the family level does not exist and the tables sit
    directly in the run root, so that root is returned when it has the files.

    AIConfigurator often ships a newer version directory that re-collected only
    a few shapes (for example the MXFP4 MoE rows in ``1.3.0rc23``) and serves the
    rest by backward fill from the previous full collection. The importer
    mirrors that: callers walk the list newest-first and keep the first row seen
    for every key.
    """
    root = directory.parent
    if not directory.is_dir() and any(_has_table(root, f) for f in required_files):
        return [root]
    backend_dir = directory / backend
    if version:
        candidate = backend_dir / version
        return [candidate] if candidate.is_dir() else []
    if not backend_dir.is_dir():
        return []
    versions = [p for p in backend_dir.iterdir() if p.is_dir()]
    if required_files:
        versions = [p for p in versions if any(_has_table(p, f) for f in required_files)]
    return sorted(versions, key=_version_key, reverse=True)


def _version_label(resolved: Path, rows: list[dict[str, Any]], fallback: str) -> str:
    """Version directory name, or the ``version`` column of a flat collector output."""
    if resolved.parent.name in {"trtllm", "vllm", "sglang", "nccl", "oneccl"}:
        return resolved.name
    for row in rows:
        value = str(row.get("version", "")).strip()
        if value:
            return value
    return fallback


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
    gpus_per_node: int = 8,
) -> OperatorDataPackage:
    """Build an :class:`OperatorDataPackage` from one AIConfigurator system directory.

    ``gpus_per_node`` is only used for SGLang's wide-EP DeepEP tables, which
    record ``node_num`` but not ``ep_size``; the EP group is assumed to span all
    GPUs of the participating nodes.
    """
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

    flat_layout = not any((system_path / family).is_dir() for family in ("gemm", "attention", "moe", "comm"))

    def source_for(family: str, resolved: Path, version_label: str, backend_label: str | None = None) -> str:
        backend_label = backend_label or backend
        if flat_layout:
            source_id = f"aiconfigurator-collector:{device_id}:{family}:{backend_label}:{version_label}"
            reference = str(resolved.resolve())
            notes = (
                "Measured on this device with the AIConfigurator collector "
                f"(commit {upstream_commit}); converted ms -> us."
            )
        else:
            source_id = f"aiconfigurator:{system_name}:{family}:{backend_label}:{version_label}"
            reference = (
                "https://github.com/ai-dynamo/aiconfigurator/tree/"
                f"{upstream_commit}/aic-core/src/aiconfigurator_core/systems/data/{system_name}/{family}/{backend_label}/{version_label}"
            )
            notes = "Measured kernel latencies collected by NVIDIA AIConfigurator (Apache-2.0); converted ms -> us."
        if source_id not in sources:
            sources[source_id] = SourceRecord(
                source_id=source_id,
                grade="A",
                method="measured" if flat_layout else "imported",
                reference=reference,
                notes=notes,
                device=device_id,
                backend=f"{backend_label}:{version_label}",
            )
        return source_id

    def keep_kernel(row: Mapping[str, Any]) -> bool:
        if kernels is None:
            return True
        kernel = str(row.get("kernel_source", "")).lower()
        return any(token in kernel for token in kernels)

    # GEMM -------------------------------------------------------------------
    rows: dict[tuple, dict[str, Any]] = {}
    for resolved in _candidates(system_path / "gemm", backend, version, ("gemm_perf.parquet",)):
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

    # Attention ----------------------------------------------------------------
    attention_candidates = _candidates(
        system_path / "attention", backend, version, ("context_attention_perf.parquet", "generation_attention_perf.parquet")
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

    # MoE ------------------------------------------------------------------------
    rows = {}
    for resolved in _candidates(system_path / "moe", backend, version, ("moe_perf.parquet",)):
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
    for wide_file in ("wideep_moe_perf.parquet", "wideep_generation_moe_perf.parquet", "wideep_context_moe_perf.parquet"):
        for resolved in _candidates(system_path / "moe", backend, version, (wide_file,)):
            if not _has_table(resolved, wide_file):
                continue
            raw_rows = _read_parquet(resolved / wide_file)
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("moe", resolved, label)
            resolved_versions.setdefault("moe_wideep", []).append(f"{wide_file}:{label}")
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

    # Expert-parallel all-to-all (DeepEP) -------------------------------------------
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

    for resolved in _candidates(system_path / "comm", backend, version, ("moe_a2a_perf.parquet",)):
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
            add_a2a(key, float(row["latency"]), source_id, f"moe_a2a:{row.get('kernel_source', '')}:{row.get('comm_backend', '')}")

    for wide_file, mode in (("wideep_deepep_ll_perf.parquet", "deepep_low_latency"), ("wideep_deepep_normal_perf.parquet", "deepep_high_throughput")):
        for resolved in _candidates(system_path / "comm", backend, version, (wide_file,)):
            if not _has_table(resolved, wide_file):
                continue
            raw_rows = _read_parquet(resolved / wide_file)
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("comm", resolved, label)
            resolved_versions.setdefault("ep_all2all", []).append(f"{wide_file.replace('_perf.parquet', '')}:{label}")
            seen_here = set()
            for row in raw_rows:
                nodes = int(row.get("node_num", 1) or 1)
                base = (
                    "bf16",
                    None,
                    mode,
                    nodes * int(gpus_per_node),
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
                        latency_us = float(row.get(f"{phase}_transmit_us", 0) or 0) + float(row.get(f"{phase}_notify_us", 0) or 0)
                    key = base[:1] + (phase,) + base[2:]
                    if key in a2a_rows and key not in seen_here:
                        continue
                    seen_here.add(key)
                    add_a2a(key, latency_us, source_id, f"{wide_file.replace('_perf.parquet', '')}:{row.get('kernel_source', '')}")
    if a2a_rows:
        tables["ep_all2all"] = list(a2a_rows.values())

    # Collectives -----------------------------------------------------------------
    collective_rows: dict[tuple, dict[str, Any]] = {}
    for resolved in _candidates(system_path / "comm", "nccl", nccl_version, ("nccl_perf.parquet",)):
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
    for resolved in _candidates(system_path / "comm", "oneccl", nccl_version, ("oneccl_perf.parquet",)):
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
        resolved = _select(system_path / "comm", backend, version, ("custom_allreduce_perf.parquet",))
        if resolved is not None and _has_table(resolved, "custom_allreduce_perf.parquet"):
            raw_rows = _read_parquet(resolved / "custom_allreduce_perf.parquet")
            label = _version_label(resolved, raw_rows, version or "unknown")
            source_id = source_for("comm", resolved, label)
            resolved_versions["collective_custom"] = [label]
            for row in raw_rows:
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
        framework_version=version or (resolved_versions.get("gemm") or [""])[0],
        sources=sources,
        notes=(
            ("Collected on this device with the AIConfigurator collector. " if flat_layout else "Imported from NVIDIA AIConfigurator measured kernel database. ")
            + f"Resolved versions: {resolved_versions}. Rows dropped (unsupported dtype/kernel): {dropped}."
        ),
        extra={
            "aiconfigurator_system": system_name,
            "aiconfigurator_layout": "collector_run" if flat_layout else "database",
            "aiconfigurator_commit": upstream_commit,
            "resolved_versions": resolved_versions,
            "dropped_rows": dropped,
            "dropped_dtypes": dropped_dtypes,
            "license": "Apache-2.0 (NVIDIA AIConfigurator); see data/operator_data/LICENSES.md",
        },
    )
    return OperatorDataPackage.from_rows(meta, tables)
