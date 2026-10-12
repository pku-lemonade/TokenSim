"""Discover AIConfigurator versions and normalize upstream dtype and operation labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype


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
    if path.suffix == ".txt" or (
        not path.is_file() and path.with_suffix(".txt").is_file()
    ):
        import csv

        csv_path = path if path.suffix == ".txt" else path.with_suffix(".txt")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "importing AIConfigurator data requires pyarrow"
        ) from exc
    return pq.read_table(path).to_pylist()


def _has_table(directory: Path, filename: str) -> bool:
    path = directory / filename
    return path.is_file() or path.with_suffix(".txt").is_file()


def _version_key(path: Path) -> tuple:
    parts = []
    for piece in path.name.replace("rc", ".").replace("post", ".").split("."):
        parts.append((0, int(piece)) if piece.isdigit() else (1, piece))
    return tuple(parts)


def _latest_version(
    directory: Path, required_files: tuple[str, ...] = ()
) -> Path | None:
    """Newest version directory that actually contains ``required_files``.

    AIConfigurator sometimes ships a newer version directory holding only a
    ``reuse.yaml`` pointer; those are skipped so the import lands on data.
    """
    if not directory.is_dir():
        return None
    versions = [p for p in directory.iterdir() if p.is_dir()]
    if required_files:
        versions = [
            p for p in versions if any(_has_table(p, f) for f in required_files)
        ]
    if not versions:
        return None
    return sorted(versions, key=_version_key)[-1]


def _select(
    directory: Path,
    backend: str,
    version: str | None,
    required_files: tuple[str, ...] = (),
) -> Path | None:
    """Locate the newest directory holding ``required_files`` (see :func:`_candidates`)."""
    candidates = _candidates(directory, backend, version, required_files)
    return candidates[0] if candidates else None


def _candidates(
    directory: Path,
    backend: str,
    version: str | None,
    required_files: tuple[str, ...] = (),
) -> list[Path]:
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
        versions = [
            p for p in versions if any(_has_table(p, f) for f in required_files)
        ]
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


def _system_gpus_per_node(system_path: Path) -> int | None:
    """``node.num_gpus_per_node`` from the upstream ``systems/<name>.yaml``.

    The description is the sibling of ``systems/data`` and names the data
    directory it describes; collector run directories have none.
    """
    import yaml

    path = system_path.parent.parent / f"{system_path.name}.yaml"
    if not path.is_file():
        return None
    with path.open() as f:
        config = yaml.safe_load(f) or {}
    data_dir = config.get("data_dir")
    if not data_dir or (path.parent / data_dir).resolve() != system_path.resolve():
        return None
    value = (config.get("node") or {}).get("num_gpus_per_node")
    return int(value) if value else None
