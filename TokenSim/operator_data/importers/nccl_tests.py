"""Parse ``nccl-tests`` (``all_reduce_perf`` etc.) output into collective rows.

Example line (out-of-place columns first)::

    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
        33554432       8388608     float     sum      -1   491.6   68.26  119.46      0    489.9   68.49  119.87      0
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from TokenSim.errors import ConfigurationError

_TYPE_MAP = {
    "float": "fp32",
    "float32": "fp32",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "int8": "int8",
    "uint8": "int8",
    "int32": "fp32",
    "fp8e4m3": "fp8",
    "fp8e5m2": "fp8",
}

_OPERATION_FROM_BINARY = {
    "all_reduce_perf": "all_reduce",
    "all_gather_perf": "all_gather",
    "reduce_scatter_perf": "reduce_scatter",
    "alltoall_perf": "all_to_all",
    "sendrecv_perf": "send_recv",
    "broadcast_perf": "broadcast",
}


def parse_nccl_tests_output(
    text_or_path: str | Path,
    *,
    operation: str,
    group_size: int,
    nodes: int,
    source_id: str,
    use_in_place: bool = False,
) -> list[dict[str, Any]]:
    path = Path(text_or_path) if isinstance(text_or_path, Path) or (isinstance(text_or_path, str) and "\n" not in text_or_path and Path(text_or_path).is_file()) else None
    text = path.read_text(encoding="utf-8", errors="replace") if path else str(text_or_path)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) < 9 or not parts[0].isdigit():
            continue
        size_bytes = int(parts[0])
        dtype = _TYPE_MAP.get(parts[2].lower())
        if dtype is None:
            continue
        # columns: size count type redop root time algbw busbw wrong [time algbw busbw wrong]
        try:
            time_us = float(parts[9] if use_in_place and len(parts) >= 13 else parts[5])
        except ValueError:
            continue
        rows.append(
            {
                "dtype": dtype,
                "operation": operation,
                "group_size": int(group_size),
                "nodes": int(nodes),
                "message_bytes": size_bytes,
                "latency_us": time_us,
                "source_id": source_id,
                "kernel": "nccl-tests",
            }
        )
    if not rows:
        raise ConfigurationError("no nccl-tests result rows found in input")
    return rows


def operation_from_binary_name(name: str) -> str:
    try:
        return _OPERATION_FROM_BINARY[Path(name).name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown nccl-tests binary {name!r}") from exc
