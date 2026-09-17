"""Shared helpers for the profiling scripts (torch is imported lazily)."""

from __future__ import annotations

import csv
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TokenSim.operator_data.manifest import ShapeManifest  # noqa: E402

TORCH_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "int8": "int8",
    "fp8": "float8_e4m3fn",
}


def torch_dtype(name: str):
    import torch

    return getattr(torch, TORCH_DTYPES[name])


def time_kernel(fn: Callable[[], Any], *, warmup: int = 5, iters: int = 20) -> float:
    """Median wall time of ``fn`` in microseconds, measured with CUDA events."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def environment_record() -> dict[str, str]:
    import torch

    record = {
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "python": platform.python_version(),
    }
    try:
        import subprocess

        record["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,clocks.sm,clocks.mem", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        record["driver"] = "unknown"
    return record


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})
    print(f"wrote {len(rows)} rows to {path}")


def manifest_keys(manifest_path: str, table: str) -> list[dict[str, Any]]:
    manifest = ShapeManifest.load(manifest_path)
    return list(manifest.keys.get(table, []))
