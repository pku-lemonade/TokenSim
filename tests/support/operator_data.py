"""Small measured operator rows and source metadata for deterministic tests."""

from __future__ import annotations

from pathlib import Path
from TokenSim.operator_data.package import PackageMeta, SourceRecord


SOURCE = SourceRecord(
    "measured:test", "A", "measured", device="TestGPU", backend="test"
)

ROOT = Path(__file__).resolve().parents[2]


def _meta(**calibration) -> PackageMeta:
    return PackageMeta(
        "test-v1",
        "TestGPU",
        "test",
        sources={"measured:test": SOURCE},
        calibration=calibration,
    )


def _gemm_rows():
    return [
        {
            "dtype": "bf16",
            "m": m,
            "n": n,
            "k": k,
            "latency_us": float(m * n * k) / 1e6 + 5.0,
            "source_id": "measured:test",
        }
        for m in (1, 16, 256)
        for n in (1024, 4096)
        for k in (1024, 4096)
    ]
