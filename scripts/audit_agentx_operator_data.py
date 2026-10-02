#!/usr/bin/env python3
"""Audit native operator tables without silently deleting unusual rows.

The audit distinguishes data that cannot be queried (null/non-finite/non-positive
latency and missing key fields) from rows that are merely suspicious (duplicate
keys or a latency decrease along one interpolation axis).  Both classes remain
in the source parquet files; the output is evidence for the cleaning policy,
not a filtered replacement dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TokenSim.operator_data.schema import TABLE_SPECS


DEFAULT_DEVICES = ("b200_sxm", "b300_sxm", "gb200", "gb300")


def _audit_table(path: Path, table_name: str) -> dict:
    spec = TABLE_SPECS[table_name]
    parquet = pq.ParquetFile(path)
    columns = list(parquet.schema_arrow.names)
    required = list(spec.required_fields)
    missing_columns = [field for field in required if field not in columns]
    result = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": parquet.metadata.num_rows,
        "columns": columns,
        "required_columns": required,
        "missing_columns": missing_columns,
        "invalid_rows": 0,
        "invalid_reasons": {},
        "invalid_examples": [],
        "duplicate_key_groups": 0,
        "duplicate_rows": 0,
        "duplicate_examples": [],
        "nonmonotonic_axis_groups": {},
        "policy": {
            "invalid_rows_removed": False,
            "duplicate_rows_removed": False,
            "nonmonotonic_rows_removed": False,
            "notes": "Suspicious rows remain in the original parquet and are reported only.",
        },
    }
    if missing_columns:
        return result

    frame = pq.read_table(path, columns=required).to_pandas()
    bad_mask = frame["latency_us"].isna() | ~frame["latency_us"].map(
        lambda value: isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
    ) | (frame["latency_us"] <= 0)
    for field in spec.key_fields:
        bad_mask |= frame[field].isna()
    result["invalid_rows"] = int(bad_mask.sum())
    for reason, mask in (
        ("missing_or_nonfinite_latency", frame["latency_us"].isna() | ~frame["latency_us"].map(
            lambda value: isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
        )),
        ("nonpositive_latency", frame["latency_us"].notna() & (frame["latency_us"] <= 0)),
    ):
        count = int(mask.sum())
        if count:
            result["invalid_reasons"][reason] = count
    for index in frame.index[bad_mask][:5]:
        result["invalid_examples"].append({"row": int(index), **frame.loc[index].to_dict()})

    key_frame = frame[list(spec.key_fields)]
    duplicated = key_frame.duplicated(keep=False)
    if duplicated.any():
        groups = frame.loc[duplicated].groupby(list(spec.key_fields), dropna=False, sort=False)
        result["duplicate_key_groups"] = int(groups.ngroups)
        result["duplicate_rows"] = int(duplicated.sum())
        for key, group in list(groups)[:5]:
            if not isinstance(key, tuple):
                key = (key,)
            result["duplicate_examples"].append(
                {"key": dict(zip(spec.key_fields, key)), "latencies_us": group["latency_us"].tolist()[:10]}
            )

    # A decrease along an interpolation axis is evidence to retain and inspect,
    # not a reason to delete a measurement. Only compare rows where all other
    # axes and discrete fields are identical.
    for axis in spec.axis_names:
        others = [field for field in spec.key_fields if field != axis]
        decreases = 0
        groups = frame.groupby(others, dropna=False, sort=False)
        for _, group in groups:
            ordered = group.sort_values(axis)[[axis, "latency_us"]]
            values = ordered["latency_us"].tolist()
            decreases += sum(current < previous for previous, current in zip(values, values[1:]))
        if decreases:
            result["nonmonotonic_axis_groups"][axis] = int(decreases)
    return result


def audit(*, devices=DEFAULT_DEVICES, backend="vllm") -> dict:
    tables = []
    for device in devices:
        directory = ROOT / "data/operator_data" / device / backend
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*_perf.parquet")):
            table_name = path.name.removesuffix("_perf.parquet")
            if table_name not in TABLE_SPECS:
                continue
            tables.append({"device": device, "table": table_name, **_audit_table(path, table_name)})
    return {
        "schema_version": 1,
        "scope": "native operator parquet cleaning audit",
        "devices": list(devices),
        "backend": backend,
        "table_count": len(tables),
        "tables": tables,
        "cleaning_policy": "Preserve raw rows; exclude only confirmed invalid values at query/import validation; retain duplicate and non-monotonic rows with provenance.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", action="append", dest="devices")
    args = parser.parse_args()
    report = audit(devices=tuple(args.devices or DEFAULT_DEVICES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(args.output)
