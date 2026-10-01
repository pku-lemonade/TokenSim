#!/usr/bin/env python3
"""Import verified Pro MegaMoE curves without changing other operator tables."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TokenSim.operator_data.importers.dsv4_megamoe import attach


def run(upstream):
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    evidence = ROOT / "data/reference/agentx-native-megamoe"
    originals = evidence / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream / "LICENSE", originals / "AIConfigurator-LICENSE")
    report = {
        "upstream_commit": commit,
        "source_scope": "raw data preserved; Pro, TP1, EP4/8 single-node scope imported",
        "imports": {},
    }
    for source_device, target_device in (("gb200", "b200_sxm"), ("gb300", "b300_sxm")):
        relative = f"aic-core/src/aiconfigurator_core/systems/data/{source_device}/moe/sglang/0.5.10/dsv4_megamoe_module_perf.parquet"
        source = upstream / relative
        copied = originals / f"{source_device}-dsv4_megamoe_module_perf.parquet"
        shutil.copyfile(source, copied)
        shutil.copyfile(
            source.parent / "collection_meta.yaml",
            originals / f"{source_device}-collection_meta.yaml",
        )
        raw = pq.read_table(source).to_pylist()
        rows = attach(
            ROOT / "data/operator_data" / target_device / "vllm",
            raw,
            source_hardware=source_device,
            collection_version="0.5.10",
            upstream_commit=commit,
            relative_source=relative,
        )
        report["imports"][target_device] = {
            "source_file": relative,
            "raw_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "raw_rows": len(raw),
            "selected_rows": len(rows),
            "recorded_versions": sorted(
                {r["recorded_framework_version"] for r in rows}
            ),
            "excluded_rows": {
                "reason": "EP ranks outside current single-node SXM EP4/8 scope",
                "count": len(raw) - len(rows),
            },
            "data_scope_grade": "B",
            "shared_experts_in_measurement": 0,
            "dispatch_combine_included": True,
            "gate_topk_included": False,
            "initial_cross_hardware_latency_scale": 1.0,
        }
    (evidence / "import-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream_root), indent=2))
