#!/usr/bin/env python3
"""Import measured vLLM DSV4 mHC pre/post modules for B200 and B300."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TokenSim.operator_data.importers.mhc import attach


def run(upstream):
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    evidence = ROOT / "data/reference/agentx-native-mhc"
    originals = evidence / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream / "LICENSE", originals / "AIConfigurator-LICENSE")
    report = {"upstream_commit": commit, "imports": {}}
    for source_device, target_device in (("b200_sxm", "b200_sxm"), ("b300_sxm", "b300_sxm")):
        base = upstream / "aic-core/src/aiconfigurator_core/systems/data" / source_device / "mhc/vllm/0.24.0"
        meta_path = base / "collection_meta.yaml"
        shutil.copyfile(meta_path, originals / f"{source_device}-collection_meta.yaml")
        source_version = yaml.safe_load(meta_path.read_text())["runtime"]["version"]
        source = base / "mhc_module_perf.parquet"
        shutil.copyfile(source, originals / f"{source_device}-mhc_module_perf.parquet")
        raw = pq.read_table(source).to_pylist()
        rows = attach(
            ROOT / "data/operator_data" / target_device / "vllm",
            raw,
            source_hardware=source_device,
            source_version=source_version,
            collection_version="0.24.0",
            upstream_commit=commit,
            relative_source=f"aic-core/src/aiconfigurator_core/systems/data/{source_device}/mhc/vllm/0.24.0/mhc_module_perf.parquet",
        )
        report["imports"][target_device] = {
            "raw_rows": len(raw),
            "normalized_rows": len(rows),
            "pro_rows": len(rows),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_version": source_version,
            "scope_grade": "B",
            "num_sites": 2,
        }
    (evidence / "import-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream_root), indent=2))
