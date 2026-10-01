#!/usr/bin/env python3
"""Import the measured DSV4 CSA/HCA attention modules for both target SKUs."""

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
from TokenSim.operator_data.importers.dsv4_attention import attach


FILES = {
    ("csa", "context"): "dsv4_csa_context_module_perf.parquet",
    ("csa", "generation"): "dsv4_csa_generation_module_perf.parquet",
    ("hca", "context"): "dsv4_hca_context_module_perf.parquet",
    ("hca", "generation"): "dsv4_hca_generation_module_perf.parquet",
}


def run(upstream):
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    evidence = ROOT / "data/reference/agentx-native-attention"
    originals = evidence / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream / "LICENSE", originals / "AIConfigurator-LICENSE")
    report = {
        "upstream_commit": commit,
        "source_scope": "DeepSeek-V4-Pro CSA/HCA complete attention modules, vLLM 0.24.0",
        "imports": {},
    }
    for source_device, target_device in (("b200_sxm", "b200_sxm"), ("b300_sxm", "b300_sxm")):
        base = (
            upstream
            / "aic-core/src/aiconfigurator_core/systems/data"
            / source_device
            / "sparse_attention/vllm/0.24.0"
        )
        source_meta = base / "collection_meta.yaml"
        shutil.copyfile(source_meta, originals / f"{source_device}-collection_meta.yaml")
        source_version = yaml.safe_load(source_meta.read_text())["runtime"]["version"]
        grouped = []
        import_report = {
            "source_hardware": source_device,
            "source_version": source_version,
            "tables": {},
            "selected_rows": 0,
            "raw_rows": 0,
            "data_scope_grade": "B",
            "kernel_source": "FLASHMLA_SPARSE_DSV4",
            "reference_backend": "FLASHINFER_MLA_SPARSE_DSV4",
        }
        for (layer_kind, phase), filename in FILES.items():
            source = base / filename
            shutil.copyfile(source, originals / f"{source_device}-{filename}")
            raw = pq.read_table(source).to_pylist()
            grouped.append((layer_kind, phase, raw))
            selected = sum(
                1
                for row in raw
                if row.get("model") == "deepseek-ai/DeepSeek-V4-Pro"
            )
            import_report["tables"][filename] = {
                "raw_rows": len(raw),
                "selected_pro_rows": selected,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            import_report["raw_rows"] += len(raw)
            import_report["selected_rows"] += selected
        rows = attach(
            ROOT / "data/operator_data" / target_device / "vllm",
            grouped,
            source_hardware=source_device,
            source_version=source_version,
            collection_version="0.24.0",
            upstream_commit=commit,
            relative_source=(
                f"aic-core/src/aiconfigurator_core/systems/data/{source_device}/"
                "sparse_attention/vllm/0.24.0/dsv4_csa_context_module_perf.parquet"
            ),
        )
        import_report["normalized_rows"] = len(rows)
        report["imports"][target_device] = import_report
    (evidence / "import-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream_root), indent=2))
