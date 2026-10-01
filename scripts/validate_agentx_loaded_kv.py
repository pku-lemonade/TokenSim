#!/usr/bin/env python3
"""Audit ready-KV admission and compare full runs with the first-source baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


def audit(directory, baseline, *, hardware=("b200", "b300")):
    current = validate(directory, hardware=hardware, verify_reservations=True)
    for sku, point in current["checkpoints"].items():
        old_dir, new_dir = baseline / sku, directory / sku
        old_manifest = json.loads((old_dir / "run_manifest.json").read_text())
        new_manifest = json.loads((new_dir / "run_manifest.json").read_text())
        for field in (
            "point",
            "defaults",
            "backend",
            "dataset_sha256",
            "transfer_utilization_assumption",
        ):
            assert old_manifest[field] == new_manifest[field], field
        changed = sorted(
            name
            for name, value in new_manifest["source_sha256"].items()
            if old_manifest["source_sha256"].get(name) != value
        )
        assert set(changed) <= {
            "TokenSim/llm/llm_scheduler.py",
            "TokenSim/llm/llm_engine.py",
            "TokenSim/kv_transfer/connectors/simple_cpu.py",
            "TokenSim/workload/agentx_native/logging.py",
        }, changed
        c = new_manifest["point"]["concurrency"]
        before_path = old_dir / f"agentx_c{c}.json"
        before = json.loads(before_path.read_text())
        old_metrics = before["agentx_metrics"]
        after = json.loads((new_dir / f"agentx_c{c}.json").read_text())
        metrics = after["agentx_metrics"]
        records = [
            json.loads(line)
            for line in (new_dir / "native_replay/requests.jsonl")
            .read_text()
            .splitlines()
        ]
        profile = [r for r in records if r["phase"] == "profiling"]
        assert all(w["loaded_waiting"] == 0 for w in point["final_workers"])
        point["comparison"] = {
            "baseline": str(old_dir),
            "baseline_result_sha256": hashlib.sha256(
                before_path.read_bytes()
            ).hexdigest(),
            "changed_runtime_files": changed,
            "before": {
                "preemptions": before["preemption_count"],
                "readmissions": old_metrics["readmission_count"],
                "reloaded_cpu_tokens": old_metrics["reloaded_cpu_tokens"],
                "output_tps": old_metrics["output_token_throughput_tps"],
                "gpu_hit_percent": old_metrics["gpu_cache_hit_rate"] * 100,
                "cpu_hit_percent": old_metrics["cpu_cache_hit_rate"] * 100,
            },
            "after": {
                "preemptions": after["preemption_count"],
                "readmissions": metrics["readmission_count"],
                "reloaded_cpu_tokens": metrics["reloaded_cpu_tokens"],
                "output_tps": metrics["output_token_throughput_tps"],
                "gpu_hit_percent": metrics["gpu_cache_hit_rate"] * 100,
                "cpu_hit_percent": metrics["cpu_cache_hit_rate"] * 100,
            },
            "max_admissions_of_any_profile_request": max(
                r["prefill_admission_count"] for r in profile
            ),
            "loaded_capacity_gates": [
                w["scheduler_stats"]["loaded_kv_capacity_gate"]
                for w in point["final_workers"]
            ],
            "loaded_resumptions": [
                w["scheduler_stats"]["loaded_kv_resumptions"]
                for w in point["final_workers"]
            ],
        }
    return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / "results/agentx-dsv4-full/simple-cpu/20261001-first-admission-stats",
    )
    parser.add_argument(
        "--hardware", choices=("b200", "b300"), nargs="+", default=("b200", "b300")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.baseline, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {sku: p["comparison"] for sku, p in result["checkpoints"].items()}, indent=2
        )
    )
