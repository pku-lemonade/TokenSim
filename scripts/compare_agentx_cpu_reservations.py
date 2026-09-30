#!/usr/bin/env python3
"""Compare a fully validated admission change against its previous full runs."""

import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


def compare(directory, baseline, baseline_directory):
    current = validate(directory, verify_reservations=True)
    previous = json.loads(baseline.read_text())
    points = {}
    for sku, after in current["checkpoints"].items():
        before = previous["checkpoints"][sku]
        old_dir = baseline_directory / sku
        old_manifest = json.loads((old_dir / "run_manifest.json").read_text())
        new_manifest = json.loads((directory / sku / "run_manifest.json").read_text())
        for field in (
            "point",
            "defaults",
            "backend",
            "dataset_sha256",
            "transfer_utilization_assumption",
        ):
            assert old_manifest[field] == new_manifest[field], field
        concurrency = new_manifest["point"]["concurrency"]
        old_result = old_dir / f"agentx_c{concurrency}.json"
        assert (
            hashlib.sha256(old_result.read_bytes()).hexdigest()
            == before["result_sha256"]
        )
        changed = sorted(
            name
            for name, value in new_manifest["source_sha256"].items()
            if old_manifest["source_sha256"].get(name) != value
        )
        assert set(changed) <= {
            "TokenSim/block/hybrid_manager.py",
            "TokenSim/llm/llm_scheduler.py",
            "TokenSim/workload/agentx_native/logging.py",
        }, changed
        points[sku] = {
            "before_result_sha256": before["result_sha256"],
            "changed_runtime_files": changed,
            "after_result_sha256": after["result_sha256"],
            "before": {
                k: before[k]
                for k in (
                    "preemptions",
                    "recomputed_tokens",
                    "output_tps",
                    "gpu_hit_percent",
                    "cpu_hit_percent",
                    "ttft_p90",
                    "tpot_p90",
                    "e2e_mean",
                )
            },
            "after": {
                k: after[k]
                for k in (
                    "preemptions",
                    "recomputed_tokens",
                    "output_tps",
                    "gpu_hit_percent",
                    "cpu_hit_percent",
                    "ttft_p90",
                    "tpot_p90",
                    "e2e_mean",
                )
            },
            "output_tps_change_percent": (
                after["output_tps"] / before["output_tps"] - 1
            )
            * 100,
            "reservation_gates_per_rank": [
                w["scheduler_stats"]["async_load_reservation_gate"]
                for w in after["final_workers"]
            ],
            "max_reservations_per_rank": [
                w["scheduler_stats"]["max_inflight_prefill_reserved_slots"]
                for w in after["final_workers"]
            ],
            "phase_counts": after["phase_counts"],
        }
    return {
        "scope": "full SimpleCPU admission comparison; generic costs and one-token decode",
        "baseline": str(baseline),
        "current": str(directory),
        "checkpoints": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "data/reference/agentx-simple-cpu/pipeline-validation.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-directory",
        type=Path,
        default=ROOT / "results/agentx-dsv4-full/simple-cpu/20261001-running-quota",
    )
    args = parser.parse_args()
    result = compare(args.directory, args.baseline, args.baseline_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checkpoints"], indent=2))
