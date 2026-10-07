#!/usr/bin/env python3
"""Strict full-window audit of native attention SOL extrapolation."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-decode-batch/20261007-full-batch-v2")
BASELINE_EVIDENCE = Path("data/reference/agentx-decode-batch/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/latency/deepseek_v4_attention.py",
    "TokenSim/latency/deepseek_v4_attention_sol.py",
    "TokenSim/operator_data/lookup.py",
}


def audit(root, *, hardware=("b200", "b300")):
    result = audit_batches(root, hardware=hardware)
    previous = json.loads(BASELINE_EVIDENCE.read_text())["checkpoints"]
    for sku, concurrency in (("b200", 196), ("b300", 384)):
        if sku not in hardware:
            continue
        relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
        old, new = BASELINE / relative, Path(root) / relative
        old_result = old / f"agentx_c{concurrency}.json"
        assert hashlib.sha256(old_result.read_bytes()).hexdigest() == previous[sku]["result_sha256"]
        old_manifest = json.loads((old / "run_manifest.json").read_text())
        new_manifest = json.loads((new / "run_manifest.json").read_text())
        for field in ("point", "defaults", "backend", "simulation_seed", "dataset_sha256", "transfer_utilization_assumption"):
            assert old_manifest[field] == new_manifest[field], field
        before, after = old_manifest["source_sha256"], new_manifest["source_sha256"]
        changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
        assert changed == SOURCE_CHANGES, changed
        current = json.loads((new / f"agentx_c{concurrency}.json").read_text())
        old_data = json.loads(old_result.read_text())
        # This result field also carries measured pressure counters. They
        # change with the closed-loop request set even when capacity is fixed.
        dynamic_fields = {"peak_used_slots", "evictions"}
        pressure = []
        assert len(current["worker_cache_budgets"]) == len(old_data["worker_cache_budgets"])
        for old_worker, new_worker in zip(old_data["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert old_worker.keys() == new_worker.keys()
            assert {k: v for k, v in old_worker.items() if k not in dynamic_fields} == {
                k: v for k, v in new_worker.items() if k not in dynamic_fields
            }
            pressure.append({"worker_id": new_worker["worker_id"], "baseline": {
                k: old_worker[k] for k in dynamic_fields
            }, "current": {k: new_worker[k] for k in dynamic_fields}})
        provenance = result["checkpoints"][sku]["attention_provenance"]
        assert "FLOP/byte SOL growth" in provenance["extrapolation"]
        sol = provenance["extrapolation_sol"]
        assert not sol["source_to_reference_layout_correction"]
        assert sol["config"]["kv_state_bytes"] == 584
        assert sol["config"]["indexer_state_bytes"] == 132
        assert sol["config"]["compressor_dtype"] == "bf16"
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "result_sha256": previous[sku]["result_sha256"],
            "output_tps": previous[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / previous[sku]["output_tps"] - 1),
            "source_changes": sorted(changed), "configuration_and_capacity_unchanged": True,
            "dynamic_cache_pressure": pressure,
        }
    result["scope"] = "DSV4 physical SOL boundary-efficiency extrapolation; multi-axis binding; native scale 1.0"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.batch_root, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
