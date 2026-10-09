#!/usr/bin/env python3
"""Audit native mHC residual ownership and preserved native MoE summaries."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-full-mtp/20261007-three-passes")
EVIDENCE = Path("data/reference/agentx-full-mtp/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/config/psla_config.py", "TokenSim/latency/operator_table.py"}


def audit(root, *, hardware=("b200", "b300")):
    result = audit_batches(root, hardware=hardware)
    previous = json.loads(EVIDENCE.read_text())["checkpoints"]
    for sku, concurrency in (("b200", 196), ("b300", 384)):
        if sku not in hardware:
            continue
        relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
        old, new = BASELINE / relative, Path(root) / relative
        before = json.loads((old / "run_manifest.json").read_text())
        after = json.loads((new / "run_manifest.json").read_text())
        for key in ("point", "defaults", "backend", "simulation_seed", "dataset_sha256", "transfer_utilization_assumption"):
            assert before[key] == after[key], key
        a, b = before["source_sha256"], after["source_sha256"]
        changes = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        assert changes == SOURCE_CHANGES, changes
        old_path = old / f"agentx_c{concurrency}.json"
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == previous[sku]["result_sha256"]
        old_result, current = json.loads(old_path.read_text()), json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(old_result["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(old_result["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert first.keys() == second.keys()
            assert {key: value for key, value in first.items() if key not in ("peak_used_slots", "evictions")} == {
                key: value for key, value in second.items() if key not in ("peak_used_slots", "evictions")
            }
        assert current["native_megamoe_routed_seconds"] > 0
        assert current["native_megamoe_shared_seconds"] > 0
        assert current["native_megamoe_counts"] and current["native_megamoe_max_rank_rows"] > 0
        steps = current["operator_step_count"]
        counts = current["speculative_counts"]
        assert counts["target_steps"] == counts["initial_passes"] == steps
        assert counts["decode_passes"] == 2 * steps and counts["passes"] == 3 * steps
        assert current["native_megamoe_counts"]["rank_estimates"] == 4 * steps
        assert current["native_megamoe_counts"]["graph_pass"] + current["native_megamoe_counts"]["eager_pass"] == 4 * steps
        total = current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"]
        assert abs(total - current["operator_component_seconds"]["moe"]) < 1e-6
        draft = current["speculative_component_seconds"]["moe"]
        assert abs(current["moe_compute_latency"] + current["moe_straggler_latency"] + draft - total) < 1e-6
        # The cumulative missing-shape ledger includes every analytical op
        # requested in warmup/profile. Native post now owns both residuals.
        assert current["operator_missing_shape_groups"] == len(current["operator_missing_shapes"])
        assert not any(row["table"] == "elementwise" and row["key"].get("op_name") == "residual_add"
                       for row in current["operator_missing_shapes"])
        assert any(row["table"] == "elementwise" and row["key"].get("op_name") == "rmsnorm"
                   and row["query_count"] > 0 for row in current["operator_missing_shapes"])
        result["checkpoints"][sku]["native_summaries"] = {
            "routed_seconds": current["native_megamoe_routed_seconds"],
            "shared_seconds": current["native_megamoe_shared_seconds"],
            "main_plus_draft_conservation": True,
            "main_plus_three_draft_pass_counts": True,
        }
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": previous[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / previous[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
    result["scope"] = "native mHC residual ownership and SP rows; native routed/shared summary schema preservation"
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
