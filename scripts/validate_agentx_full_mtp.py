#!/usr/bin/env python3
"""Strict full-window audit of all DSV4 target/draft passes."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-target-budget/20261007-target-scope")
EVIDENCE = Path("data/reference/agentx-target-budget/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/config/psla_config.py", "TokenSim/latency/operator_table.py",
    "TokenSim/latency/deepseek_v4_mtp.py", "TokenSim/latency/deepseek_v4_moe.py",
    "TokenSim/llm/dp_coordinator.py", "TokenSim/llm/llm_scheduler.py", "util/results.py",
}


def audit(root, *, hardware=("b200", "b300")):
    result = audit_batches(root, hardware=hardware)
    baseline = json.loads(EVIDENCE.read_text())["checkpoints"]
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
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == baseline[sku]["result_sha256"]
        previous, current = json.loads(old_path.read_text()), json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(previous["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(previous["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert first.keys() == second.keys()
            assert {key: value for key, value in first.items() if key not in ("peak_used_slots", "evictions")} == {
                key: value for key, value in second.items() if key not in ("peak_used_slots", "evictions")
            }
        c, t = current["native_dsv4_mtp_counts"], current["native_dsv4_mtp_component_seconds"]
        steps = current["operator_step_count"]
        assert c["target_steps"] == c["initial_passes"] == steps
        assert c["decode_passes"] == 2 * steps and c["passes"] == 3 * steps
        assert c["e_proj_rows"] == c["forward_rows"]  # DEP8 TP1
        assert c["h_proj_rows"] == 4 * c["e_proj_rows"]
        assert c["logits_rows"] > 0 and c["graph_passes"] + c["eager_passes"] == c["passes"]
        for name in ("gemm", "attention", "moe", "elementwise", "lm_head"):
            assert 0 < t[name] <= current["operator_component_seconds"][name]
        # Older LLMResult schemas discard the native routed/shared summary.
        # The independently recorded main-stack MoEStats ledger and explicit
        # draft ledger still prove the full component sum, rather than
        # silently skipping time conservation when those optional keys vanish.
        main_moe = current["moe_compute_latency"] + current["moe_straggler_latency"]
        assert abs(main_moe + t["moe"] - current["operator_component_seconds"]["moe"]) < 1e-6
        native_exported = "native_megamoe_routed_seconds" in current
        if native_exported:
            assert abs(current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"]
                       - current["operator_component_seconds"]["moe"]) < 1e-6
        description = next(iter(current["latency_backends"].values()))["native_dsv4_mtp"]
        assert description["K"] == 3 and description["attention"].startswith("SWA only")
        result["checkpoints"][sku]["draft_counts"] = c
        result["checkpoints"][sku]["draft_component_seconds"] = t
        result["checkpoints"][sku]["draft_provenance"] = description
        result["checkpoints"][sku]["moe_time_ledger"] = {
            "main_stack_seconds": main_moe, "draft_seconds": t["moe"],
            "total_seconds": current["operator_component_seconds"]["moe"],
            "main_plus_draft_conservation": True,
            "native_routed_shared_fields_preserved": native_exported,
            "schema_export_followup_needed": not native_exported,
        }
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": baseline[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / baseline[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
    result["scope"] = "all MTP forward/head passes and prefill lookahead; measured primitives plus explicit analytical SWA/hc_head gaps"
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
