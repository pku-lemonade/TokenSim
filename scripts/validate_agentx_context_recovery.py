#!/usr/bin/env python3
"""Audit complete native context recovery windows and unchanged input metrics."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-output-head/20261008-full-forward")
EVIDENCE = Path("data/reference/agentx-output-head/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/block/prefix_cache.py", "TokenSim/block/hybrid_manager.py",
    "TokenSim/llm/llm_request.py", "TokenSim/llm/llm_scheduler.py",
    "TokenSim/kv_transfer/connectors/simple_cpu.py",
    "TokenSim/kv_transfer/connectors/native_mooncake.py",
    "TokenSim/workload/agentx_native/replay.py", "util/results.py",
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
        previous = json.loads(old_path.read_text())
        current = json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(previous["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(previous["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert first.keys() == second.keys()
            assert {key: value for key, value in first.items() if key not in ("peak_used_slots", "evictions")} == {
                key: value for key, value in second.items() if key not in ("peak_used_slots", "evictions")
            }
        records = [json.loads(line) for line in (new / "native_replay/requests.jsonl").read_text().splitlines()]
        successful = [r for r in records if r["phase"] == "profiling" and r["outcome"] == "successful"]
        metrics = current["agentx_metrics"]
        for r in records:
            assert r["reloaded_generated_gpu_tokens"] >= 0 and r["reloaded_generated_cpu_tokens"] >= 0
            assert r["latest_gpu_context_hit_tokens"] >= r["latest_gpu_hit_tokens"] >= 0
            assert r["latest_cpu_context_hit_tokens"] >= r["latest_cpu_hit_tokens"] >= 0
            assert r["latest_gpu_hit_tokens"] + r["latest_cpu_hit_tokens"] + r["latest_computed_input_tokens"] == r["input_tokens"]
            if r["prefill_source_observed"]:
                assert r["gpu_hit_tokens"] + r["cpu_hit_tokens"] + r["computed_input_tokens"] == r["input_tokens"]
        for key in ("reloaded_generated_gpu_tokens", "reloaded_generated_cpu_tokens"):
            assert metrics[key] == sum(r[key] for r in successful), key
        steps = current["operator_step_count"]
        counts = current["speculative_counts"]
        assert counts["target_steps"] == counts["initial_passes"] == steps
        assert counts["decode_passes"] == 2 * steps and counts["passes"] == 3 * steps
        assert current["native_megamoe_counts"]["rank_estimates"] == 4 * steps
        native_moe = current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"]
        assert abs(native_moe - current["operator_component_seconds"]["moe"]) < 1e-6
        assert abs(current["native_mhc_main_seconds"] + current["native_mhc_draft_seconds"] - current["native_mhc_seconds"]) < 1e-6
        assert current["native_mhc_seconds"] <= current["operator_component_seconds"]["elementwise"]
        head = current["residual_output_counts"]
        assert head["target_calls"] == steps and head["draft_calls"] == 3 * steps
        assert head["target_rows"] == current["target_logits"]["final_norm_rows"]
        assert head["draft_rows"] == counts["logits_rows"]
        result["checkpoints"][sku]["context_recovery"] = {
            key: metrics[key] for key in (
                "readmission_count", "reloaded_gpu_tokens", "reloaded_cpu_tokens",
                "reloaded_generated_gpu_tokens", "reloaded_generated_cpu_tokens", "recomputed_tokens",
            )
        }
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": baseline[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / baseline[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
    result["scope"] = "whole-context cache identity and n-1 hit cap; same-request generated state; original input denominator"
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
