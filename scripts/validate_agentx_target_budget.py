#!/usr/bin/env python3
"""Reconcile full target-query/KV/logits budget checkpoints with the tile baseline."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE_EVIDENCE = Path("data/reference/agentx-mtp-kv/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/config/psla_config.py", "TokenSim/latency/operator_table.py",
    "TokenSim/llm/llm_scheduler.py", "util/results.py",
}


def audit(root, *, hardware=("b200", "b300")):
    result = audit_batches(root, hardware=hardware)
    baseline = json.loads(BASELINE_EVIDENCE.read_text())
    for sku, concurrency in (("b200", 196), ("b300", 384)):
        if sku not in hardware:
            continue
        relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
        old = Path(baseline["result_roots"][sku]) / relative
        new = Path(root) / relative
        before = json.loads((old / "run_manifest.json").read_text())
        after = json.loads((new / "run_manifest.json").read_text())
        for field in ("point", "defaults", "backend", "simulation_seed", "dataset_sha256", "transfer_utilization_assumption"):
            assert before[field] == after[field], field
        a, b = before["source_sha256"], after["source_sha256"]
        changes = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
        assert changes == SOURCE_CHANGES, changes
        old_path = old / f"agentx_c{concurrency}.json"
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == baseline["checkpoints"][sku]["result_sha256"]
        previous, current = json.loads(old_path.read_text()), json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(previous["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(previous["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert first.keys() == second.keys()
            assert {k: v for k, v in first.items() if k not in ("peak_used_slots", "evictions")} == {
                k: v for k, v in second.items() if k not in ("peak_used_slots", "evictions")
            }
        logits = current["dsv4_target_logits"]
        batches = current["native_dsv4_attention_decode_batches"]
        assert logits["steps"] == current["operator_step_count"]
        assert logits["decode_rows"] == batches["query_rows"] == 4 * batches["requests"]
        assert batches["indexer_query_tiles"] == 2 * batches["requests"]
        assert logits["prefill_rows"] > 0
        assert logits["rows"] == logits["decode_rows"] + logits["prefill_rows"]
        scheduled = sum(w["counts"]["scheduled_decode_query_tokens"] for w in current["scheduler_stats"])
        assert scheduled % 4 == 0
        result["checkpoints"][sku]["target_logits"] = logits
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": baseline["checkpoints"][sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / baseline["checkpoints"][sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
    result["scope"] = "full terminal verification width, query+lookahead KV allocation, target decode/prefill logits rows"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    args = parser.parse_args()
    report = audit(args.batch_root, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
