#!/usr/bin/env python3
"""Strict full-window audit of DSV4 output-row scope and mHC summaries."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-mhc-scope/20261007-residual-scope")
EVIDENCE = Path("data/reference/agentx-mhc-scope/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/config/psla_config.py", "TokenSim/latency/operator_table.py",
    "TokenSim/latency/deepseek_v4_head.py", "TokenSim/latency/deepseek_v4_mtp.py",
    "TokenSim/latency/deepseek_v4_moe.py", "util/results.py",
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
        steps = current["operator_step_count"]
        logits = current["target_logits"]
        counts = current["residual_output_counts"]
        seconds = current["residual_output_seconds"]
        draft = current["speculative_counts"]
        assert logits["steps"] == counts["target_calls"] == draft["target_steps"] == steps
        assert logits["hc_head_rows"] == logits["final_norm_rows"] == logits["output_forward_rows"] == counts["target_rows"]
        assert logits["output_forward_rows"] >= logits["forward_rows"] >= logits["rows"] > 0
        assert counts["draft_calls"] == draft["passes"] == 3 * steps
        assert draft["initial_passes"] == steps and draft["decode_passes"] == 2 * steps
        assert counts["draft_rows"] == draft["logits_rows"] > 0
        assert 0 < seconds["draft"] < current["speculative_component_seconds"]["lm_head"]
        assert seconds["target"] > 0 and sum(seconds.values()) < current["operator_component_seconds"]["lm_head"]
        main, mtp = current["native_mhc_main_seconds"], current["native_mhc_draft_seconds"]
        assert main > 0 and mtp > 0 and abs(main + mtp - current["native_mhc_seconds"]) < 1e-6
        assert main + mtp <= current["operator_component_seconds"]["elementwise"]
        assert mtp <= current["speculative_component_seconds"]["elementwise"]
        assert main <= current["operator_component_seconds"]["elementwise"] - current["speculative_component_seconds"]["elementwise"]
        # mHC counts describe shape lookups. Main seconds, unlike its lookup
        # count, are weighted by the actual 61 transformer-layer copies.
        assert sum(current["native_mhc_counts"]["main"].values()) == 2 * steps
        assert sum(current["native_mhc_counts"]["draft"].values()) == 6 * steps
        native = current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"]
        assert abs(native - current["operator_component_seconds"]["moe"]) < 1e-6
        assert abs(current["moe_compute_latency"] + current["moe_straggler_latency"]
                   + current["speculative_component_seconds"]["moe"] - native) < 1e-6
        assert current["native_megamoe_counts"]["rank_estimates"] == 4 * steps
        assert current["operator_missing_shape_groups"] == len(current["operator_missing_shapes"])
        assert not any(row["table"] == "elementwise" and row["key"].get("op_name") == "residual_add"
                       for row in current["operator_missing_shapes"])
        description = next(iter(current["latency_backends"].values()))["target_output"]
        assert description["hc_head"]["source_grade"] == "D"
        result["checkpoints"][sku]["output_head"] = {
            "counts": counts, "seconds": seconds, "target_logits": logits,
            "provenance": description,
        }
        result["checkpoints"][sku]["native_mhc"] = {
            "main_seconds": main, "draft_seconds": mtp,
            "total_seconds": main + mtp, "shape_lookup_counts": current["native_mhc_counts"],
        }
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": baseline[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / baseline[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
    result["scope"] = "target hc_head/final norm full-forward rows; selected-row draft head; actual main/draft native mHC summary"
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
