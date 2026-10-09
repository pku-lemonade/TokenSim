#!/usr/bin/env python3
"""Audit full padding/metadata windows with real progress and query ledgers."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-context-recovery/20261008-private-context")
EVIDENCE = Path("data/reference/agentx-context-recovery/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/block/hybrid_manager.py", "TokenSim/latency/base.py",
    "TokenSim/latency/deepseek_v4_mtp.py", "TokenSim/latency/operator_table.py",
    "TokenSim/llm/dp_coordinator.py", "TokenSim/llm/llm_request.py",
    "TokenSim/llm/llm_scheduler.py", "TokenSim/workload/agentx_native/replay.py",
}


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
        baseline, current = json.loads(old_path.read_text()), json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(baseline["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(baseline["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert first.keys() == second.keys()
            assert {key: value for key, value in first.items() if key not in ("peak_used_slots", "evictions")} == {
                key: value for key, value in second.items() if key not in ("peak_used_slots", "evictions")
            }
        counts = [worker["counts"] for worker in current["scheduler_stats"]]
        totals = {key: sum(row[key] for row in counts) for key in (
            "scheduled_context_tokens", "scheduled_context_query_tokens", "scheduled_decode_query_tokens",
            "scheduled_placeholder_queries", "padded_context_admissions", "scheduled_logits_rows",
        )}
        assert totals["scheduled_context_query_tokens"] == totals["scheduled_context_tokens"] + totals["scheduled_placeholder_queries"]
        assert totals["scheduled_placeholder_queries"] == 3 * totals["padded_context_admissions"]
        logits = current["target_logits"]
        assert logits["placeholder_rows"] == totals["scheduled_placeholder_queries"]
        assert logits["rows"] >= totals["scheduled_logits_rows"]  # idle-rank dummy rows
        assert logits["output_forward_rows"] >= logits["forward_rows"] >= logits["rows"]
        assert current["attention_module_decode_batches"]["query_rows"] >= (
            totals["scheduled_decode_query_tokens"] + logits["context_decode_metadata_rows"]
        )
        records = [json.loads(line) for line in (new / "native_replay/requests.jsonl").read_text().splitlines()]
        # All raw requests fit this 1M model; length clipping must remain zero
        # in formal windows, though native boundary unit cases exercise it.
        assert not any(row["length_capped"] for row in records)
        steps = current["operator_step_count"]
        draft = current["speculative_counts"]
        assert draft["target_steps"] == draft["initial_passes"] == steps
        assert draft["decode_passes"] == 2 * steps and draft["passes"] == 3 * steps
        assert current["residual_output_counts"]["target_calls"] == steps
        assert current["residual_output_counts"]["target_rows"] == logits["final_norm_rows"]
        assert current["residual_output_counts"]["draft_rows"] == draft["logits_rows"]
        moe = current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"]
        assert abs(moe - current["operator_component_seconds"]["moe"]) < 1e-6
        assert current["native_megamoe_counts"]["rank_estimates"] == 4 * steps
        assert abs(current["native_mhc_main_seconds"] + current["native_mhc_draft_seconds"] - current["native_mhc_seconds"]) < 1e-6
        result["checkpoints"][sku]["query_metadata"] = {"scheduler": totals, "target_logits": logits, "formal_length_caps": 0}
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": previous[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / previous[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
        reference = json.loads(Path(f"data/reference/{sku}-c{concurrency}-dep8-simplecpu/key_point.json").read_text())["metrics"]
        metric_paths = {
            "input_tps": ("input_token_throughput_tps", "input_tput_tps"),
            "output_tps": ("output_token_throughput_tps", "output_tput_tps"),
            "ttft_p90_s": (("ttft_s", "p90"), "p90_ttft"),
            "itl_p90_s": (("tpot_s", "p90"), "p90_itl"),
            "e2e_mean_s": (("e2e_s", "mean"), "mean_e2e"),
            "gpu_hit_fraction": ("gpu_cache_hit_rate", "gpu_cache_hit_rate"),
            "cpu_hit_fraction": ("cpu_cache_hit_rate", "cpu_cache_hit_rate"),
        }
        comparison = {}
        for name, (path, ref_key) in metric_paths.items():
            def value(metrics):
                return metrics[path[0]][path[1]] if isinstance(path, tuple) else metrics[path]
            first, second, target = value(baseline["agentx_metrics"]), value(current["agentx_metrics"]), reference[ref_key]
            comparison[name] = {
                "baseline": first, "current": second, "reference": target,
                "baseline_change_percent": 100 * (second / first - 1),
                "reference_error_percent": 100 * (second / target - 1),
            }
            if name.endswith("hit_fraction"):
                comparison[name]["reference_difference_percentage_points"] = 100 * (second - target)
        result["checkpoints"][sku]["paired_metrics"] = comparison
        result["checkpoints"][sku]["paired_components"] = {
            key: {"baseline_seconds": value, "current_seconds": current["operator_component_seconds"][key],
                  "difference_seconds": current["operator_component_seconds"][key] - value}
            for key, value in baseline["operator_component_seconds"].items()
        }
    result["scope"] = "waiting placeholder verify, real progress/logits/attention/graph roles and model-length boundaries"
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
