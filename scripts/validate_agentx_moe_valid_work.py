#!/usr/bin/env python3
"""Audit complete MegaMoE full/valid work windows, source scope and timing."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-context-grid/20261008-preserve-query")
EVIDENCE = Path("data/reference/agentx-context-grid/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/latency/deepseek_v4_moe.py", "TokenSim/latency/deepseek_v4_moe_work.py",
    "TokenSim/latency/deepseek_v4_mtp.py", "TokenSim/llm/dp_coordinator.py",
    "TokenSim/config/psla_config.py", "util/results.py",
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
        changed = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        assert changed == SOURCE_CHANGES, changed
        old_path = old / f"agentx_c{concurrency}.json"
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == previous[sku]["result_sha256"]
        baseline = json.loads(old_path.read_text())
        current = json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(baseline["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(baseline["worker_cache_budgets"], current["worker_cache_budgets"], strict=True):
            assert {key: value for key, value in first.items() if key not in ("peak_used_slots", "evictions")} == {
                key: value for key, value in second.items() if key not in ("peak_used_slots", "evictions")
            }
        c, ledger = current["native_megamoe_counts"], current["native_megamoe_composition"]
        steps = current["operator_step_count"]
        assert c["rank_estimates"] == 4 * steps
        assert c["graph_pass"] + c["eager_pass"] == c["rank_estimates"]
        assert c["full_input_rows"] == c["routed_valid_rows"] + c["padding_rows"]
        assert c["full_input_layer_rows"] >= c["routed_valid_layer_rows"] >= c["routed_valid_rows"]
        assert 0 <= c["zero_routed_rank_passes"] <= c["rank_estimates"]
        assert c["padding_rows"] > 0
        # Counter exports omit categories never observed in this window.
        assert c.get("uniform_measured_passes", 0) + c["composed_passes"] == c["rank_estimates"]
        assert c["composed_passes"] > 0
        routed, shared = current["native_megamoe_routed_seconds"], current["native_megamoe_shared_seconds"]
        assert abs(ledger["composed_routed_seconds"] - routed) < 1e-6
        assert abs(ledger["uniform_anchor_seconds"] - ledger["anchor_difference_seconds"] - routed) < 1e-6
        assert abs(routed + shared - current["operator_component_seconds"]["moe"]) < 1e-6
        description = next(iter(current["latency_backends"].values()))["native_megamoe"]
        assert description["latency_scale"] == 1
        assert description["work_composition"]["fitted_parameters"] is False
        assert description["work_composition"]["grade"].startswith("D")
        counts = [worker["counts"] for worker in current["scheduler_stats"]]
        query_rows = sum(row["scheduled_context_query_tokens"] + row["scheduled_decode_query_tokens"] for row in counts)
        # Main61 layers plus initial draft1 have the same real query rows;
        # two later passes have one routed row per real forward request.
        assert c["routed_valid_rows"] >= 2 * query_rows
        assert c["routed_valid_layer_rows"] >= 62 * query_rows
        logits, draft = current["dsv4_target_logits"], current["native_dsv4_mtp_counts"]
        # These main checkpoints use TP1: target and draft full forward
        # tensors are exactly the owned MegaMoE inputs. Only the target is
        # repeated over61 layers; all three draft passes use one layer.
        assert c["full_input_rows"] == logits["output_forward_rows"] + draft["forward_rows"]
        assert c["full_input_layer_rows"] == 61 * logits["output_forward_rows"] + draft["forward_rows"]
        assert c["routed_valid_layer_rows"] == 60 * query_rows + c["routed_valid_rows"]
        assert draft["target_steps"] == draft["initial_passes"] == steps
        assert draft["decode_passes"] == 2 * steps and draft["passes"] == 3 * steps
        assert current["dsv4_hc_head_counts"]["target_calls"] == steps
        assert current["dsv4_hc_head_counts"]["target_rows"] == logits["final_norm_rows"]
        assert current["dsv4_hc_head_counts"]["draft_rows"] == draft["logits_rows"]
        assert logits["output_forward_rows"] >= logits["forward_rows"] >= logits["rows"]
        assert abs(current["native_mhc_main_seconds"] + current["native_mhc_draft_seconds"] - current["native_mhc_seconds"]) < 1e-6
        records = [json.loads(line) for line in (new / "native_replay/requests.jsonl").read_text().splitlines()]
        assert not any(row["length_capped"] for row in records)
        point = result["checkpoints"][sku]
        point["moe_counts"], point["moe_composition"] = c, ledger
        point["moe_description"] = description
        point["formal_length_caps"] = 0
        point["paired_baseline"] = {
            "root": str(old), "output_tps": previous[sku]["output_tps"],
            "throughput_change_percent": 100 * (point["output_tps"] / previous[sku]["output_tps"] - 1),
            "source_changes": sorted(changed), "configuration_capacity_and_dataset_unchanged": True,
        }
        ref = json.loads(Path(f"data/reference/{sku}-c{concurrency}-dep8-simplecpu/key_point.json").read_text())["metrics"]
        paths = {
            "input_tps": ("input_token_throughput_tps", "input_tput_tps"),
            "output_tps": ("output_token_throughput_tps", "output_tput_tps"),
            "ttft_p90_s": (("ttft_s", "p90"), "p90_ttft"),
            "itl_p90_s": (("tpot_s", "p90"), "p90_itl"),
            "e2e_mean_s": (("e2e_s", "mean"), "mean_e2e"),
            "gpu_hit_fraction": ("gpu_cache_hit_rate", "gpu_cache_hit_rate"),
            "cpu_hit_fraction": ("cpu_cache_hit_rate", "cpu_cache_hit_rate"),
        }
        comparison = {}
        for name, (path, ref_key) in paths.items():
            def value(metrics):
                return metrics[path[0]][path[1]] if isinstance(path, tuple) else metrics[path]
            first, second, target = value(baseline["agentx_metrics"]), value(current["agentx_metrics"]), ref[ref_key]
            comparison[name] = {"baseline": first, "current": second, "reference": target,
                                "baseline_change_percent": 100 * (second / first - 1),
                                "reference_error_percent": 100 * (second / target - 1)}
            if name.endswith("hit_fraction"):
                comparison[name]["reference_difference_percentage_points"] = 100 * (second - target)
        point["paired_metrics"] = comparison
        point["paired_components"] = {
            key: {"baseline_seconds": value, "current_seconds": current["operator_component_seconds"][key],
                  "difference_seconds": current["operator_component_seconds"][key] - value}
            for key, value in baseline["operator_component_seconds"].items()
        }
    result["scope"] = "full captured inputs, routed-valid/SP/dummy/MTP rows and uniform-anchor physical work composition"
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
