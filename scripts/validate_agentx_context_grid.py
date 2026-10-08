#!/usr/bin/env python3
"""Validate complete windows after coupled-axis context query correction."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-attention-composition/20261008-shared-projections")
EVIDENCE = Path("data/reference/agentx-attention-composition/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/latency/deepseek_v4_attention.py"}


def audit(root, *, hardware=("b200", "b300")):
    report = audit_batches(root, hardware=hardware)
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
        baseline = json.loads(old_path.read_text())
        current = json.loads((new / f"agentx_c{concurrency}.json").read_text())
        assert len(baseline["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(baseline["worker_cache_budgets"], current["worker_cache_budgets"], strict=True):
            assert {k: v for k, v in first.items() if k not in ("peak_used_slots", "evictions")} == {
                k: v for k, v in second.items() if k not in ("peak_used_slots", "evictions")
            }
        provenance = next(iter(current["latency_backends"].values()))["native_dsv4_attention"]
        assert provenance["context_query_grid"].startswith("query floors across prefixes")
        composition = current["native_dsv4_attention_composition"]
        shared, phase, standalone = (composition[name] for name in (
            "shared_projection_seconds", "phase_seconds", "standalone_module_seconds"))
        assert shared > 0 and phase > 0 and standalone > 0
        assert composition["batches"] >= composition["mixed_phase_batches"] > 0
        assert composition["shared_layer_invocations"] == 61 * composition["batches"]
        assert composition["logical_projection_rows"] == 61 * composition["query_rows"]
        assert composition["source_floor_projection_rows"] >= composition["logical_projection_rows"]
        assert abs(standalone - shared - phase - composition["avoided_shared_seconds"]) < 1e-6
        assert shared + phase <= current["native_dsv4_attention_seconds"] + 1e-6
        counts = [worker["counts"] for worker in current["scheduler_stats"]]
        totals = {key: sum(row[key] for row in counts) for key in (
            "scheduled_context_tokens", "scheduled_context_query_tokens", "scheduled_decode_query_tokens",
            "scheduled_placeholder_queries", "padded_context_admissions", "scheduled_logits_rows",
        )}
        assert totals["scheduled_context_query_tokens"] == totals["scheduled_context_tokens"] + totals["scheduled_placeholder_queries"]
        assert totals["scheduled_placeholder_queries"] == 3 * totals["padded_context_admissions"]
        logits = current["dsv4_target_logits"]
        assert logits["placeholder_rows"] == totals["scheduled_placeholder_queries"]
        assert logits["output_forward_rows"] >= logits["forward_rows"] >= logits["rows"] >= totals["scheduled_logits_rows"]
        assert composition["query_rows"] <= logits["forward_rows"]
        assert current["native_dsv4_attention_decode_batches"]["query_rows"] >= (
            totals["scheduled_decode_query_tokens"] + logits["context_decode_metadata_rows"]
        )
        records = [json.loads(line) for line in (new / "native_replay/requests.jsonl").read_text().splitlines()]
        assert not any(row["length_capped"] for row in records)
        steps = current["operator_step_count"]
        draft = current["native_dsv4_mtp_counts"]
        assert draft["target_steps"] == draft["initial_passes"] == steps
        assert draft["decode_passes"] == 2 * steps and draft["passes"] == 3 * steps
        assert current["dsv4_hc_head_counts"]["target_calls"] == steps
        assert current["dsv4_hc_head_counts"]["target_rows"] == logits["final_norm_rows"]
        assert current["dsv4_hc_head_counts"]["draft_rows"] == draft["logits_rows"]
        assert abs(current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"]
                   - current["operator_component_seconds"]["moe"]) < 1e-6
        assert current["native_megamoe_counts"]["rank_estimates"] == 4 * steps
        assert abs(current["native_mhc_main_seconds"] + current["native_mhc_draft_seconds"] - current["native_mhc_seconds"]) < 1e-6
        point = report["checkpoints"][sku]
        point["composition"] = composition
        point["query_metadata"] = {"scheduler": totals, "target_logits": logits, "formal_length_caps": 0}
        point["paired_baseline"] = {
            "root": str(old), "output_tps": previous[sku]["output_tps"],
            "throughput_change_percent": 100 * (point["output_tps"] / previous[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
            "context_clamped_queries_before": sum(v for k, v in baseline["native_dsv4_attention_counts"].items()
                                                   if k.startswith("context:") and k.endswith(":clamped")),
            "context_clamped_queries_after": sum(v for k, v in current["native_dsv4_attention_counts"].items()
                                                  if k.startswith("context:") and k.endswith(":clamped")),
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
        point["paired_per_rank"] = [
            {"rank": first["rank"], "baseline": first, "current": second}
            for first, second in zip(previous[sku]["per_rank"], point["per_rank"], strict=True)
        ]
    report["scope"] = "context query/prefix grid correction with measured anchors, true floors and unchanged runtime budgets"
    return report


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
