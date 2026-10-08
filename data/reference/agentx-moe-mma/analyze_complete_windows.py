"""Summarize strictly accepted full windows; normalization is analysis only."""

import hashlib
import json
from pathlib import Path


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
BATCH = ROOT / "results/agentx-dsv4-full/native-moe-mma/20261008-dynamic-tails-pipe"
BASELINE = ROOT / "results/agentx-dsv4-full/native-moe-valid-work/20261008-full-valid-rows"
validation_path = STAGE / "pipeline-validation.json"
accepted = json.loads(validation_path.read_text())
previous = json.loads((ROOT / "data/reference/agentx-moe-valid-work/pipeline-validation.json").read_text())
results = {}
for sku, concurrency in (("b200", 196), ("b300", 384)):
    point = accepted["checkpoints"][sku]
    relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
    current_path = BATCH / relative / f"agentx_c{concurrency}.json"
    old_path = BASELINE / relative / f"agentx_c{concurrency}.json"
    assert hashlib.sha256(current_path.read_bytes()).hexdigest() == point["result_sha256"]
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == previous["checkpoints"][sku]["result_sha256"]
    terminal = json.loads((BATCH / relative / "run_status.json").read_text())
    assert terminal["exit_code"] == 0
    current, old = json.loads(current_path.read_text()), json.loads(old_path.read_text())
    counts, composition = current["native_megamoe_counts"], current["native_megamoe_composition"]
    components = {}
    for key, value in current["operator_component_seconds"].items():
        before = old["operator_component_seconds"][key] / (old["agentx_metrics"]["output_tokens"] / 1e6)
        after = value / (current["agentx_metrics"]["output_tokens"] / 1e6)
        components[key] = {"baseline_seconds_per_million_output_tokens": before,
                           "current_seconds_per_million_output_tokens": after,
                           "change_percent": 100 * (after / before - 1) if before else None}
    results[sku] = {
        "result_sha256": point["result_sha256"], "terminal": terminal,
        "metrics": point["paired_metrics"], "phase_counts": point["phase_counts"],
        "preemptions": {"baseline": old["preemption_count"], "current": current["preemption_count"]},
        "sync_wait_fraction": {"baseline": previous["checkpoints"][sku]["sync_wait_fraction"],
                               "current": point["sync_wait_fraction"]},
        "owned_work": {
            "full_rows": counts["full_input_rows"], "valid_rows": counts["routed_valid_rows"],
            "padding_rows": counts["padding_rows"],
            "padding_percent": 100 * counts["padding_rows"] / counts["full_input_rows"],
            "layer_padding_percent": 100 * (counts["full_input_layer_rows"] - counts["routed_valid_layer_rows"]) / counts["full_input_layer_rows"],
            "zero_routed_passes": counts["zero_routed_rank_passes"],
            "uniform_measured_passes": counts.get("uniform_measured_passes", 0),
            "composed_passes": counts["composed_passes"],
        },
        "composition": composition, "normalized_components": components,
        "per_rank": [{**row, "gpu_hit_percent": 100 * row["gpu_hit_tokens"] / row["input_tokens"],
                      "cpu_hit_percent": 100 * row["cpu_hit_tokens"] / row["input_tokens"]}
                     for row in point["per_rank"]],
        "diagnostic_metrics_within25percent": {
            name: abs(metrics["reference_error_percent"]) <= 25
            for name, metrics in point["paired_metrics"].items()
        },
    }
report = {
    "scope": "paired seed0 complete windows after dynamic16 MMA tails and mixed-input theoretical pipe",
    "strict_validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    "formal_model_accuracy_accepted": False, "component_normalization_is_analysis_only": True,
    "hardware": results,
    "known_next_stage_issues": [
        "critical-rank integer spectrum and occupancy differ from repeated local midpoint proxy",
        "catalog FP8 theoretical pipe adopted; mixed-input issue cycles/efficiency still unmeasured",
        "uniform work normalization beyond adjacent pairs fails directed8x holdouts",
        "source issued weight TMA is not a measurement of HBM/L2 cache traffic",
    ],
}
destination = STAGE / "paired-analysis.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, hardware in results.items():
    print(sku, "output", hardware["metrics"]["output_tps"]["current"],
          "change", hardware["metrics"]["output_tps"]["baseline_change_percent"],
          "reference error", hardware["metrics"]["output_tps"]["reference_error_percent"])
