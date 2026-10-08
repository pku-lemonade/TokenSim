"""Analyze the two strictly accepted original windows; never tune runtime costs."""

import hashlib
import json
from pathlib import Path
import argparse


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
BATCH = ROOT / "results/agentx-dsv4-full/native-moe-phase-efficiency/20261008-expert-shape"
BASELINE = ROOT / "results/agentx-dsv4-full/native-mtp-execution/20261008-effective-K/main"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
args = parser.parse_args()
selected = set(args.hardware)
validation_path = STAGE / (f"{args.hardware[0]}-validation.json" if len(selected) == 1 else "pipeline-validation.json")
accepted = json.loads(validation_path.read_text())
previous = json.loads((ROOT / "data/reference/agentx-moe-phase-efficiency/baseline-validation.json").read_text())
results = {}
for sku, concurrency in (("b200", 196), ("b300", 384)):
    if sku not in selected:
        continue
    point = accepted["checkpoints"][sku]
    relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
    current_path = BATCH / relative / f"agentx_c{concurrency}.json"
    old_path = BASELINE / relative / f"agentx_c{concurrency}.json"
    assert hashlib.sha256(current_path.read_bytes()).hexdigest() == point["result_sha256"]
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == previous["checkpoints"][sku]["result_sha256"]
    terminal = json.loads((BATCH / relative / "run_status.json").read_text())
    assert terminal["exit_code"] == 0
    current, old = json.loads(current_path.read_text()), json.loads(old_path.read_text())
    reload_fields = ("readmission_count", "reloaded_gpu_tokens", "reloaded_cpu_tokens",
                     "reloaded_generated_gpu_tokens", "reloaded_generated_cpu_tokens", "recomputed_tokens")
    readmission_ranks = {}
    for label, directory in (("baseline", BASELINE / relative), ("current", BATCH / relative)):
        path = directory / "native_replay/requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ranks = []
        for rank in range(8):
            requests = [row for row in rows if row["phase"] == "profiling" and row["dp_rank"] == rank]
            ranks.append({
                "rank": rank, "scope": "all profiling terminal requests, including cancelled",
                "requests": len(requests),
                "readmitted_requests": sum(row["prefill_admission_count"] > 1 for row in requests),
                "extra_admissions": sum(max(0, row["prefill_admission_count"] - 1) for row in requests),
                "reloaded_cpu_tokens": sum(row["reloaded_cpu_tokens"] for row in requests),
                "reloaded_gpu_tokens": sum(row["reloaded_gpu_tokens"] for row in requests),
                "recomputed_tokens": sum(row["recomputed_tokens"] for row in requests),
            })
        readmission_ranks[label] = {"request_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "ranks": ranks}
        assert sum(row["requests"] for row in ranks) == sum(row["phase"] == "profiling" for row in rows)
    components = {}
    for key, value in current["operator_component_seconds"].items():
        before = old["operator_component_seconds"][key] / (old["agentx_metrics"]["output_tokens"] / 1e6)
        after = value / (current["agentx_metrics"]["output_tokens"] / 1e6)
        components[key] = {"baseline_seconds_per_million_output_tokens": before,
                           "current_seconds_per_million_output_tokens": after,
                           "change_percent": 100 * (after / before - 1) if before else None}
    main = current["dsv4_main_forward_counts"]
    results[sku] = {
        "result_sha256": point["result_sha256"], "terminal": terminal,
        "metrics": point["paired_metrics"], "phase_counts": point["phase_counts"],
        "preemptions": {"baseline": old["preemption_count"], "current": current["preemption_count"]},
        "reload_accounting": {key: {"baseline": old["agentx_metrics"][key],
                                    "current": current["agentx_metrics"][key]}
                              for key in reload_fields},
        "readmission_per_rank": readmission_ranks,
        "sync_wait_fraction": {"baseline": previous["checkpoints"][sku]["sync_wait_fraction"],
                               "current": point["sync_wait_fraction"]},
        "expert_phase_transfer": point["expert_phase_transfer"],
        "moe_work_ledger": point["moe_composition"],
        "moe_counts": point["moe_counts"],
        "main_forward_counts": main,
        "capture_padding_percent_of_full": 100 * main["capture_padding_rows"] / main["full_rows"],
        "capture_projection_counts": point["capture_projection_counts"],
        "target_logits_counts": current["dsv4_target_logits"],
        "normalized_components": components,
        "per_rank": [{**row, "gpu_hit_percent": 100 * row["gpu_hit_tokens"] / row["input_tokens"],
                      "cpu_hit_percent": 100 * row["cpu_hit_tokens"] / row["input_tokens"]}
                     for row in point["per_rank"]],
        "diagnostic_metrics_within25percent": {
            name: abs(metrics["reference_error_percent"]) <= 25
            for name, metrics in point["paired_metrics"].items()
        },
    }
report = {
    "scope": "paired seed0 original complete windows after same-capacity expert-phase measured shape-efficiency transfer",
    "baseline_commit": "02f81a4",
    "strict_validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    "formal_model_accuracy_accepted": False,
    "component_normalization_is_analysis_only": True,
    "hardware": results,
    "known_remaining_issues": [
        "capture source-floor/projection ledgers are row-weighted SOL budgets, not physical tensor measurements",
        "shared-only capture increment is an analytical allocation of module timing; component launch/fusion remains unresolved",
        "TP4/8 row contracts have source-derived unit coverage; final current-code three-seed curve matrix remains required",
        "source kernel provenance, wider MoE shape efficiency/traffic and mixed-input issue rate remain unresolved",
        "expert-phase allocation is gradeD;component timers and actual nonuniform efficiency are not measured",
        "native allocator/stream/graph-driver and generic fused-MoE storage remain unmeasured",
        "closed-loop trajectories change with timing; normalized components and preemption deltas alone do not isolate causality",
    ],
}
destination = STAGE / (f"{args.hardware[0]}-analysis.json" if len(selected) == 1 else "paired-analysis.json")
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, hardware in results.items():
    point = hardware["metrics"]["output_tps"]
    print(sku, "output", point["current"], "change", point["baseline_change_percent"],
          "reference error", point["reference_error_percent"])
