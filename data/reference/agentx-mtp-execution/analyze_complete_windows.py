"""Analyze strictly accepted original K3/K0 windows and unchanged K3 baseline."""

import argparse
import hashlib
import json
from pathlib import Path


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
BATCH = ROOT / "results/agentx-dsv4-full/native-mtp-execution/20261008-effective-K"
BASELINE = ROOT / "results/agentx-dsv4-full/native-memory/20261008-storage-live-phases"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--point", choices=("b200-main", "b300-main", "b200-disabled", "b300-disabled"))
args = parser.parse_args()
validation_path = STAGE / (f"{args.point}-validation.json" if args.point else "pipeline-validation.json")
accepted = json.loads(validation_path.read_text())["checkpoints"]
prior = json.loads((ROOT / "data/reference/agentx-native-memory/pipeline-validation.json").read_text())["checkpoints"]
points = {}
for key, validated in accepted.items():
    sku, mode = key.split("-", 1)
    concurrency = 196 if sku == "b200" else 384
    case = f"{sku}-c{concurrency}-dep8-simplecpu" + ("-nomtp" if mode == "disabled" else "")
    directory = BATCH / mode / "seed0" / case / sku
    path = directory / f"agentx_c{concurrency}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == validated["result_sha256"]
    terminal = json.loads((directory / "run_status.json").read_text())
    assert terminal["exit_code"] == 0
    result = json.loads(path.read_text())
    old_path = BASELINE / "seed0" / f"{sku}-c{concurrency}-dep8-simplecpu" / sku / f"agentx_c{concurrency}.json"
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == prior[sku]["result_sha256"]
    baseline = json.loads(old_path.read_text())
    records_path = directory / "native_replay/requests.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    first, second = baseline["agentx_metrics"], result["agentx_metrics"]
    metrics = {}
    for name, field in (("output_tps", "output_token_throughput_tps"), ("input_tps", "input_token_throughput_tps"),
                        ("gpu_hit_fraction", "gpu_cache_hit_rate"), ("cpu_hit_fraction", "cpu_cache_hit_rate")):
        metrics[name] = {"prior_k3": first[field], "current": second[field],
                         "change_vs_prior_k3_percent": 100 * (second[field] / first[field] - 1)}
    for name, field, statistic in (("ttft_p90_s", "ttft_s", "p90"), ("itl_p90_s", "tpot_s", "p90"),
                                   ("e2e_mean_s", "e2e_s", "mean")):
        metrics[name] = {"prior_k3": first[field][statistic], "current": second[field][statistic],
                         "change_vs_prior_k3_percent": 100 * (second[field][statistic] / first[field][statistic] - 1)}
    cache = result["worker_cache_budgets"][0]["cache"]
    counts = result.get("native_dsv4_mtp_counts") or {}
    components = result["operator_component_seconds"]
    points[key] = {
        "result_sha256": validated["result_sha256"], "terminal": terminal,
        "phase_counts": second["phase_request_counts"], "metrics": metrics,
        "execution_k": 0 if mode == "disabled" else 3,
        "preemptions": result["preemption_count"], "request_log_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "extra_admissions_all_profiling_terminal": sum(max(0, row["prefill_admission_count"] - 1)
                                                       for row in records if row["phase"] == "profiling"),
        "derived_capacity": {"slots": cache["num_gpu_blocks"], "weight_bytes": cache["model_param_size"],
                              "available_kv_bytes": cache["memory_budget"]["available_kv_bytes"],
                              "log_equivalent_tokens": cache["logged_equivalent_tokens_computed"]},
        "draft_counts": counts, "draft_component_seconds": result.get("native_dsv4_mtp_component_seconds") or {},
        "main_counts": result["dsv4_main_forward_counts"],
        "normalized_components_seconds_per_million_output": {name: value / (second["output_tokens"] / 1e6)
                                                               for name, value in components.items()},
        "normalization_is_analysis_only": True,
        "per_rank": [{"rank": rank,
                      "profiling_terminal_requests": sum(row["phase"] == "profiling" and row["dp_rank"] == rank for row in records),
                      "extra_admissions": sum(max(0, row["prefill_admission_count"] - 1) for row in records
                                              if row["phase"] == "profiling" and row["dp_rank"] == rank)} for rank in range(8)],
        "independent_reference_compatible": mode == "main",
    }
    if mode == "main":
        points[key]["paired_reference_metrics"] = validated["paired_metrics"]
        points[key]["native_memory_cache_configuration_unchanged"] = validated["paired_baseline"]["configuration_capacity_and_dataset_unchanged"]

ablation = {}
for sku in ("b200", "b300"):
    main, disabled = points.get(f"{sku}-main"), points.get(f"{sku}-disabled")
    if main and disabled:
        ablation[sku] = {"scope": "same current source; K0 removes module costs/storage, not only output progress",
                         "independent_disabled_reference_available": False,
                         "metrics": {name: {"k3": row["current"], "k0": disabled["metrics"][name]["current"],
                                             "k0_change_percent": 100 * (disabled["metrics"][name]["current"] / row["current"] - 1)}
                                     for name, row in main["metrics"].items()},
                         "k3_capacity": main["derived_capacity"], "k0_capacity": disabled["derived_capacity"]}
report = {"scope": "current K3 versus validated memory-stage baseline and complete current K0 ablation",
          "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
          "formal_model_accuracy_accepted": False, "points": points, "ablation": ablation,
          "limits": ["seed0 complete windows do not replace final current-code seed coverage",
                     "disabled points have no independent reference identity; compare K0/K3 current-source behavior",
                     "closed-loop traces/cache/admissions change when K and module allocation change",
                     "existing native operator/kernel/allocator uncertainty unchanged"]}
destination = STAGE / (f"{args.point}-analysis.json" if args.point else "paired-analysis.json")
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for key, point in points.items():
    print(key, point["metrics"]["output_tps"]["current"], "draft passes", point["draft_counts"].get("passes", 0))
