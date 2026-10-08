"""Hold out measured uniform rows to audit the unfitted MegaMoE work growth."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from TokenSim.latency.deepseek_v4_moe_work import MegaMoEWorkModel


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
links = LinkCatalog.load(ROOT / "data/topologies/links.yaml")
results = {}
for sku in ("b200", "b300"):
    device = context.device(sku)
    package = context.operator_package(device.device_id, "vllm")
    work_model = MegaMoEWorkModel(
        model, device, 8,
        nvlink_bytes_per_s=links.get(device.scale_up_link).bandwidth_per_direction_bytes_per_s,
    )
    rows = [row for row in package.tables["dsv4_megamoe"] if all(
        row[key] == value for key, value in {
            "dtype": "mxfp4", "kernel_source": "deepgemm_megamoe",
            "kernel_dtype": "fp8_fp4", "distribution": "power_law_1.01",
            "pre_dispatch": "sglang_jit", "source_policy": "random",
            "hidden_size": model.hidden_size, "inter_size": model.moe_intermediate_size,
            "top_k": model.moe.num_experts_per_tok, "num_experts": model.moe.num_experts,
            "num_fused_shared_experts": 0, "tp_size": 1, "ep_size": 8,
        }.items()
    )]
    groups = defaultdict(list)
    estimates = []
    for row in rows:
        full = {rank: row["num_tokens"] for rank in range(8)}
        physical = work_model.estimate(full, full)
        sample = {key: row[key] for key in (
            "phase", "num_tokens", "latency_us", "num_max_tokens_per_rank",
            "effective_num_max_tokens_per_rank", "source_id", "collection_version",
        )}
        sample.update({"uniform_work_us": physical.latency_us,
                       "block_m": physical.block_ms[0], "expert_work_us": physical.expert_us,
                       "measured_over_work_ratio": row["latency_us"] / physical.latency_us})
        groups[row["phase"]].append(sample)
        estimates.append(sample)
    comparisons = []
    for samples in groups.values():
        samples.sort(key=lambda row: row["num_tokens"])
        for first, second in zip(samples, samples[1:]):
            same_capacity = all(first[key] == second[key] for key in (
                "num_max_tokens_per_rank", "effective_num_max_tokens_per_rank",
            ))
            prediction = first["latency_us"] * second["uniform_work_us"] / first["uniform_work_us"]
            comparisons.append({
                "phase": first["phase"], "anchor_rows": first["num_tokens"],
                "holdout_rows": second["num_tokens"], "anchor_block_m": first["block_m"],
                "holdout_block_m": second["block_m"], "same_recorded_buffer_capacity": same_capacity,
                "same_effective_buffer_capacity": first["effective_num_max_tokens_per_rank"] == second["effective_num_max_tokens_per_rank"],
                "anchor_only_prediction_us": prediction, "measured_holdout_us": second["latency_us"],
                "error_percent": 100 * (prediction / second["latency_us"] - 1),
            })
    eligible = [row for row in comparisons if row["same_recorded_buffer_capacity"]]
    effective = [row for row in comparisons if row["same_effective_buffer_capacity"]]
    directed = []
    # Runtime can normalize less valid work to a larger full-input anchor.
    # Audit both growth and shrinkage, including up to the EP8 row ratio.
    for samples in groups.values():
        for anchor in samples:
            for heldout in samples:
                if anchor is heldout or max(anchor["num_tokens"], heldout["num_tokens"]) > 8 * min(anchor["num_tokens"], heldout["num_tokens"]):
                    continue
                same_requested = anchor["num_max_tokens_per_rank"] == heldout["num_max_tokens_per_rank"]
                same_effective = anchor["effective_num_max_tokens_per_rank"] == heldout["effective_num_max_tokens_per_rank"]
                if not same_effective:
                    continue
                prediction = anchor["latency_us"] * heldout["uniform_work_us"] / anchor["uniform_work_us"]
                directed.append({"phase": anchor["phase"], "anchor_rows": anchor["num_tokens"],
                                 "holdout_rows": heldout["num_tokens"], "same_requested_capacity": same_requested,
                                 "same_effective_capacity": same_effective,
                                 "direction": "growth" if heldout["num_tokens"] > anchor["num_tokens"] else "shrinkage",
                                 "prediction_us": prediction, "measured_holdout_us": heldout["latency_us"],
                                 "error_percent": 100 * (prediction / heldout["latency_us"] - 1)})
    threshold = []
    # These are analytical continuity diagnostics, not measured samples.
    for n in (80, 81, 192, 193, 512, 513, 8192):
        full = {rank: n for rank in range(8)}
        physical = work_model.estimate(full, full)
        threshold.append({"rows": n, "block_m": physical.block_ms[0],
                          "uniform_work_us": physical.latency_us})
    results[sku] = {
        "original_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "uniform_samples": estimates, "adjacent_holdouts": comparisons,
        "same_capacity_holdout_count": len(eligible),
        "same_capacity_failures_over25percent": sum(abs(row["error_percent"]) > 25 for row in eligible),
        "same_capacity_max_abs_error_percent": max((abs(row["error_percent"]) for row in eligible), default=None),
        "effective_capacity_holdout_count": len(effective),
        "effective_capacity_failures_over25percent": sum(abs(row["error_percent"]) > 25 for row in effective),
        "effective_capacity_max_abs_error_percent": max((abs(row["error_percent"]) for row in effective), default=None),
        "directed_same_effective_capacity_holdouts_within8x": directed,
        "directed_same_effective_capacity_count": len(directed),
        "directed_same_effective_capacity_failures_over25percent": sum(abs(row["error_percent"]) > 25 for row in directed),
        "directed_same_effective_capacity_max_abs_error_percent": max((abs(row["error_percent"]) for row in directed), default=None),
        "block_boundary_diagnostics": threshold,
    }

sources = [ROOT / "TokenSim/latency/deepseek_v4_moe_work.py",
           Path("/tmp/tokensim-deepgemm-nonuniform-audit/heuristics.hpp"),
           Path("/tmp/tokensim-deepgemm-nonuniform-audit/fp8-fp4-kernel.cuh"),
           Path("/tmp/tokensim-deepgemm-buffer-audit/scheduler.cuh")]
report = {
    "scope": "uniform-module growth holdouts and weight-address ownership; runtime frozen",
    "runtime_change": False, "fitted_parameters": False, "formal_accuracy_evidence": False,
    "method": "adjacent forward holdouts plus both directions within8x for same-phase/effective-capacity rows; one original anchor times unfitted physical work ratio; target latency never enters the prediction",
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
    "hardware": results,
    "facts": [
        "pinned weight TMA addresses contain expert and N/K coordinates, not M-block coordinates; repeat loads access the same weight bytes",
        "TMA instruction traffic is not evidence of an HBM miss on every repeat; current model credits no unmeasured L2 reuse",
        "source BLOCK_M heuristic explicitly discusses a per-M-block reload cost; this supports a penalty but does not measure its HBM or L2 cost",
        "B200 samples within a phase have fixed recorded buffer capacity; B300 requested capacity varies by row even where aligned capacity stays fixed",
        "recorded effective alignment512->576 differs from current pinned1920 alignment; historical DeepGEMM revision is unknown",
    ],
    "limits": [
        "uniform holdouts audit the work model as a growth approximation; current runtime exact uniform rows still reproduce their measured module anchors",
        "same recorded capacity does not prove same historical kernel revision or expert histogram",
        "failures do not authorize a fitted serving multiplier, unmeasured cache credit, or substituting mean rank rows",
    ],
}
destination = EVIDENCE / "work-growth-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, result in results.items():
    print(sku, "same-capacity holdouts", result["same_capacity_holdout_count"],
          "failures", result["same_capacity_failures_over25percent"],
          "max error", result["same_capacity_max_abs_error_percent"])
