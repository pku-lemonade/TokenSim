"""Offline source-efficiency transfer with full-input phase ownership preserved.

No runtime changes. Each target measurement is withheld. Critical expert work
uses the current native physical model; fixed/input/final-combine work keeps
the original source efficiency. Empirical expert efficiency is approximate,
since source module timers do not identify component efficiency separately.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from TokenSim.latency.deepseek_v4_moe_work import MegaMoEWorkModel


def interpolate(training, expert_work):
    ordered = sorted(training, key=lambda row: row["expert_us"])
    lower = [row for row in ordered if row["expert_us"] <= expert_work]
    upper = [row for row in ordered if row["expert_us"] >= expert_work]
    if not lower or not upper:
        row = lower[-1] if lower else upper[0]
        return row["efficiency"], "boundary", [row["rows"]]
    a, b = lower[-1], upper[0]
    if a["expert_us"] == b["expert_us"]:
        return a["efficiency"], "exact_work", [a["rows"]]
    fraction = (expert_work - a["expert_us"]) / (b["expert_us"] - a["expert_us"])
    return (1 - fraction) * a["efficiency"] + fraction * b["efficiency"], "interpolated", [a["rows"], b["rows"]]


context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
links = LinkCatalog.load(ROOT / "data/topologies/links.yaml")
hardware = {}
for sku in ("b200", "b300"):
    device = context.device(sku)
    package = context.operator_package(device.device_id, "vllm")
    work_model = MegaMoEWorkModel(model, device, 8,
        nvlink_bytes_per_s=links.get(device.scale_up_link).bandwidth_per_direction_bytes_per_s)
    selection = {"dtype": "mxfp4", "kernel_source": "deepgemm_megamoe", "kernel_dtype": "fp8_fp4",
                 "distribution": "power_law_1.01", "pre_dispatch": "sglang_jit", "source_policy": "random",
                 "hidden_size": model.hidden_size, "inter_size": model.moe_intermediate_size,
                 "top_k": model.moe.num_experts_per_tok, "num_experts": model.moe.num_experts,
                 "num_fused_shared_experts": 0, "tp_size": 1, "ep_size": 8}
    original = [row for row in package.tables["dsv4_megamoe"] if all(row[key] == value for key, value in selection.items())]
    groups = defaultdict(list)
    for row in original:
        shape = {rank: row["num_tokens"] for rank in range(8)}
        work = work_model.estimate(shape, shape)
        common = work.startup_us + work.input_us + work.combine_us
        expert = work.fused_expert_dispatch_us
        assert abs(common + expert - work.latency_us) < 1e-9
        groups[row["phase"], row["effective_num_max_tokens_per_rank"]].append({
            "rows": row["num_tokens"], "measured_us": row["latency_us"],
            "common_us": common, "expert_us": expert, "work_us": work.latency_us,
            "efficiency": row["latency_us"] / work.latency_us,
            "block_m": work.block_ms[0], "requested_capacity": row["num_max_tokens_per_rank"],
        })
    cases = []
    for (phase, capacity), samples in groups.items():
        if len(samples) < 3:
            continue
        for target in samples:
            training = [sample for sample in samples if sample is not target]
            gamma, method, neighbors = interpolate(training, target["expert_us"])
            assert target["rows"] not in neighbors
            for anchor in training:
                if max(anchor["rows"], target["rows"]) > 8 * min(anchor["rows"], target["rows"]):
                    continue
                original_prediction = anchor["efficiency"] * target["work_us"]
                prediction = anchor["efficiency"] * target["common_us"] + gamma * target["expert_us"]
                cases.append({"phase": phase, "effective_capacity": capacity,
                              "anchor_rows": anchor["rows"], "heldout_rows": target["rows"],
                              "anchor_block_m": anchor["block_m"], "heldout_block_m": target["block_m"],
                              "method": method, "training_neighbor_rows": neighbors,
                              "predicted_us": prediction, "measured_heldout_us": target["measured_us"],
                              "error_percent": 100 * (prediction / target["measured_us"] - 1),
                              "constant_efficiency_error_percent": 100 * (original_prediction / target["measured_us"] - 1),
                              "full_input_phase_us_kept_at_anchor_efficiency": anchor["efficiency"] * target["common_us"],
                              "expert_phase_us_at_interpolated_efficiency": gamma * target["expert_us"]})
    hardware[sku] = {"source_rows_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                     "directed_holdouts": cases, "directed_count": len(cases),
                     "original_failures_over25percent": sum(abs(row["constant_efficiency_error_percent"]) > 25 for row in cases),
                     "phase_transfer_failures_over25percent": sum(abs(row["error_percent"]) > 25 for row in cases),
                     "phase_transfer_max_abs_error_percent": max(abs(row["error_percent"]) for row in cases),
                     "single_row_groups_not_transfer_validated": [{"phase": key[0], "effective_capacity": key[1], "rows": len(values)}
                                                                   for key, values in groups.items() if len(values) < 3]}
report = {"scope": "exploratory directed uniform MegaMoE efficiency holdouts, full-input phases held at source anchor efficiency",
          "runtime_change": False, "formal_model_accuracy_accepted": False,
          "operator_data_interpolated": True, "serving_parameters_fitted": False,
          "hardware": hardware,
          "limits": ["source whole-module efficiency assigned to expert phase is grade D; no component timers",
                     "uniform holdouts do not validate nonuniform/padding efficiency or kernel-version correspondence",
                     "no mean-rank module lookup, cache credit, latency multiplier or row deletion introduced",
                     "B300 singleton capacities and all context rows lack same-capacity efficiency curves"]}
destination = Path(__file__).resolve().parent / "phase-efficiency-holdout-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, row in hardware.items():
    print(sku, row["directed_count"], "original failures", row["original_failures_over25percent"],
          "phase transfer failures", row["phase_transfer_failures_over25percent"],
          "max error", row["phase_transfer_max_abs_error_percent"])
