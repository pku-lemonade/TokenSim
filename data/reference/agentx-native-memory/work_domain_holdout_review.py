"""Offline uniform MoE holdouts: interpolate measured efficiency in work space.

Exploratory only. Withheld latency never enters training; all phase/capacity
rows are retained. This does not replace the full-input runtime anchor or
justify a mean-token module query, cache credit or serving-scale fit.
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


def efficiency_prediction(samples, work):
    ordered = sorted(samples, key=lambda row: row["work_us"])
    below = [row for row in ordered if row["work_us"] <= work]
    above = [row for row in ordered if row["work_us"] >= work]
    if below and above:
        lower, upper = below[-1], above[0]
        if upper["work_us"] == lower["work_us"]:
            return lower["latency_us"], "exact_work", [lower["rows"]]
        weight = (work - lower["work_us"]) / (upper["work_us"] - lower["work_us"])
        efficiency = (1 - weight) * lower["latency_us"] / lower["work_us"] + weight * upper["latency_us"] / upper["work_us"]
        return work * efficiency, "interpolated_work_efficiency", [lower["rows"], upper["rows"]]
    nearest = below[-1] if below else above[0]
    return work * nearest["latency_us"] / nearest["work_us"], "constant_boundary_efficiency", [nearest["rows"]]


context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
links = LinkCatalog.load(ROOT / "data/topologies/links.yaml")
hardware = {}
for sku in ("b200", "b300"):
    device = context.device(sku)
    package = context.operator_package(device.device_id, "vllm")
    estimator = MegaMoEWorkModel(model, device, 8,
        nvlink_bytes_per_s=links.get(device.scale_up_link).bandwidth_per_direction_bytes_per_s)
    requested = {"dtype": "mxfp4", "kernel_source": "deepgemm_megamoe", "kernel_dtype": "fp8_fp4",
                 "distribution": "power_law_1.01", "pre_dispatch": "sglang_jit", "source_policy": "random",
                 "hidden_size": model.hidden_size, "inter_size": model.moe_intermediate_size,
                 "top_k": model.moe.num_experts_per_tok, "num_experts": model.moe.num_experts,
                 "num_fused_shared_experts": 0, "tp_size": 1, "ep_size": 8}
    original = [row for row in package.tables["dsv4_megamoe"] if all(row[k] == v for k, v in requested.items())]
    groups = defaultdict(list)
    for row in original:
        shape = {rank: row["num_tokens"] for rank in range(8)}
        work = estimator.estimate(shape, shape)
        groups[row["phase"], row["effective_num_max_tokens_per_rank"]].append(
            {"rows": row["num_tokens"], "latency_us": row["latency_us"], "work_us": work.latency_us,
             "block_m": work.block_ms[0], "effective_capacity": row["effective_num_max_tokens_per_rank"]})
    cases = []
    for (phase, capacity), samples in groups.items():
        if len(samples) < 3:
            continue
        for target in samples:
            training = [row for row in samples if row is not target]
            predicted, method, neighbors = efficiency_prediction(training, target["work_us"])
            # The target is physically evaluated, but its module measurement
            # is withheld. Neighbors are chosen only by training work, never
            # by measured target latency or end-to-end serving error.
            assert target["rows"] not in neighbors
            nearest = min(training, key=lambda row: abs(row["work_us"] - target["work_us"]))
            old = nearest["latency_us"] * target["work_us"] / nearest["work_us"]
            cases.append({"phase": phase, "effective_capacity": capacity,
                          "target_rows": target["rows"], "target_work_us": target["work_us"],
                          "target_measured_us": target["latency_us"], "method": method,
                          "training_neighbor_rows": neighbors, "predicted_us": predicted,
                          "error_percent": 100 * (predicted / target["latency_us"] - 1),
                          "nearest_work_constant_efficiency_prediction_us": old,
                          "nearest_work_constant_efficiency_error_percent": 100 * (old / target["latency_us"] - 1)})
    within = [row for row in cases if row["method"] == "interpolated_work_efficiency"]
    boundaries = [row for row in cases if row["method"] == "constant_boundary_efficiency"]
    hardware[sku] = {"source_rows_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                     "uniform_groups": [{"phase": key[0], "effective_capacity": key[1], "samples": values}
                                        for key, values in groups.items()],
                     "withheld_cases": cases, "interior_cases": len(within),
                     "interior_failures_over25percent": sum(abs(row["error_percent"]) > 25 for row in within),
                     "interior_max_abs_error_percent": max((abs(row["error_percent"]) for row in within), default=None),
                     "boundary_cases": len(boundaries),
                     "boundary_failures_over25percent": sum(abs(row["error_percent"]) > 25 for row in boundaries)}

report = {
    "scope": "exploratory uniform-operator work-domain efficiency interpolation leave-one-out; runtime not changed",
    "runtime_change": False, "formal_accuracy_accepted": False,
    "operator_efficiency_interpolated": True, "serving_parameters_fitted": False,
    "hardware": hardware,
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in (ROOT / "TokenSim/latency/deepseek_v4_moe_work.py",
                                   ROOT / "TokenSim/latency/deepseek_v4_moe_spectrum.py")],
    "limits": [
        "uniform holdouts do not recover nonuniform or padding component efficiency; passing cannot close task10.2",
        "work-domain scalar can alias different BLOCK_M/traffic/shape regimes; historical kernel revision still unverified",
        "interpolation and boundary failures are reported separately; no original measurement deleted",
        "does not authorize replacing a full-input uniform anchor with mean valid rank rows",
        "needs source/phase ownership and broader directed holdouts before any independent runtime implementation",
    ],
}
destination = Path(__file__).resolve().parent / "work-domain-holdout-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, row in hardware.items():
    print(sku, "interior", row["interior_cases"], "failures", row["interior_failures_over25percent"],
          "max error", row["interior_max_abs_error_percent"], "boundary failures", row["boundary_failures_over25percent"])
