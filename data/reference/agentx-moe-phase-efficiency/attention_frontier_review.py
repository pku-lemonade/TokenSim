"""Read-only AIC joint-frontier util-hold oracle on withheld attention rows.

This only tests the multi-axis extrapolation selection with the existing SOL.
It is not a new runtime path, source-kernel correction or serving fit.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
sys.path.insert(0, str(ROOT))

from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_attention_sol import DeepSeekV4AttentionSOL
from TokenSim.operator_data.schema import TABLE_SPECS


context = HardwareContext.load(ROOT / "data")
original_audit = STAGE / "attention-current-holdout-followup.json"
old = json.loads(original_audit.read_text())
spec = TABLE_SPECS["dsv4_attention"]
hardware = {}
for sku in ("b200", "b300"):
    package = context.operator_package(context.device(sku).device_id, "vllm")
    sol = DeepSeekV4AttentionSOL(context.model("DeepSeek-V4-Pro"), context.device(sku))
    withheld = {tuple(row["key"][name] for name in spec.key_fields)
                for row in old[sku]["holdouts"]}
    groups = defaultdict(list)
    for row in package.tables["dsv4_attention"]:
        if tuple(row[name] for name in spec.key_fields) not in withheld:
            groups[tuple(row[name] for name in spec.discrete_fields)].append(row)
    cached = {}
    for discrete, rows in groups.items():
        rows.sort(key=lambda row: (row["context_len"], row["input_seq_len"], row["batch_size"]))
        coordinates = np.array([[row["context_len"], row["input_seq_len"], row["batch_size"]]
                                for row in rows], dtype=float)
        cached[discrete] = (rows, np.log2(np.maximum(coordinates, 1e-12)),
                            np.array([sol.estimate_us(row) / row["latency_us"] for row in rows]))
    points = []
    for case in old[sku]["holdouts"]:
        key = case["key"]
        rows, logs, utils = cached[tuple(key[name] for name in spec.discrete_fields)]
        query = np.log2(np.maximum(np.array([key["context_len"], key["input_seq_len"],
                                             key["batch_size"]], dtype=float), 1e-12))
        distances = np.sqrt(np.sum((logs - query) ** 2, axis=1))
        order = np.argsort(distances, kind="stable")
        picked = order[:4]  # Official AIC Grid default; not selected from serving errors.
        radius = distances[order[4]] if len(order) > 4 else np.inf
        selected_distances = distances[picked]
        weights = ((1 / (selected_distances ** 2 + 1e-12)) if np.isinf(radius) else
                   (np.maximum(0, radius - selected_distances)
                    / (radius * selected_distances + 1e-12)) ** 2)
        if weights.sum() <= 0:
            weights = np.ones(len(picked))
        prediction = float(sol.estimate_us(key) / (weights.dot(utils[picked]) / weights.sum()))
        points.append({
            "key": key, "measured_us": case["measured_us"],
            "old_prediction_us": case["sol_predicted_us"], "old_error_percent": case["sol_error_percent"],
            "joint_prediction_us": prediction,
            "joint_error_percent": 100 * (prediction / case["measured_us"] - 1),
            "anchors": [{"prefix": rows[index]["context_len"], "queries": rows[index]["input_seq_len"],
                         "batch": rows[index]["batch_size"], "latency_us": rows[index]["latency_us"],
                         "distance": float(distances[index]), "weight": float(weight)}
                        for index, weight in zip(picked, weights)],
        })
    summary = {}
    for phase in ("context", "generation"):
        for kind in ("csa", "hca"):
            for tp in (1, 2, 4, 8):
                subset = [point for point in points if
                          (point["key"]["phase"], point["key"]["layer_kind"], point["key"]["tp_size"])
                          == (phase, kind, tp)]
                if not subset:
                    continue
                summary[f"{phase}:{kind}:tp{tp}"] = {
                    "points": len(subset),
                    "old_over25": sum(abs(point["old_error_percent"]) > 25 for point in subset),
                    "joint_over25": sum(abs(point["joint_error_percent"]) > 25 for point in subset),
                    "joint_max_error_percent": max(abs(point["joint_error_percent"]) for point in subset),
                }
    hardware[sku] = {"summary": summary, "points": points}

aic = Path("/tmp/aiconfigurator")
source = aic / "aic-core/rust/aiconfigurator-core/src/perf_database/perf_interp.rs"
report = {
    "scope": "AIC Grid nn_leaves4 joint-log2 tapered modified-Shepard util hold; original SOL unchanged",
    "withholding": "all longest-context target measurements removed before anchor selection",
    "aic_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=aic, text=True).strip(),
    "aic_source": str(source), "aic_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "input_audit_sha256": hashlib.sha256(original_audit.read_bytes()).hexdigest(),
    "runtime_change": False, "serving_fit": False, "formal_model_accuracy_accepted": False,
    "hardware": hardware,
    "limits": ["only frontier hold selection tested; in-grid AIC branch semantics not ported",
               "current layout/SOL and historical kernel uncertainty retained",
               "remaining holdout failures prevent calling the extrapolation task complete"],
}
destination = STAGE / "attention-frontier-prototype-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, row in hardware.items():
    print(sku, "holdouts", len(row["points"]), "old failures",
          sum(summary["old_over25"] for summary in row["summary"].values()), "joint failures",
          sum(summary["joint_over25"] for summary in row["summary"].values()), "max error",
          max(abs(point["joint_error_percent"]) for point in row["points"]))
