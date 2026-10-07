#!/usr/bin/env python3
"""Leave the longest context out of every source shape and audit SOL growth."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_attention_sol import DeepSeekV4AttentionSOL
from TokenSim.operator_data.lookup import LookupPolicy, OperatorLookup
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS


def legacy_work(_, key):
    """Historical heuristic, for a diagnostic comparison only."""
    b, q, length, ratio = (float(key[name]) for name in ("batch_size", "input_seq_len", "context_len", "compress_ratio"))
    if key["phase"] == "generation":
        attended = min(1024, length / ratio) + 128 if key["layer_kind"] == "csa" else length / ratio + 128
        return b * (1 + attended + (length / ratio if key["layer_kind"] == "csa" else 0))
    return b * q * (1 + 128 + length / ratio + (length / ratio if key["layer_kind"] == "csa" else 0))


def audit(sku, context):
    model, device = context.model("DeepSeek-V4-Pro"), context.device(sku)
    package = context.operator_package(device.device_id, "vllm")
    original = package.tables["dsv4_attention"]
    spec = TABLE_SPECS["dsv4_attention"]
    fields = [name for name in spec.key_fields if name != "context_len"]
    groups = defaultdict(list)
    for row in original:
        groups[tuple(row[name] for name in fields)].append(row)
    holdouts = []
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: row["context_len"])
        if len(ordered) >= 3 and ordered[-2]["context_len"] > 0:
            holdouts.append(ordered[-1])
    withheld = {tuple(row[name] for name in spec.key_fields) for row in holdouts}
    training = [row for row in original if tuple(row[name] for name in spec.key_fields) not in withheld]
    reduced = OperatorDataPackage.from_rows(package.meta, {"dsv4_attention": training})
    sol = DeepSeekV4AttentionSOL(model, device)
    policy = LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64)
    lookup = OperatorLookup(reduced, policy, analytical_scaler=lambda _, key: sol.work(key).latency_us, bind_analytical_axes=True)
    legacy = OperatorLookup(reduced, policy, analytical_scaler=legacy_work)
    records = []
    for row in holdouts:
        key = {name: row[name] for name in spec.key_fields}
        prediction, old = lookup.lookup("dsv4_attention", key), legacy.lookup("dsv4_attention", key)
        assert prediction.match_type == "extrapolated"
        records.append({
            "key": key, "source_id": row["source_id"], "measured_us": row["latency_us"],
            "sol_predicted_us": prediction.latency_us, "legacy_predicted_us": old.latency_us,
            "sol_error_percent": 100 * (prediction.latency_us / row["latency_us"] - 1),
            "legacy_error_percent": 100 * (old.latency_us / row["latency_us"] - 1),
        })
    summaries = {}
    for phase in ("context", "generation"):
        for kind in ("csa", "hca"):
            rows = [r for r in records if r["key"]["phase"] == phase and r["key"]["layer_kind"] == kind]
            for tp in (1, 2, 4, 8):
                subset = [r for r in rows if r["key"]["tp_size"] == tp]
                if not subset:
                    continue
                summaries[f"{phase}:{kind}:tp{tp}"] = {"points": len(subset), **{
                    method: {
                        "mean_absolute_error_percent": statistics.mean(abs(r[f"{method}_error_percent"]) for r in subset),
                        "median_absolute_error_percent": statistics.median(abs(r[f"{method}_error_percent"]) for r in subset),
                        "max_absolute_error_percent": max(abs(r[f"{method}_error_percent"]) for r in subset),
                        "within_25_percent": sum(abs(r[f"{method}_error_percent"]) <= 25 for r in subset),
                    } for method in ("sol", "legacy")
                }}
    diagnostic = []
    for kind, ratio in (("csa", 4), ("hca", 128)):
        base = next(r for r in original if r["phase"] == "generation" and r["layer_kind"] == kind and r["tp_size"] == 1 and r["batch_size"] == 64 and r["context_len"] == 16384)
        for length in (16384, 131072, 1048576):
            key = {name: base[name] for name in spec.key_fields}
            key["context_len"] = length
            boundary = sol.work(base)
            work = sol.work(key)
            diagnostic.append({"kind": kind, "context": length, "work": work.to_dict(),
                "boundary_us": base["latency_us"], "sol_growth_prediction_us": base["latency_us"] * work.latency_us / boundary.latency_us,
                "legacy_growth_prediction_us": base["latency_us"] * legacy_work(None, key) / legacy_work(None, base)})
    return {
        "source_rows": len(original), "training_rows": len(training), "holdout_rows": len(holdouts),
        "strategy": "withhold maximum context per exact discrete/batch/query geometry; no fitting or row cleaning",
        "scope": "source-layout extrapolation; no FlashMLA-to-FlashInfer multiplier or q=4 KV reuse correction",
        "sol_provenance": sol.describe(), "summary": summaries, "diagnostic": diagnostic, "holdouts": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = HardwareContext.load(ROOT / "data")
    report = {sku: audit(sku, context) for sku in ("b200", "b300")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
