"""Test, without adopting it, whether one constant can explain work ratios."""

import json
import math
from pathlib import Path


STAGE = Path(__file__).resolve().parent
source = json.loads((STAGE / "work-growth-followup.json").read_text())


def lower_bound(interval, a, b):
    """Intersect a*c >= b with an interval for c >=0."""
    low, high = interval
    if a > 1e-12:
        low = max(low, b / a)
    elif a < -1e-12:
        high = min(high, b / a)
    elif b > 1e-12:
        low, high = 1.0, 0.0
    return low, high


def serialize(interval):
    low, high = interval
    return {"low_work_us": low, "high_work_us": high if math.isfinite(high) else None,
            "feasible": low <= high, "unbounded_high": not math.isfinite(high)}


results = {}
combined = (0.0, float("inf"))
for sku, hardware in source["hardware"].items():
    samples = {(row["phase"], row["num_tokens"]): row for row in hardware["uniform_samples"]}
    phases = {}
    all_pairs = []
    for pair in hardware["directed_same_effective_capacity_holdouts_within8x"]:
        phase = pair["phase"]
        a = samples[phase, pair["anchor_rows"]]
        t = samples[phase, pair["holdout_rows"]]
        ratio = t["latency_us"] / a["latency_us"]
        lower, upper = 0.75 * ratio, 1.25 * ratio
        # L <= (Wt+c)/(Wa+c) <= U, denominator positive for c>=0.
        inequalities = ((1 - lower, lower * a["uniform_work_us"] - t["uniform_work_us"]),
                        (upper - 1, t["uniform_work_us"] - upper * a["uniform_work_us"]))
        interval = (0.0, float("inf"))
        for coefficient, bound in inequalities:
            interval = lower_bound(interval, coefficient, bound)
        previous = phases.get(phase, (0.0, float("inf")))
        phases[phase] = (max(previous[0], interval[0]), min(previous[1], interval[1]))
        all_pairs.append({"phase": phase, "anchor_rows": pair["anchor_rows"],
                          "heldout_rows": pair["holdout_rows"],
                          "same_requested_capacity": pair["same_requested_capacity"],
                          "residual_interval": serialize(interval)})
    whole = (max(interval[0] for interval in phases.values()),
             min(interval[1] for interval in phases.values()))
    combined = (max(combined[0], whole[0]), min(combined[1], whole[1]))
    results[sku] = {"phases": {phase: serialize(interval) for phase, interval in phases.items()},
                    "all_phases": serialize(whole), "pairs": all_pairs}

report = {
    "scope": "constant-residual ratio feasibility on original uniform timing rows",
    "runtime_change": False, "selected_parameter": None,
    "formal_accuracy_evidence": False, "hardware": results,
    "common_across_hardware": serialize(combined),
    "formula": "anchor_measured_us*(heldout_work_us+c)/(anchor_work_us+c), c>=0; intersect ±25% constraints",
    "limits": [
        "c is a diagnostic value in analytical-work units, not a measured kernel startup time",
        "constraints use original measured holdouts to reject or bound an explanation, not to validate a fitted predictor",
        "a feasible interval would not identify startup, occupancy, cache traffic, overlap or nonuniform routing as its cause",
        "no value chosen or applied; serving reference throughput is absent from these equations",
    ],
}
destination = STAGE / "residual-feasibility-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, result in results.items():
    print(sku, result["phases"], result["all_phases"])
print("Common", report["common_across_hardware"])
