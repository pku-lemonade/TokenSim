"""Render recorded uniform holdout errors; no runtime or data fitting."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / "work-growth-followup.json").read_text())
figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for axis, sku in zip(axes, ("b200", "b300"), strict=True):
    hardware = report["hardware"][sku]
    pairs = hardware["directed_same_effective_capacity_holdouts_within8x"]
    axis.axhspan(-25, 25, color="#dbece1", alpha=0.65)
    axis.axhline(0, color="#536171", linewidth=0.8)
    for phase, color in (("generation", "#ca6d2d"), ("context", "#356da3")):
        selected = [pair for pair in pairs if pair["phase"] == phase]
        if selected:
            axis.scatter([pair["holdout_rows"] / pair["anchor_rows"] for pair in selected],
                         [pair["error_percent"] for pair in selected],
                         color=color, label=phase, alpha=0.7, s=34)
    axis.set_xscale("log", base=2)
    ticks = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]
    axis.set_xticks(ticks, ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    axis.set_xlim(0.105, 9.5)
    axis.set_ylim(-80, 80)
    axis.set_xlabel("Held-out rows / anchor rows")
    axis.grid(axis="y", alpha=0.2)
    count = hardware["directed_same_effective_capacity_count"]
    failed = hardware["directed_same_effective_capacity_failures_over25percent"]
    capacity = "requested + effective" if sku == "b200" else "effective; requested differs"
    axis.set_title(f"{sku.upper()}: {failed}/{count} beyond 25%\nCapacity matched: {capacity}", fontsize=11)
    largest = max(pairs, key=lambda pair: abs(pair["error_percent"]))
    axis.annotate(f"{largest['anchor_rows']} → {largest['holdout_rows']}: {largest['error_percent']:+.1f}%",
                  xy=(largest["holdout_rows"] / largest["anchor_rows"], largest["error_percent"]),
                  xytext=(1.0, 70), arrowprops={"arrowstyle": "-", "color": "#637284"}, fontsize=9)
    axis.legend(loc="lower left", frameon=False)
axes[0].set_ylabel("Predicted / measured latency error (%)")
figure.suptitle("Unfitted MegaMoE work normalization: uniform holdouts", fontsize=14)
figure.text(0.5, 0.02, "One original anchor × physical work ratio; ≤8× rows; green band ±25%. Analytical check, not serving accuracy.",
            ha="center", fontsize=9)
figure.tight_layout(rect=(0, 0.055, 1, 0.94))
for suffix in ("png", "svg"):
    destination = ROOT / f"work-growth-holdouts.{suffix}"
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    if suffix == "svg":
        destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")
    print(destination)
