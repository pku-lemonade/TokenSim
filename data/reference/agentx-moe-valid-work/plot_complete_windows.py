"""Plot the paired validated metrics with separate reference bands per SKU."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


STAGE = Path(__file__).resolve().parent
report = json.loads((STAGE / "paired-analysis.json").read_text())
metrics = (("output_tps", "Output throughput", "token/s", 1),
           ("ttft_p90_s", "p90 TTFT", "seconds", 1),
           ("itl_p90_s", "p90 ITL (TPOT)", "seconds", 1),
           ("e2e_mean_s", "Mean end-to-end", "seconds", 1),
           ("gpu_hit_fraction", "GPU input hit rate", "%", 100),
           ("cpu_hit_fraction", "CPU input hit rate", "%", 100))
series = (("reference", "Reference", "#7b8692"),
          ("baseline", "c019302", "#356da3"),
          ("current", "Full/valid work", "#ca6d2d"))
figure, axes = plt.subplots(2, 3, figsize=(13, 8))
for axis, (key, title, unit, scale) in zip(axes.flat, metrics, strict=True):
    largest = 0
    for index, sku in enumerate(("b200", "b300")):
        point = report["hardware"][sku]["metrics"][key]
        target = scale * point["reference"]
        upper_band = min(100, target * 1.25) if scale == 100 else target * 1.25
        axis.add_patch(Rectangle((index - 0.44, target * 0.75), 0.88, upper_band - target * 0.75,
                                 facecolor="#dbece1", edgecolor="none", zorder=0))
        for offset, (field, _label, color) in zip((-0.25, 0, 0.25), series, strict=True):
            value = scale * point[field]
            axis.bar(index + offset, value, width=0.21, color=color, zorder=2)
            largest = max(largest, value, upper_band)
        axis.annotate(f"{point['reference_error_percent']:+.1f}%", (index + 0.25, scale * point["current"]),
                      xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    axis.set_title(title)
    axis.set_xticks((0, 1), ("B200 c196", "B300 c384"))
    axis.set_ylabel(unit)
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(0, min(100, largest * 1.22) if scale == 100 else largest * 1.22)
    axis.grid(axis="y", alpha=0.18, zorder=1)
figure.suptitle("MegaMoE full/valid work: complete seed0 windows, 8 GPUs", fontsize=14)
figure.legend(handles=[Patch(color=color, label=label) for _field, label, color in series]
              + [Patch(color="#dbece1", label="Reference ±25%")],
              loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False)
figure.text(0.5, 0.015, "Labels: current reference error. Initial composition is grade D; serving accuracy remains unaccepted.",
            ha="center", fontsize=9)
figure.tight_layout(rect=(0, 0.05, 1, 0.88))
for suffix in ("png", "svg"):
    path = STAGE / f"paired-comparison.{suffix}"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    if suffix == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path)
