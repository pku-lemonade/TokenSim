"""Plot current complete K3/K0 ablation metrics; no K0 reference band."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


STAGE = Path(__file__).resolve().parent
report = json.loads((STAGE / "paired-analysis.json").read_text())
assert set(report["points"]) == {"b200-main", "b300-main", "b200-disabled", "b300-disabled"}
metrics = (("output_tps", "Output throughput", "token/s", 1),
           ("ttft_p90_s", "p90 TTFT", "seconds", 1),
           ("itl_p90_s", "p90 ITL (TPOT)", "ms", 1000),
           ("e2e_mean_s", "Mean end-to-end", "seconds", 1),
           ("gpu_hit_fraction", "GPU input hit rate", "%", 100),
           ("cpu_hit_fraction", "CPU input hit rate", "%", 100))
series = (("main", "K3 / AL2.49", "#c87435", -.14),
          ("disabled", "K0 / AL1", "#347baa", .14))
figure, axes = plt.subplots(2, 3, figsize=(13, 8))
for axis, (name, title, unit, scale) in zip(axes.flat, metrics, strict=True):
    largest = 0
    for index, sku in enumerate(("b200", "b300")):
        for mode, label, color, offset in series:
            point = report["points"][f"{sku}-{mode}"]
            assert point["terminal"]["exit_code"] == 0
            value = point["metrics"][name]["current"] * scale
            largest = max(largest, value)
            axis.bar(index + offset, value, width=.25, color=color)
        change = report["ablation"][sku]["metrics"][name]["k0_change_percent"]
        value = report["points"][f"{sku}-disabled"]["metrics"][name]["current"] * scale
        axis.annotate(f"{change:+.1f}%", (index + .14, value), xytext=(0, 5),
                      textcoords="offset points", ha="center", fontsize=9)
    axis.set_title(title)
    axis.set_ylabel(unit)
    axis.set_xticks((0, 1), ("B200 c196", "B300 c384"))
    axis.set_ylim(0, min(100, largest * 1.2) if scale == 100 else largest * 1.2)
    axis.grid(axis="y", alpha=.18)
figure.suptitle("Explicit MTP execution: complete seed0 K3/K0 windows, 8 GPUs", fontsize=14)
figure.legend(handles=[Patch(color=color, label=label) for mode, label, color, offset in series],
              loc="upper center", bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
figure.text(.5, .02, "Labels: K0 change versus current K3. Disabled ablations have no independent reference identity; absolute accuracy remains unaccepted.",
            ha="center", fontsize=9)
figure.tight_layout(rect=(0, .055, 1, .89))
for suffix in ("png", "svg"):
    path = STAGE / f"paired-comparison.{suffix}"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    if suffix == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path)
