#!/usr/bin/env python3
"""Show held-out attention predictions without hiding large errors."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    colors = {1: "#2563eb", 2: "#059669", 4: "#d97706", 8: "#9333ea"}
    for row, sku in enumerate(("b200", "b300")):
        for col, (phase, kind) in enumerate((("context", "csa"), ("context", "hca"), ("generation", "csa"), ("generation", "hca"))):
            ax = axes[row, col]
            ax.axhspan(-25, 25, color="#dcfce7", alpha=0.6)
            ax.axhline(0, color="#475569", linewidth=0.7)
            for tp in (1, 2, 4, 8):
                points = [x for x in report[sku]["holdouts"] if x["key"]["phase"] == phase and x["key"]["layer_kind"] == kind and x["key"]["tp_size"] == tp]
                ax.scatter([p["measured_us"] for p in points], [p["sol_error_percent"] for p in points], s=10, alpha=0.6, color=colors[tp], label=f"TP{tp}")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{sku.upper()} {phase} {kind.upper()}")
            ax.set_xlabel("Held-out measured module latency (us)")
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel("Prediction error (%)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Physical SOL extrapolation: longest context held out per batch/query shape\nGreen band: +/-25%; all held-out rows retained; no fitted multiplier", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)
