"""Publication-style plots with separate digitized and simulated provenance."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"rubin": "#438b7a", "draft": "#5cb6a4", "af": "#4e7827", "pd": "#89b638"}
LABELS = {
    "rubin": "Vera Rubin",
    "draft": "Rubin verifier + LP30 external draft",
    "af": "Rubin attention + LP30 FFN",
    "pd": "Rubin prefill + LP30 decode",
}


def style_axis(ax, x_max=8000, y_max=1.02):
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#999999")
    ax.tick_params(length=0, pad=8, labelsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}x"))
    ax.set_xlabel("Interactivity\n(TPS / User)", fontsize=11, labelpad=10)
    ax.set_ylabel("Throughput\n(TPS / MW, normalized)", fontsize=11, labelpad=12)


def save(fig, output, name):
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=160, facecolor="white")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def draw_reference(ax, curves, labels=False):
    for mode, points in curves.items():
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=COLORS[mode],
            linewidth=2.5,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=LABELS[mode],
        )
    if not labels:
        return
    for mode, x, y, text in (
        ("rubin", 120, 0.385, "Vera Rubin"),
        ("draft", 960, 0.86, "Vera Rubin Verifier +\nLP30 External-Draft"),
        ("af", 2860, 0.47, "Vera Rubin ATTN\n+ LP30 FFN"),
        ("pd", 6310, 0.18, "Vera Rubin Prefill + LP30\nDecode"),
    ):
        ax.text(
            x,
            y,
            text,
            color=COLORS[mode],
            fontsize=11,
            fontweight="bold",
            ha="left" if mode in {"rubin", "draft"} else "center",
        )
    arrows = (
        (1000, 0.29, 1000, 0.77, "3X"),
        (1000, 0.29, 3300, 0.29, "3X"),
        (1540, 0.085, 7050, 0.085, "5X"),
    )
    for x0, y0, x1, y1, label in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", linestyle=(0, (4, 4)), color="#888888", lw=1.3
            ),
        )
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(
            x,
            y,
            label,
            color="#777777",
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=2),
        )


def draw_simulation(ax, summary, marker="o", legend=True):
    for mode in COLORS:
        points = summary["frontiers"][mode]
        if points:
            ax.plot(
                [p["tokens_per_s_per_user"] for p in points],
                [p["normalized_throughput"] for p in points],
                color=COLORS[mode],
                linewidth=2.1,
                marker=marker,
                markersize=4.5,
                label=LABELS[mode],
            )
    if legend:
        ax.legend(loc="upper right", frameon=False, fontsize=9)


def generate_plots(summary_path, output=None, reference_path=None):
    summary_path = Path(summary_path)
    output = Path(output or summary_path.parent)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text())
    reference = json.loads(
        Path(reference_path or ROOT / "experiments/lpx/reference.json").read_text()
    )
    if (
        summary.get("schema") != "tokensim.lpx_projection"
        or reference.get("kind") != "approximate_manual_digitization"
    ):
        raise ValueError(
            "plots require TokenSim summary and explicitly digitized reference"
        )
    workload = summary["workload"]
    fidelity = (
        "Sampled native-DAG replay; numerical accuracy requires separate native calibration."
        if summary.get("executor") == "sampled_native_dag_replay"
        else "Native per-pass operator DAGs."
    )
    caption = (
        f"{summary['model']['model_id']} | Cached ISL = {workload['cached_tokens']:,} | "
        f"New ISL = {workload['new_tokens']:,} | OSL = {workload['output_tokens']:,}"
    )
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(12.8, 7.45))
        fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.75)
        fig.text(
            0.015,
            0.955,
            "LPU Boosts Vera Rubin Inference Throughput and Interactivity",
            fontsize=22,
        )
        fig.text(
            0.015,
            0.905,
            "Disaggregation benefits enhanced at large context agentic workloads",
            fontsize=13,
            color="#777777",
        )
        fig.text(
            0.57,
            0.82,
            "GPT-OSS-2T  |  Cached ISL = 400K / New ISL = 4K / OSL = 400",
            fontsize=12,
            ha="center",
            fontweight="bold",
        )
        style_axis(ax)
        ax.set_ylabel(
            "Throughput\n(TPS / MW)", rotation=0, ha="right", va="center", labelpad=26
        )
        draw_reference(ax, reference["curves"], labels=True)
        fig.text(
            0.15,
            0.025,
            "Approximate reconstruction of NVIDIA Figure 11 from the supplied image; manual samples, not simulator output.",
            fontsize=9,
            color="#666666",
        )
        save(fig, output, "reference")

        fig, ax = plt.subplots(figsize=(12.8, 7.45))
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.2, top=0.8)
        valid = [r for r in summary["results"] if r["status"] == "ok"]
        x_max = max(1000, max(r["tokens_per_s_per_user"] for r in valid) * 1.15)
        y_max = max(1.02, max(r["normalized_throughput"] for r in valid) * 1.05)
        style_axis(ax, x_max, y_max)
        draw_simulation(ax, summary)
        fig.suptitle(
            "TokenSim projection: throughput and interactivity", fontsize=22, y=0.95
        )
        fig.text(0.5, 0.865, caption, ha="center", fontsize=11)
        zoom = ax.inset_axes([0.47, 0.35, 0.48, 0.26])
        draw_simulation(zoom, summary, legend=False)
        zoom.set_xlim(0, x_max)
        zoom.set_ylim(0, 0.12)
        zoom.spines[["top", "right"]].set_visible(False)
        zoom.tick_params(labelsize=8)
        zoom.set_title("Efficiency range 0–0.12x", fontsize=9, loc="left")
        fig.text(
            0.11,
            0.075,
            fidelity + " All allocated processors charged; peak Rubin normalization.",
            fontsize=9,
            color="#666666",
        )
        fig.text(
            0.11,
            0.045,
            "Warm prefix excluded from prefill work; suffix prefill, KV handoff and draft verification included in output throughput.",
            fontsize=9,
            color="#666666",
        )
        save(fig, output, "simulation")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.29, top=0.79, wspace=0.2)
        for ax in axes:
            style_axis(ax, max(8000, x_max), y_max)
        axes[0].set_title(
            "NVIDIA Figure 11: approximate manual samples", fontsize=12, pad=15
        )
        draw_reference(axes[0], reference["curves"])
        axes[1].set_title(
            "TokenSim: "
            + (
                "sampled native-DAG replay"
                if summary.get("version") == 3
                else "native analytical operators"
            ),
            fontsize=12,
            pad=15,
        )
        draw_simulation(axes[1], summary, legend=False)
        fig.suptitle(
            "LPX long-context figure reproduction and simulation", fontsize=20, y=0.96
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.08),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        fig.text(
            0.5,
            0.04,
            "Each panel has its own normalization. NVIDIA's model, placement and power inputs are unpublished; simulated 3x/5x gains are not established.",
            ha="center",
            fontsize=10,
            color="#666666",
        )
        save(fig, output, "comparison")
    return [
        output / f"{name}.png" for name in ("reference", "simulation", "comparison")
    ]
