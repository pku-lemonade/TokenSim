"""Sweep virtual 1A x NF DeepSeek-V3 decode points and plot throughput curves."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v3-virtual-1aN-20260919"
F_COUNTS = (2, 4, 8, 16, 32)
BATCHES = (1, 4, 16, 32)
POWER_W_PER_CARD = 250.0


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_case(f_count: int) -> list[dict]:
    case_dir = OUT / f"f{f_count}"
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        command = [
            sys.executable, "-m", "scripts.run_deepseek_mla", "--full-model",
            "--a-tp", "1", "--f-tp", str(f_count), "--af-batch-size", "8",
            "--batches", *(str(x) for x in BATCHES), "--context", "2048",
            "--steps", "1", "--hypothetical-f-memory-gib", "1536",
            "--fallback", "analytical_only", "--output", str(case_dir),
        ]
        (case_dir / "command.json").parent.mkdir(parents=True, exist_ok=True)
        (case_dir / "run.log").write_text("$ " + " ".join(command) + "\n", encoding="utf-8")
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (case_dir / "run.log").write_text(
            "$ " + " ".join(command) + "\n" + result.stdout + result.stderr,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(f"F={f_count} failed; see {case_dir/'run.log'}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in payload["results"]:
        power_w = (1 + f_count) * POWER_W_PER_CARD
        rows.append({
            "f_gpus": f_count,
            "a_gpus": 1,
            "batch_size": row["batch_size"],
            "context_len": row["context_len"],
            "af_batch_size": payload["af_batch_size"],
            "avg_step_us": row["avg_step_us"],
            "tokens_per_s": row["tokens_per_s"],
            "tps_user": row["tokens_per_s_per_user"],
            "power_w": power_w,
            "tps_mw": row["tokens_per_s"] / power_w * 1e6,
            "profile": str(case_dir / row["profile"]),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row for f_count in F_COUNTS for row in run_case(f_count)]
    write_json(OUT / "experiment.json", {
        "model": "DeepSeek-V3 671B BF16",
        "topology": "1A x NF",
        "f_counts": list(F_COUNTS),
        "batches": list(BATCHES),
        "context_len": 2048,
        "af_batch_size": 8,
        "steps": 1,
        "power_w_per_card": POWER_W_PER_CARD,
        "virtual_f_memory_gib_per_worker": 1536,
        "timing": "CPU operator-table simulation; no GPU execution",
    })
    with (OUT / "sweep-data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(OUT / "sweep-data.json", rows)
    plot(rows)


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green", 16: "tab:red", 32: "tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    for f_count in F_COUNTS:
        series = sorted((r for r in rows if r["f_gpus"] == f_count), key=lambda r: r["batch_size"])
        color = colors[f_count]
        axes[0].plot([r["batch_size"] for r in series], [r["tps_user"] for r in series], "o-", color=color, label=f"1A{f_count}F")
        points = sorted(series, key=lambda r: r["tps_user"])
        axes[1].plot([r["tps_user"] for r in points], [r["tps_mw"] for r in points], "o-", color=color, label=f"1A{f_count}F")
    axes[0].set(xlabel="Concurrent users (batch)", ylabel="Interactivity (TPS/user)", title="DeepSeek-V3 decode: F-count sweep")
    axes[0].set_xscale("log", base=2); axes[0].grid(alpha=.25)
    axes[1].set(xlabel="Interactivity (TPS/user)", ylabel="Throughput (TPS/MW)", title="Figure 11-style view")
    axes[1].grid(alpha=.25)
    for ax in axes: ax.legend(fontsize=8); ax.margins(x=.08, y=.08)
    fig.suptitle("Virtual 1A×NF | 671B BF16 | 2048 cached context | AF block 8", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "f-count-sweep.png", bbox_inches="tight")
    fig.savefig(OUT / "f-count-sweep.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
