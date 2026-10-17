"""Plot saved QPS=1 AF scan results without importing or running TokenSim.

Derived from plot_results() in the archived qps1-20261015 launch_scan.py.
Requires: python -m pip install matplotlib
SVG export uses stable element identifiers and omits the export timestamp.
"""

import argparse
import json
import math
from pathlib import Path


POWER_MW = 0.0144  # 36 GPUs * 400 W / 1,000,000; assumed GPU power only.
EXPECTED_POINTS = 8
AXES = (
    (
        "active",
        "Output TPS / time-weighted mean active users (Prefill + Decode)",
        "pareto-tps-user-vs-tps-mw",
    ),
    (
        "decode",
        "Decode interactivity: 1 / mean adjacent TBT (tokens/s)",
        "pareto-decode",
    ),
    (
        "registered",
        "Output TPS / registered user pool (supplementary)",
        "pareto-registered-users",
    ),
)


def number(row, key, *, positive=False):
    value = float(row[key])
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {key}: {value}")
    return value


def load_points(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Input must be a nonempty results.json list of completed points"
        )
    if len(rows) > EXPECTED_POINTS:
        raise ValueError("This plot expects the fixed eight-point QPS=1 scan")
    points = []
    for row in rows:
        users = number(row, "users", positive=True)
        if users not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError("Unexpected registered user count")
        batch = number(row, "max_batch", positive=True)
        microbatch = number(row, "microbatch", positive=True)
        if batch != min(users, 32) or microbatch != min(users, 8):
            raise ValueError("Batch settings differ from the fixed scan")
        tps = number(row, "window_tps")
        active = number(row, "mean_active_users")
        if active > users + 1e-9 or (active == 0 and tps > 0):
            raise ValueError("Invalid mean active-user count")
        for key, expected in (
            ("qps", 1),
            ("warmup_s", 100),
            ("measurement_s", 200),
            ("power_w", 14400),
        ):
            if key in row and number(row, key) != expected:
                raise ValueError(f"This plot requires {key}={expected}")
        tbt = row.get("mean_tbt_ms")
        decode = (
            None if tbt is None else 1000 / number(row, "mean_tbt_ms", positive=True)
        )
        point = dict(
            users=int(users),
            batch=int(batch),
            microbatch=int(microbatch),
            mean_active_users=active,
            window_tps=tps,
            active=tps / active if active > 0 else None,
            decode=decode,
            registered=tps / users,
            tps_mw=tps / POWER_MW,
        )
        for source, target in (
            ("x_active_tps_user", "active"),
            ("x_decode_tps_user", "decode"),
            ("x_app_tps_user", "registered"),
            ("tps_mw", "tps_mw"),
        ):
            if row.get(source) is not None:
                if point[target] is None or not math.isclose(
                    number(row, source), point[target], rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(f"Saved {source} disagrees with its definition")
        points.append(point)
    if len({p["users"] for p in points}) != len(points):
        raise ValueError("Duplicate user counts; do not mix repeated seeds or runs")
    return sorted(points, key=lambda p: p["users"])


def frontier(points, x):
    """Observed nondominated points when both axes are maximized."""
    return sorted(
        [
            p
            for p in points
            if not any(
                q[x] >= p[x]
                and q["tps_mw"] >= p["tps_mw"]
                and (q[x] > p[x] or q["tps_mw"] > p["tps_mw"])
                for q in points
            )
        ],
        key=lambda p: p[x],
    )


def plot(points, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    files = []
    if len(points) < 2:
        raise ValueError("At least two completed points are required for a curve")
    if sum(p["active"] is not None for p in points) < 2:
        raise ValueError("At least two points need a defined active-user denominator")
    output.mkdir(parents=True, exist_ok=True)
    footer = (
        "DeepSeek-V3 671B MLA, 61 layers BF16; 4A32F; analytical simulation.\n"
        "400K cached + 4K new + 400 output; 404K extrapolates beyond native 163,840 context.\n"
        "A 4096 GiB / F 1536 GiB virtual capacity; A100 compute/HBM assumptions.\n"
        "36 x 400 W = 0.0144 MW assumed GPU budget; excludes host/network; not measured efficiency.\n"
        "Window [100,300) s; QPS=1 provisional, uncalibrated; finite window, convergence unconfirmed.\n"
        "Observed points and frontier only; not an NVIDIA numerical reproduction."
    )
    for x, label, stem in AXES:
        valid = [p for p in points if p[x] is not None]
        if len(valid) < 2:
            continue
        front = frontier(valid, x)
        fig, ax = plt.subplots(figsize=(10, 7.2))
        ax.scatter(
            [p[x] for p in valid],
            [p["tps_mw"] for p in valid],
            label="Recorded finite-window points",
        )
        ax.plot(
            [p[x] for p in front],
            [p["tps_mw"] for p in front],
            "o-",
            label="Observed Pareto frontier",
        )
        for p in valid:
            ax.annotate(
                f'Ccap={p["users"]}, active={p["mean_active_users"]:.2f}\n'
                f'B{p["batch"]} m{p["microbatch"]}',
                (p[x], p["tps_mw"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        suffix = "; partial points" if len(valid) < EXPECTED_POINTS else ""
        ax.set(
            xlabel=label,
            ylabel="Output tokens/s/MW (assumed GPU budget)",
            title=f"QPS=1 (provisional); {len(valid)}/8 recorded points{suffix}",
        )
        ax.grid(alpha=0.25)
        ax.margins(x=0.12, y=0.16)
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.text(0.5, 0.015, footer, ha="center", fontsize=7)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        for extension in ("png", "svg"):
            path = output / f"{stem}.{extension}"
            if extension == "svg":
                with matplotlib.rc_context({"svg.hashsalt": f"tokensim-qps1-{stem}"}):
                    fig.savefig(path, dpi=180, metadata={"Date": None})
            else:
                fig.savefig(path, dpi=180)
            files.append(path)
        plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved results.json")
    parser.add_argument("--output", type=Path, required=True, help="Figure directory")
    args = parser.parse_args(argv)
    try:
        points = load_points(args.input)
        for path in plot(points, args.output):
            print(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
