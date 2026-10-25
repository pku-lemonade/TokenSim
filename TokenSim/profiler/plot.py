"""Export paired, profiler-style comparison timelines as PNG and SVG."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.reporting import _publish
from TokenSim.profiler.timeline import prepare_timeline, pack_tracks, bar_record


COLORS = {
    "GEMM": "#5688c7",
    "Attention": "#49b7b5",
    "MoE": "#b79f42",
    "Norm / elementwise": "#bd74ac",
    "Collective": "#8a79c9",
    "Copy / transfer": "#b7bdc8",
    "Unmatched / excluded": "#888888",
    "Known peer wait": "#d45445",
}


def _category(bar):
    if bar.status != "matched":
        return "Unmatched / excluded"
    if bar.category in {"transfer", "gpu_memcpy"}:
        return "Copy / transfer"
    if bar.operation in {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}:
        return "Collective"
    if bar.operation in {"qkv", "output", "ffn_up", "ffn_down", "lm_head", "router"}:
        return "GEMM"
    if "attention" == bar.operation or bar.operation == "attention_module":
        return "Attention"
    if "moe" in bar.operation:
        return "MoE"
    return "Norm / elementwise"


def _fit_label(text, width, renderer, properties):
    if width < 15:
        return ""

    def fits(value):
        return (
            renderer.get_text_width_height_descent(value, properties, False)[0] <= width
        )

    if fits(text):
        return text
    left, right = 0, len(text)
    while left < right:
        middle = (left + right + 1) // 2
        if fits(text[:middle] + "..."):
            left = middle
        else:
            right = middle - 1
    return text[:left] + "..." if left else ""


def _save(figure, path, dpi):
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, suffix=path.suffix, delete=False
        ) as file:
            temporary = Path(file.name)
        metadata = (
            {"Creator": "TokenSim", "Date": None} if path.suffix == ".svg" else None
        )
        figure.savefig(temporary, dpi=dpi, metadata=metadata, facecolor="white")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _draw_panel(ax, tracks, selection, renderer, properties, scale):
    from matplotlib.patches import Rectangle

    lower, upper = selection["start_us"], selection["end_us"]
    records = []
    for row, (_, bars) in enumerate(tracks):
        for bar in bars:
            shift = bar.translation_us
            left, right = max(lower, bar.start_us + shift), min(
                upper, bar.end_us + shift
            )
            color = COLORS[_category(bar)]
            if selection["detail"] == "module":
                ax.add_patch(
                    Rectangle(
                        (left / scale, row - 0.29),
                        (right - left) / scale,
                        0.58,
                        facecolor="#edf0f4",
                        edgecolor="#bdc3cc",
                        linewidth=0.45,
                    )
                )
            for a, b in bar.intervals:
                a, b = max(lower, a + shift), min(upper, b + shift)
                if b < a:
                    continue
                if a == b:
                    ax.plot(
                        [a / scale, a / scale],
                        [row - 0.29, row + 0.29],
                        color=color,
                        linewidth=0.7,
                    )
                else:
                    ax.add_patch(
                        Rectangle(
                            (a / scale, row - 0.29),
                            (b - a) / scale,
                            0.58,
                            facecolor=color,
                            edgecolor="white",
                            linewidth=0.25,
                            linestyle="--" if bar.status != "matched" else "-",
                        )
                    )
            for a, b in bar.waits:
                a, b = max(lower, a + shift), min(upper, b + shift)
                if b > a:
                    ax.add_patch(
                        Rectangle(
                            (a / scale, row - 0.29),
                            (b - a) / scale,
                            0.58,
                            facecolor=COLORS["Known peer wait"],
                            edgecolor="none",
                        )
                    )
            pixels = (
                ax.transData.transform((right / scale, 0))[0]
                - ax.transData.transform((left / scale, 0))[0]
            )
            label = _fit_label(bar.name, pixels - 10, renderer, properties)
            if label:
                gap_label = selection["detail"] == "module" and sum(
                    b - a for a, b in bar.intervals
                ) < 0.8 * (bar.end_us - bar.start_us)
                ax.text(
                    (left + right) / (2 * scale),
                    row,
                    label,
                    ha="center",
                    va="center",
                    color="#26374d" if gap_label else "white",
                    fontsize=9,
                    clip_on=True,
                    parse_math=False,
                )
            records.append(dict(bar_record(bar), row=row, rendered_label=label))
    return records


def _render(selection, output, formats, title, quality, width, dpi):
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise ConfigurationError(
            "image export requires pip install 'tokensim[plot]'"
        ) from error

    tracks = {
        side: pack_tracks(selection["bars"], side) for side in ("simulated", "profiled")
    }
    scale = 1000 if selection["end_us"] - selection["start_us"] >= 5000 else 1
    unit = "ms" if scale == 1000 else "us"
    height = 4.3 + 0.55 * sum(max(1, len(rows)) for rows in tracks.values())
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.usetex": False,
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.axisbelow": True,
        }
    ):
        figure = Figure(figsize=(width, height), dpi=dpi)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={
                "height_ratios": [max(1, len(rows)) + 0.5 for rows in tracks.values()]
            },
        )
        figure.subplots_adjust(
            left=0.15, right=0.985, top=0.78, bottom=0.26, hspace=1.25
        )
        figure.suptitle(title, x=0.15, y=0.97, ha="left", fontsize=15)
        for ax, (side, rows), caption in zip(
            axes,
            tracks.items(),
            (
                "(a) Generated trace with TokenSim",
                "(b) Profiled trace with PyTorch Profiler",
            ),
        ):
            ax.set_xlim(selection["start_us"] / scale, selection["end_us"] / scale)
            ax.set_ylim(max(1, len(rows)) - 0.4, -0.6)
            ax.set_yticks(range(len(rows)), [label for label, _ in rows])
            ax.tick_params(axis="y", length=0, labelsize=9)
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.grid(axis="x", color="#d9dde4", linewidth=0.7)
            for name, spine in ax.spines.items():
                spine.set_visible(name == "top")
                spine.set_color("#a7b2c3")
            ax.text(
                0.5, -0.38, caption, transform=ax.transAxes, ha="center", fontsize=14
            )
            if not rows:
                ax.text(
                    0.5,
                    0.5,
                    "No selected intervals",
                    transform=ax.transAxes,
                    ha="center",
                )
        axes[0].set_xlabel(f"Time ({unit})")
        axes[0].xaxis.set_label_position("top")
        axes[0].xaxis.labelpad = 17
        canvas.draw()
        renderer, properties = canvas.get_renderer(), FontProperties(size=9)
        records = []
        for ax, rows in zip(axes, tracks.values()):
            records.extend(
                _draw_panel(ax, rows, selection, renderer, properties, scale)
            )
        categories = {_category(bar) for bar in selection["bars"]}
        if any(bar.waits for bar in selection["bars"]):
            categories.add("Known peer wait")
        figure.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for label, color in COLORS.items()
                if label in categories
            ],
            loc="lower left",
            bbox_to_anchor=(0.15, 0.12),
            ncol=4,
            frameon=False,
        )
        alignment = (
            "Each side uses its selected origin; rank offsets on proven shared clocks retained."
            if selection["alignment"] == "selection"
            else "Comparison clock translations retained."
        )
        notes = [alignment + " Same time scale; gaps and kernel residence retained."]
        notes.extend(quality)
        notes.extend(selection["warnings"])
        figure.text(0.15, 0.025, "\n".join(notes), fontsize=9, va="bottom")
        files = []
        for suffix in formats:
            path = Path(str(output) + "." + suffix)
            _save(figure, path, dpi)
            files.append(str(path))
    return files, records


def export_timeline(
    comparison,
    output,
    *,
    formats=("png", "svg"),
    title=None,
    width=20,
    dpi=180,
    **selection_options,
):
    """Read a comparison report; publish paired images and their exact geometry."""
    if (
        not formats
        or isinstance(formats, str)
        or any(f not in {"png", "svg"} for f in formats)
    ):
        raise ConfigurationError("timeline formats must be a nonempty PNG/SVG sequence")
    if (
        type(width) not in (int, float)
        or not math.isfinite(width)
        or not 8 <= width <= 80
        or type(dpi) is not int
        or not 72 <= dpi <= 600
    ):
        raise ConfigurationError(
            "timeline width must be 8..80 inches and DPI must be 72..600"
        )
    comparison, output = Path(comparison), Path(output)
    raw = comparison.read_bytes()
    report = json.loads(raw)
    selection = prepare_timeline(report, **selection_options)
    quality = []
    measurements = report.get("measurement_quality", [])
    if not isinstance(measurements, list) or not all(
        isinstance(item, dict) for item in measurements
    ):
        raise ConfigurationError("measurement_quality must be an array of objects")
    for item in measurements:
        ratio = item.get("profile_to_baseline_ratio")
        if ratio is not None:
            if (
                type(ratio) not in (int, float)
                or not math.isfinite(ratio)
                or ratio <= 0
            ):
                raise ConfigurationError(
                    "collection perturbation ratio must be positive and finite"
                )
            note = f"Capture/baseline wall time: {ratio:.3f} ({item.get('status', 'unreported')})."
            if note not in quality:
                quality.append(note)
    if not quality:
        quality.append("Collection perturbation is unreported.")
    filters = selection["filters"]
    title = title or "Operator timeline comparison" + "".join(
        f" | {name} {filters[name]}"
        for name in ("forward", "layer")
        if filters[name] is not None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    files, records = _render(
        selection, output, tuple(dict.fromkeys(formats)), title, quality, width, dpi
    )
    manifest = dict(
        schema_version=1,
        comparison=str(comparison),
        comparison_sha256=hashlib.sha256(raw).hexdigest(),
        files=files,
        selection={key: value for key, value in selection.items() if key != "bars"},
        measurement_quality=report.get("measurement_quality", []),
        events=records,
    )
    _publish(
        Path(str(output) + ".timeline.json"),
        lambda stream: json.dump(manifest, stream, indent=2, allow_nan=False),
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="output filename stem"
    )
    parser.add_argument("--forward", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--rank", action="append", dest="ranks")
    parser.add_argument("--detail", choices=("kernel", "module"), default="kernel")
    parser.add_argument(
        "--alignment", choices=("global", "selection"), default="global"
    )
    parser.add_argument("--start-us", type=float)
    parser.add_argument("--end-us", type=float)
    parser.add_argument(
        "--formats", choices=("png", "svg"), nargs="+", default=("png", "svg")
    )
    parser.add_argument("--title")
    parser.add_argument("--width", type=float, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    args = vars(parser.parse_args(argv))
    try:
        result = export_timeline(**args)
    except (ConfigurationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(dict(files=result["files"], events=len(result["events"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
