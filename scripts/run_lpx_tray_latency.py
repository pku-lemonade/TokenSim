"""Cross-tray latency sensitivity of the TP32/PP8 AA match on 256 LP30 chips.

A TP32 group spans two 16-chip trays, so its collectives use the middle fabric
tier, whose 0.75-microsecond latency comes from a Hot Chips reachable-capacity
chart and not from a measured collective.  This driver repeats the batch-one
article-condition point with other middle-tier latencies.  See
docs/lpx-tray-latency.md for the expectations and results.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.experiments import ExperimentRunner
from scripts.lpx.fidelity import load
from scripts.lpx.replay_profiles import source_identity
from scripts.run_lpx_article_sweep import analysis, article_config, run_point
from scripts.search_lpx_native import write_json

LAYOUTS = ("tp32-pp8-kvp8",)
ATTENTION_MFU = (0.3, 0.5)
MIDDLE_US = (0.5, 0.65, 0.75, 0.85, 1.0, 1.15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-article")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    base = article_config(load(args.config))
    runner = ExperimentRunner(
        args.output / "tray-latency-points",
        inputs=(ROOT / "data",),
        code_paths=(
            ROOT / "TokenSim",
            ROOT / "scripts/lpx",
            ROOT / "scripts/run_lpx_article_sweep.py",
            Path(__file__),
        ),
    )
    rows = []
    for middle in MIDDLE_US:
        config = {**base, "fabric": {**base["fabric"], "middle_us": middle}}
        for name in LAYOUTS:
            for mfu in ATTENTION_MFU:
                result = run_point(runner, config, name, mfu, 1, "tray-latency")
                rows.append(analysis(result) | dict(middle_us=middle))
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                assumptions=[
                    "Article-condition sweep settings except the middle-tier latency.",
                    "One AA request per forward without speculative decoding.",
                ],
                rows=rows,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
