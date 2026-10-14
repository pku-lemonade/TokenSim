"""Plot the LPX manual reference, simulation and comparison from saved results."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lpx.plotting import generate_plots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "outputs/lpx-reproduction/summary.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    for path in generate_plots(args.summary, args.output, args.reference):
        print(path)


if __name__ == "__main__":
    main()
