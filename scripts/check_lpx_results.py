"""Audit saved native LPX results without rerunning the simulation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError
from scripts.lpx.validation import validate_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--original-workload", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.summary.read_text())
        report = validate_summary(
            payload,
            hardware=HardwareContext.load(ROOT / "data"),
            original_workload=args.original_workload,
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text, end="")
        return 0
    except (ConfigurationError, KeyError, TypeError, ValueError) as exc:
        print(f"LPX result audit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
