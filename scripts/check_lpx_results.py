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
from scripts.lpx.config import load_cases
from scripts.lpx.export import write_evidence_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--original-workload", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args(argv)
    try:
        destinations = ([args.output] if args.output else []) + (
            [args.export / name for name in ("evidence.json", "points.csv")]
            if args.export
            else []
        )
        resolved = [path.resolve() for path in destinations]
        if args.summary.resolve() in resolved or len(set(resolved)) != len(resolved):
            raise ConfigurationError(
                "audit outputs must be distinct from source and each other"
            )
        if args.cases and not args.config:
            raise ConfigurationError("--cases requires --config")
        cases, batches = None, args.batches
        if args.config:
            raw, _, cases = load_cases(args.config)
            if args.cases:
                unknown = set(args.cases) - {case.name for case in cases}
                if unknown:
                    raise ConfigurationError(f"unknown case names: {sorted(unknown)}")
                cases = [case for case in cases if case.name in args.cases]
            if batches is None:
                batches = raw["batches"]
        source_bytes = args.summary.read_bytes()
        payload = json.loads(source_bytes)
        report = validate_summary(
            payload,
            hardware=HardwareContext.load(ROOT / "data"),
            original_workload=args.original_workload,
            expected_cases=cases,
            expected_batches=batches,
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        if args.export:
            write_evidence_bundle(
                args.export,
                payload,
                report,
                source_bytes=source_bytes,
                source_name=args.summary.name,
            )
        print(text, end="")
        return 0
    except (ConfigurationError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"LPX result audit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
