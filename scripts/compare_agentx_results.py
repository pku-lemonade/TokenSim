"""Compare a frozen AgentX before/after matrix against archived hardware points."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.experiments.agentx_comparison import compare_manifest
from TokenSim.errors import ConfigurationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-no-regression", action="store_true")
    args = parser.parse_args()
    try:
        report = compare_manifest(args.manifest)
    except (OSError, ValueError, KeyError, TypeError, ConfigurationError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for case in report["cases"]:
        print(
            case["hardware"],
            case["status"],
            "critical regressions:",
            case.get("critical_regressions", []),
        )
    if report["status"] != "valid":
        return 2
    return int(
        args.require_no_regression
        and any(case["critical_regressions"] for case in report["cases"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
