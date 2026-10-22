"""Compare a simulated operator trace with PyTorch Profiler Chrome JSON."""

import argparse
import hashlib
import json
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.compatibility import validate_inputs
from TokenSim.profiler.importer import read_trace
from TokenSim.profiler.matching import match_spans
from TokenSim.profiler.reporting import write_comparison


def compare_traces(simulated, profiles, output, *, mapping=None):
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise ConfigurationError("comparison mapping must be an object")
    aliases = mapping.get("aliases", {})
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in aliases.items()
    ):
        raise ConfigurationError("comparison aliases must map strings to strings")
    simulated = read_trace(simulated, simulated=True, aliases=aliases)
    profiled = read_trace(profiles, aliases=aliases)
    if not simulated.spans or not profiled.spans:
        raise ConfigurationError(
            "comparison requires nonempty simulated and GPU activity windows"
        )
    profiled.diagnostics["compatibility"] = validate_inputs(simulated, profiled)
    pairs, unmatched, excluded = match_spans(simulated.spans, profiled.spans, mapping)
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return write_comparison(
        simulated, profiled, pairs, unmatched, excluded, output, mapping_hash=digest
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulated", required=True, type=Path)
    parser.add_argument("--profile", required=True, nargs="+", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.mapping.read_text()) if args.mapping else None
        report = compare_traces(
            args.simulated, args.profile, args.output, mapping=mapping
        )
    except (ConfigurationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=report["status"], **report["coverage"])))
    return 0 if report["status"] == "compared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
