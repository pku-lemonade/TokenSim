"""Analyze a collected T6 trace using explicit, hash-bound decode selections.

This tool does not infer decode boundaries or semantic kernel labels from names.
The output is a GPU wall-time partition, not a completed T6 attribution.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.lpx.gpu_profile import (
    file_record,
    read_profile_turn,
    read_trace_events,
    verify_records,
)
from scripts.lpx.gpu_trace import summarize_decode
from scripts.run_lpx_gpu_study import read_json, write_json


def analyze(case_path, selection_path):
    case = read_json(case_path)
    selection = read_json(selection_path)
    if (
        not isinstance(case, dict)
        or case.get("status") != "collected"
        or case.get("mode") not in ("graphs", "eager")
        or case.get("context") not in (10000, 100000)
        or len(case.get("outputs", [])) != 2
        or not case.get("traces")
    ):
        raise WorkloadValidationError("Expected a collected T6 case manifest")
    if not isinstance(selection, dict) or not isinstance(selection.get("trace"), str):
        raise WorkloadValidationError("Selection must identify its trace file")
    trace = Path(selection["trace"]).resolve()
    matches = [
        record for record in case["traces"] if Path(record["path"]).resolve() == trace
    ]
    if len(matches) != 1 or matches[0]["sha256"] != selection.get("trace_sha256"):
        raise WorkloadValidationError(
            "Selection trace or hash differs from collected case"
        )
    verify_records([*case["outputs"], *case["traces"]])
    measured, turn = read_profile_turn(
        Path(case["outputs"][1]["path"]), case["context"]
    )
    for field in ("requests_sha256", "environment_sha256"):
        if measured["config"][field] != case.get(field):
            raise WorkloadValidationError(
                "Case input identity differs from profiled output"
            )
    result = summarize_decode(
        read_trace_events(trace), selection, turn["completion_tokens"] - 1
    )
    result.update(
        mode=case["mode"],
        context=case["context"],
        case=file_record(case_path),
        selection=file_record(selection_path),
        trace=matches[0],
        profiled_output=case["outputs"][1],
        control_timing=case.get("timing"),
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.case, args.selection)
    case = read_json(args.case)
    inputs = [args.case, args.selection]
    inputs.extend(
        Path(record["path"]) for record in [*case["outputs"], *case["traces"]]
    )
    if args.summary.resolve() in {path.resolve() for path in inputs}:
        raise WorkloadValidationError("Summary must not overwrite a profiling input")
    write_json(args.summary, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
