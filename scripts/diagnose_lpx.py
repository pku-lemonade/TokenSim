"""Explain LPX projection differences using reproducible isolated probes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.hardware import HardwareContext
from scripts.lpx.config import load_cases
from scripts.lpx.diagnostics import latency_probes, saved_result_diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "outputs/lpx-reproduction/summary.json"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/sweep.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-diagnosis/diagnosis.json"
    )
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    _, _, cases = load_cases(args.config)
    if [asdict(case) for case in cases] != summary["cases"]:
        parser.error("diagnostic configuration must match saved case layouts")
    hardware = HardwareContext.load(ROOT / "data")
    model = hardware.model(summary["model"]["model_id"])
    context = (
        summary["workload"]["cached_tokens"] + summary["workload"]["new_tokens"] + 200
    )
    probes = [
        latency_probes(hardware, model, case, context=context)
        for case in cases
        if case.name in {"rubin-tp8", "af-tp8", "pd-cp1"}
    ]
    result = {
        "schema": "tokensim.lpx_diagnosis",
        "source_revision": summary["git_revision"],
        "latency_probes": probes,
        **saved_result_diagnostics(hardware, model, summary),
        "limitations": [
            "Zero launch and zero bandwidth ramp are isolated lower bounds, not measured hardware settings.",
            "The hybrid-window probe changes layer latency only; KV memory and placement require separate implementation.",
            "Delta handoff is valid only when the decode pool already retains the cached prefix; full-copy mode remains required for a cold destination.",
            "These probes leave catalog parameters, serving code and saved experiment results unchanged.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
