#!/usr/bin/env python3
"""Validate complete windows with synthetic global integer/critical spectra."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_agentx_moe_mma import audit as audit_mma


BASELINE = Path("results/agentx-dsv4-full/native-moe-mma/20261008-dynamic-tails-pipe")
EVIDENCE = Path("data/reference/agentx-moe-mma/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/latency/deepseek_v4_moe_work.py",
                  "TokenSim/latency/deepseek_v4_moe_spectrum.py"}


def audit(root, *, hardware=("b200", "b300"), baseline_root=BASELINE,
          baseline_evidence=EVIDENCE, source_changes=SOURCE_CHANGES, cache_budget_auditor=None):
    report = audit_mma(root, hardware=hardware, baseline_root=baseline_root,
                       baseline_evidence=baseline_evidence, source_changes=source_changes,
                       cache_budget_auditor=cache_budget_auditor)
    for point in report["checkpoints"].values():
        description = point["moe_description"]["work_composition"]
        assert description["expert_spectrum"].startswith("global power_law_1.01 integer spectra")
        assert "synthetic, not recovered historical routes" in description["expert_spectrum"]
        assert description["spectrum_samples"] == [0, 1, 2]
        assert "all cyclic hot-rank placements" in description["spectrum_placement"]
        assert "no mean-row module substitution" in description["spectrum_placement"]
        assert "exact integer selection bounds retained" in description["spectrum_rounding"]
        point["integer_critical_model_grade"] = "D; fixed source-family samples and cyclic placement expectation, not actual serving routes"
    report["scope"] = "global integer synthetic spectra, dynamic expert tails and expectation of critical-rank costs with full-input/source fees preserved"
    report["formal_model_accuracy_accepted"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.batch_root, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
