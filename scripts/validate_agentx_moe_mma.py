#!/usr/bin/env python3
"""Validate full windows after native dynamic-MMA and mixed-pipe ownership."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TokenSim.hardware.context import HardwareContext
from validate_agentx_moe_valid_work import audit as audit_valid_work


BASELINE = Path("results/agentx-dsv4-full/native-moe-valid-work/20261008-full-valid-rows")
EVIDENCE = Path("data/reference/agentx-moe-valid-work/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/latency/deepseek_v4_moe.py",
                  "TokenSim/latency/deepseek_v4_moe_work.py"}


def audit(root, *, hardware=("b200", "b300"), baseline_root=BASELINE,
          baseline_evidence=EVIDENCE, source_changes=SOURCE_CHANGES, cache_budget_auditor=None):
    report = audit_valid_work(root, hardware=hardware, baseline_root=baseline_root,
                              baseline_evidence=baseline_evidence, source_changes=source_changes,
                              cache_budget_auditor=cache_budget_auditor)
    context = HardwareContext.load("data")
    model = context.model("DeepSeek-V4-Pro")
    flops_per_row = 6 * model.hidden_size * model.moe_intermediate_size
    topk = model.moe.num_experts_per_tok
    for sku, point in report["checkpoints"].items():
        description = point["moe_description"]["work_composition"]
        assert description["mma_rows"].startswith("valid expert task tails aligned to16")
        assert "kind::mxf8f6f4" in description["mma_opcode"]
        assert "UMMA_K32" in description["mma_opcode"]
        assert description["compute_flops_per_s"] == context.device(sku).peak_compute_for("fp8_block").value
        assert description["compute_rate_grade"].startswith("D")
        assert "unmeasured" in description["compute_rate_grade"]
        assert "not measured HBM misses" in description["weight_traffic"]
        ledger, counts = point["moe_composition"], point["moe_counts"]
        for prefix in ("source", "target"):
            rows = ledger[f"{prefix}_mma_layer_rows"]
            flops = ledger[f"{prefix}_expert_flops"]
            assert rows > 0 and flops > 0
            assert abs(flops / flops_per_row - rows) <= 1e-8 * max(1, rows)
            assert ledger[f"{prefix}_issued_weight_bytes"] > 0
        for prefix, key in (("source", "full_input_layer_rows"),
                            ("target", "routed_valid_layer_rows")):
            minimum = topk * counts[key]
            assert ledger[f"{prefix}_mma_layer_rows"] + 1e-8 * max(1, minimum) >= minimum
        point["mma_ledger_conservation"] = True
        point["mixed_pipe_theoretical_only"] = True
    report["scope"] = "dynamic16-aligned expert MMA tails, per-M-task traffic and catalog FP8 theoretical pipe for MXF8F6F4"
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
