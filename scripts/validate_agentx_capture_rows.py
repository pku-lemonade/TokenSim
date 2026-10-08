#!/usr/bin/env python3
"""Validate native capture/SP main-row ownership on complete checkpoints."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_agentx_moe_spectrum import audit as audit_spectrum


BASELINE = Path("results/agentx-dsv4-full/native-moe-spectrum/20261008-global-integer-critical")
EVIDENCE = Path("data/reference/agentx-moe-spectrum/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/latency/operator_table.py", "TokenSim/latency/deepseek_v4_attention.py",
                  "TokenSim/config/psla_config.py", "util/results.py"}


def audit(root, *, hardware=("b200", "b300"), baseline_root=BASELINE,
          baseline_evidence=EVIDENCE, source_changes=SOURCE_CHANGES, cache_budget_auditor=None):
    report = audit_spectrum(root, hardware=hardware, baseline_root=baseline_root,
                            baseline_evidence=baseline_evidence, source_changes=source_changes,
                            cache_budget_auditor=cache_budget_auditor)
    for sku, point in report["checkpoints"].items():
        concurrency = 196 if sku == "b200" else 384
        path = Path(root) / "seed0" / f"{sku}-c{concurrency}-dep8-simplecpu" / sku / f"agentx_c{concurrency}.json"
        result = json.loads(path.read_text())
        main, logits = result["dsv4_main_forward_counts"], result["dsv4_target_logits"]
        assert main["steps"] == result["operator_step_count"]
        assert main["logical_rows"] == logits["forward_rows"]
        assert main["full_rows"] == logits["output_forward_rows"]
        assert main["full_rows"] == main["logical_rows"] + main["capture_padding_rows"]
        # Both main checkpoints use TP1; TP4/8 shard/tail contracts are
        # exercised independently against source-derived unit cases.
        assert main["sp_rows"] == main["full_rows"]
        assert main["sp_tail_rows"] == main["sp_steps"] == 0
        assert main["attention_projection_layer_rows"] == 61 * main["full_rows"]
        assert main["mhc_layer_rows"] == main["router_layer_rows"] == 61 * main["full_rows"]
        assert main["norm_layer_rows"] == 122 * main["full_rows"]
        capture = result["native_dsv4_attention_composition"]
        assert capture["capture_batches"] == main["graph_padded_steps"] > 0
        assert capture["capture_descriptor_rows"] == capture["capture_logical_rows"] + capture["capture_padding_rows"]
        assert capture["capture_padding_rows"] == main["capture_padding_rows"]
        floor, projection = capture["capture_source_floor_layer_rows"], capture["capture_projection_bound_layer_rows"]
        increment = capture["capture_increment_layer_rows"]
        assert floor >= 61 * capture["capture_logical_rows"] - 1e-8 * max(1, floor)
        assert projection >= floor and projection >= 61 * capture["capture_descriptor_rows"] - 1e-8 * max(1, projection)
        assert abs(projection - floor - increment) <= 1e-8 * max(1, projection)
        assert 0 <= increment <= 61 * capture["capture_padding_rows"] + 1e-8 * max(1, increment)
        assert capture["capture_shared_increment_seconds"] >= 0
        point["main_forward_counts"] = main
        point["capture_projection_counts"] = {key: value for key, value in capture.items() if key.startswith("capture_")}
        point["main_capture_sp_row_conservation"] = True
    report["scope"] = "native main full capture/SP owners, source-floor incremental shared projections and preserved actual sparse query ownership"
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
