#!/usr/bin/env python3
"""Audit full native checkpoints after ragged decode module composition."""

import argparse
import json
from pathlib import Path

from validate_agentx_cpu import validate


def audit(batch_root, *, seed=0, hardware=("b200", "b300")):
    points = {}
    for sku, concurrency in (("b200", 196), ("b300", 384)):
        if sku not in hardware:
            continue
        point = Path(batch_root) / f"seed{seed}" / f"{sku}-c{concurrency}-dep8-simplecpu"
        base = validate(point, hardware=(sku,), verify_reservations=True)
        result = json.loads((point / sku / f"agentx_c{concurrency}.json").read_text())
        backend = next(iter(result["latency_backends"].values()))
        attention = backend["native_dsv4_attention"]
        assert attention["context_latency_scale"] == 1.0
        assert attention["generation_latency_scale"] == 1.0
        assert attention["decode_batch_composition"].startswith("query-row-weighted")
        batches = result["native_dsv4_attention_decode_batches"]
        assert 0 < batches["batches"] <= result["operator_step_count"]
        assert batches["length_groups"] >= batches["batches"]
        query_rows = sum(
            worker["counts"]["scheduled_decode_query_tokens"]
            for worker in result["scheduler_stats"]
        )
        # Idle DP ranks run synthetic target batches too. They are included in
        # operator time/rows and absent from actual scheduler token progress.
        assert batches["query_rows"] >= query_rows
        assert result["native_dsv4_attention_seconds"] > 0
        components = result["operator_component_seconds"]
        draft_attention = (result.get("native_dsv4_mtp_component_seconds") or {}).get("attention", 0)
        assert abs(result["native_dsv4_attention_seconds"] + draft_attention - components["attention"]) < 1e-6
        local = sum(result["dp_step_stats"]["local_compute_seconds_per_rank"].values())
        assert abs(sum(components.values()) - local) < 1e-6 * max(1, local)
        points[sku] = {
            **base["checkpoints"][sku],
            "decode_batch_stats": batches,
            "component_seconds": components,
            "attention_provenance": attention,
            "native_attention_seconds_preserved_in_result": True,
            "component_time_conservation": True,
        }
    return {
        "scope": "full-batch ragged decode attention module composition; native scale 1.0",
        "seed": seed,
        "trace_pool": 393,
        "issue_window_seconds": 3600,
        "formal_model_accuracy_accepted": False,
        "checkpoints": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.batch_root, seed=args.seed, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
