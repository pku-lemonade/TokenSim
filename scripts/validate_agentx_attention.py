#!/usr/bin/env python3
"""Validate native CSA/HCA attention plus MTP q=4 full windows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


def audit(directory, *, hardware=("b200", "b300")):
    result = validate(
        Path(directory), hardware=hardware, verify_reservations=True, verify_sources=False
    )
    result["scope"] = (
        "native DSV4 CSA/HCA complete attention modules (borrowed FlashMLA table) "
        "with MTP accepted 2/3 output tokens and target q=4; mHC remains generic"
    )
    for sku in tuple(result["checkpoints"]):
        c = 196 if sku == "b200" else 384
        full = json.loads(
            (Path(directory) / sku / f"agentx_c{c}.json").read_text()
        )
        device_id = next(iter(full["latency_backends"]))
        backend = full["latency_backends"][device_id]
        assert backend.get("attention_module")
        assert backend["attention_module"]["layer_counts"] == {"csa": 30, "hca": 31}
        assert sum(full["operator_table_match_counts"].get("dsv4_attention", {}).values()) > 0
        assert sum(full["operator_table_match_counts"].get("dsv4_megamoe", {}).values()) > 0
        assert not full["operator_table_match_counts"].get("ep_all2all")
        attention_seconds = full.get(
            "attention_module_seconds", full["operator_component_seconds"]["attention"]
        )
        attention_counts = full.get(
            "attention_module_counts",
            full["operator_table_match_counts"]["dsv4_attention"],
        )
        assert attention_seconds > 0
        assert attention_counts
        scheduled_decode = sum(
            w["counts"].get("scheduled_decode_tokens", 0)
            for w in full["scheduler_stats"]
        )
        query_decode = sum(
            w["counts"].get("scheduled_decode_query_tokens", 0)
            for w in full["scheduler_stats"]
        )
        assert query_decode >= scheduled_decode
        assert query_decode > scheduled_decode
        components = full["operator_component_seconds"]
        local_time = sum(
            full["dp_step_stats"]["local_compute_seconds_per_rank"].values()
        )
        assert abs(sum(components.values()) - local_time) < 1e-6 * max(1, local_time)
        result["checkpoints"][sku]["native_attention"] = {
            "backend": backend["attention_module"],
            "attention_matches": full["operator_table_match_counts"]["dsv4_attention"],
            "attention_seconds": attention_seconds,
            "attention_counts": attention_counts,
            "scheduled_decode_tokens": scheduled_decode,
            "scheduled_decode_query_tokens": query_decode,
            "query_width_expansion": query_decode / scheduled_decode,
            "component_time_conservation": True,
            "output_token_throughput_tps": full["agentx_metrics"]["output_token_throughput_tps"],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.directory), indent=2) + "\n")
    print(args.output)
