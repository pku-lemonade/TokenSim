#!/usr/bin/env python3
"""Validate full AgentX windows using the native MegaMoE/shared-expert path."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, *, hardware=("b200", "b300")):
    directory = Path(directory)
    result = validate(
        directory,
        hardware=hardware,
        verify_reservations=True,
        verify_sources=False,
    )
    result["scope"] = (
        "borrowed DSV4 MegaMoE routed dispatch/compute/combine with explicit "
        "shared FP8 fused incremental estimate; attention and MTP are excluded "
        "from this checkpoint"
    )
    for sku, checkpoint in result["checkpoints"].items():
        c = 196 if sku == "b200" else 384
        full = json.loads((directory / sku / f"agentx_c{c}.json").read_text())
        device_id = next(iter(full["latency_backends"]))
        backend = full["latency_backends"][device_id]
        native = backend.get("native_megamoe")
        assert native and native["backend"] == "deep_gemm_amxf4_mega_moe"
        assert native["shared_mode"] == "fused_incremental_estimate"
        assert native["shared_dtype"] == "fp8_block"
        assert backend["moe_kernel_scope"].startswith("native borrowed MegaMoE")
        assert full["moe_straggler_latency"] == 0
        components = full["operator_component_seconds"]
        local_time = sum(
            full["dp_step_stats"]["local_compute_seconds_per_rank"].values()
        )
        assert abs(sum(components.values()) - local_time) < 1e-6 * max(1, local_time)
        assert abs(components["moe"] - full["moe_compute_latency"]) < 1e-6 * max(
            1, full["moe_compute_latency"]
        )
        assert full["moe_all2all_latency"] == 0
        assert full["moe_all2all_event_count"] == full["operator_step_count"]
        records = full.get("operator_table_match_counts", {})
        assert sum(records.get("dsv4_megamoe", {}).values()) > 0
        assert not records.get("ep_all2all")
        checkpoint["native_megamoe"] = {
            "backend": native,
            "operator_table_matches": records.get("dsv4_megamoe", {}),
            "routed_seconds": full.get("native_megamoe_routed_seconds"),
            "shared_seconds": full.get("native_megamoe_shared_seconds"),
            "component_time_conservation": True,
            "dispatch_combine_separate_ep_comm": False,
            "shared_expert_scope": "fused incremental analytical estimate, grade D",
            "output_token_throughput_tps": full["agentx_metrics"][
                "output_token_throughput_tps"
            ],
            "reference_error_percent": checkpoint.get("reference_error_percent"),
            "result_sha256": digest(directory / sku / f"agentx_c{c}.json"),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", nargs="+", default=("b200", "b300"))
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(audit(args.directory, hardware=tuple(args.hardware)), indent=2)
        + "\n"
    )
    print(args.output)
