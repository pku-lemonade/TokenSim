#!/usr/bin/env python3
"""Validate full windows with native DSV4 mHC pre/post costs."""

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
    result["scope"] = "native measured DSV4 mHC pre/post modules with CSA/HCA and MTP"
    for sku in tuple(result["checkpoints"]):
        c = 196 if sku == "b200" else 384
        full = json.loads((Path(directory) / sku / f"agentx_c{c}.json").read_text())
        backend = next(iter(full["latency_backends"].values()))
        assert backend.get("native_mhc")
        assert backend["native_mhc"]["hc_mult"] == 4
        assert backend["native_mhc"]["scope"].startswith("pre + post")
        assert full["operator_table_match_counts"].get("mhc")
        assert full.get("native_mhc_seconds", full["operator_component_seconds"]["elementwise"]) > 0
        components = full["operator_component_seconds"]
        local = sum(full["dp_step_stats"]["local_compute_seconds_per_rank"].values())
        assert abs(sum(components.values()) - local) < 1e-6 * max(1, local)
        result["checkpoints"][sku]["native_mhc"] = {
            "backend": backend["native_mhc"],
            "matches": full["operator_table_match_counts"]["mhc"],
            "seconds": full.get("native_mhc_seconds"),
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
