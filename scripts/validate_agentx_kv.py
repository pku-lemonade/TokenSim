#!/usr/bin/env python3
"""Compute native KV budgets, then compare independently read historical logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware.context import HardwareContext


def audit(reference_root, source_root):
    hardware = HardwareContext.load(ROOT / "data")
    model = hardware.model("DeepSeek-V4-Pro")
    # Construct both configurations before reading any historical capacities.
    caches = {}
    for sku, budget, seqs, util in (("b200", 8192, 49, 0.9), ("b300", 16384, 96, 0.92)):
        parallel = ClusterConfig.from_file(
            ROOT / f"data/clusters/8_{sku}/dep8.json"
        ).effective_parallel_config()
        caches[sku] = CacheConfig(
            256,
            hardware.device(sku),
            model,
            parallel,
            usable_memory_fraction=util,
            max_num_batched_tokens=budget,
            max_num_seqs=seqs,
        )
    points = {}
    for sku, cache in caches.items():
        path = reference_root / sku / "agentic/server.log"
        text = path.read_text(errors="replace")
        available = sorted(
            set(
                float(x)
                for x in re.findall(r"Available KV cache memory: ([\d.]+) GiB", text)
            )
        )
        equivalents = sorted(
            set(
                int(x.replace(",", ""))
                for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
            )
        )
        profiler = next(line for line in text.splitlines() if "GiB for weight" in line)
        current = cache.memory_budget["available_kv_bytes"] / 2**30
        equivalent = cache.hybrid_layout.equivalent_tokens(cache.num_gpu_blocks)
        computed = cache.to_dict()
        computed["weight_memory"] = {
            key: value
            for key, value in computed["weight_memory"].items()
            if key != "tensors"
        }
        points[sku] = {
            "computed": computed,
            "historical_log": {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kv_gib": available,
                "log_equivalent_tokens": equivalents,
                "profiler_line": profiler,
            },
            "kv_gib_error_percent": [
                (current / value - 1) * 100 for value in available
            ],
            "equivalent_token_error_percent": [
                (equivalent / value - 1) * 100 for value in equivalents
            ],
            "capacity_goal_met": all(
                abs(current / value - 1) <= 0.01 for value in available
            ),
        }
    paths = [
        "vllm/models/deepseek_v4/attention.py",
        "vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py",
        "vllm/v1/kv_cache_interface.py",
        "vllm/v1/core/kv_cache_utils.py",
        "vllm/v1/core/kv_cache_coordinator.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/block_pool.py",
    ]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    return {
        "scope": "physical HBM layout and allocation; CPU offload and native operator costs pending",
        "capacity_policy": "computed before logs; no historical capacity overrides",
        "source": {
            "repository": "https://github.com/vllm-project/vllm",
            "revision": revision,
            "files": {
                path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
                for path in paths
            },
        },
        "checkpoints": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("/tmp/infx-art"))
    parser.add_argument(
        "--source-root", type=Path, default=Path("/home/zhuohang/Serving/vllm-note")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/reference/agentx-kv-layout/validation.json",
    )
    args = parser.parse_args()
    result = audit(args.reference_root, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                sku: {
                    "physical_slots": p["computed"]["num_gpu_blocks"],
                    "kv_error_percent": p["kv_gib_error_percent"],
                    "capacity_goal_met": p["capacity_goal_met"],
                }
                for sku, p in result["checkpoints"].items()
            },
            indent=2,
        )
    )
