#!/usr/bin/env python3
"""Offline ownership fixtures and homogeneous anchors; not accuracy evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend


def audit():
    hardware = HardwareContext.load("data")
    model = hardware.model("DeepSeek-V4-Pro")
    points, homogeneous = [], []
    for sku in ("b200", "b300"):
        device = hardware.device(sku)
        for tp in (1, 4, 8):
            parallel = ParallelConfig(tensor_parallel_size=tp, data_parallel_size=1)
            backend = OperatorTableLatencyBackend(
                device=device, model=model, parallel_config=parallel,
                rank_info=ParallelRankInfo.from_global_rank(0, parallel),
                package=hardware.operator_package(device.device_id, "vllm"),
                attention_backend="native_dsv4",
            )
            estimator = backend.native_attention
            before = estimator.estimate_prefill(1, 512, 65536)
            estimator.reset_stats()
            after = estimator.estimate_batch({(512, 65536): 1}, {})
            assert before == after
            homogeneous.append({"hardware": sku, "tp": tp, "phase": "context", "latency_us": after, "exactly_preserved": True})
            estimator.reset_stats()
            before = estimator.estimate_decode_batch({(8192, 4): 16})
            estimator.reset_stats()
            after = estimator.estimate_batch({}, {(8192, 4): 16})
            assert before == after
            homogeneous.append({"hardware": sku, "tp": tp, "phase": "generation-q4", "latency_us": after, "exactly_preserved": True})
            if tp != 1:
                continue
            for groups, decode in (
                ({(512, 65536): 1, (512, 131072): 1}, {(8192, 4): 16}),
                ({(512, 131072): 1}, {(131072, 4): 32}),
                ({(512, 65536): 8}, {(65536, 4): 40}),
                ({(512, 65536): 1, (512, 131072): 1}, {}),
            ):
                estimator.reset_stats()
                old = sum(estimator.estimate_prefill(n, q, prefix) for (q, prefix), n in groups.items())
                old += estimator.estimate_decode_batch(decode)
                estimator.reset_stats()
                new = estimator.estimate_batch(groups, decode)
                stats = dict(estimator.composition_stats)
                assert abs(stats["standalone_module_seconds"] - old * 1e-6) < 1e-10
                assert abs(stats["shared_projection_seconds"] + stats["phase_seconds"] - new * 1e-6) < 1e-10
                points.append({
                    "hardware": sku, "context_groups": [dict(queries=q, prefix=prefix, count=n) for (q, prefix), n in groups.items()],
                    "decode_buckets": [dict(context=c, query_width=q, count=n) for (c, q), n in decode.items()],
                    "standalone_us": old, "composed_us": new,
                    "change_percent": 100 * (new / old - 1), "composition": stats,
                })
    return {"scope": "offline table-shape ownership diagnostic; no replay or fitted calibration", "homogeneous": homogeneous,
            "composition": points, "formal_accuracy_evidence": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
