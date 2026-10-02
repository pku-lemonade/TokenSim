#!/usr/bin/env python3
"""Compare one audited AIC shape with TokenSim's native attention lookup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_attention import DeepSeekV4AttentionEstimator


def tokensim_attention(device: str) -> dict:
    hardware = HardwareContext.load("data")
    model = hardware.model("DeepSeek-V4-Pro")
    package = hardware.operator_package(device, "vllm")
    parallel = ParallelConfig(tensor_parallel_size=8, data_parallel_size=1)

    class Stats:
        def record(self, *_args):
            return None

    class Backend:
        pass

    backend = Backend()
    backend.model = model
    backend.parallel_config = parallel
    backend.package = package
    backend.stats = Stats()
    estimator = DeepSeekV4AttentionEstimator(backend)
    return {
        "device": device,
        "shape": {"batch_size": 8, "query_tokens": 1024, "context_tokens": 1024, "tp": 8},
        "prefill_attention_us": estimator.estimate_prefill(8, 1024, 0),
        "decode_attention_us": estimator.estimate_decode(8, 1024),
        "provenance": estimator.describe(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aic = json.loads(args.aic_report.read_text())
    report = {
        "schema_version": 1,
        "shape": aic["shape"],
        "aic_source_commit": aic["aic_source_commit"],
        "aic": aic["systems"],
        "tokensim": {"b200_sxm": tokensim_attention("b200_sxm")},
        "interpretation": [
            "B200 uses the same batch/sequence/TP shape but AIC's NVFP4 preview config; dtype and backend provenance differ from formal native FP8/MXFP4 simulation.",
            "B300 AIC has an explicit missing MegaMoE row and is not synthesized from B200.",
            "This is a diagnostic composition comparison, not a reference calibration coefficient.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(args.output)
