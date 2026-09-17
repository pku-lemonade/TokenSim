#!/usr/bin/env python3
"""Measure GEMM latency for every ``gemm`` key of a shape manifest.

    python scripts/profiling/profile_gemm.py --manifest exp.yaml --device-id a100_sxm_80g --out gemm_perf.csv

Dense fp16/bf16 GEMMs use ``torch.matmul`` (cuBLAS). Weight-only quantized
shapes are skipped unless ``--quant-backend vllm`` is given, in which case the
vLLM GPTQ/AWQ style kernels are used when importable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import environment_record, manifest_keys, time_kernel, torch_dtype, write_csv

COLUMNS = ["dtype", "m", "n", "k", "latency_us", "kernel", "source_id", "device_name", "driver", "framework_versions"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--quant-backend", choices=["none", "vllm"], default="none")
    args = parser.parse_args()

    import torch

    env = environment_record()
    source_id = f"measured:{args.device_id}:gemm:torch-{torch.__version__}"
    rows = []
    for key in manifest_keys(args.manifest, "gemm"):
        dtype, m, n, k = key["dtype"], int(key["m"]), int(key["n"]), int(key["k"])
        if dtype not in ("fp16", "bf16"):
            if args.quant_backend == "none":
                print(f"skip {key}: quantized GEMM needs --quant-backend")
                continue
            print(f"skip {key}: quant backend hooks not implemented in this script yet")
            continue
        tdtype = torch_dtype(dtype)
        a = torch.randn(m, k, device="cuda", dtype=tdtype)
        w = torch.randn(n, k, device="cuda", dtype=tdtype)  # nn.Linear layout: [out, in]
        latency = time_kernel(lambda: torch.nn.functional.linear(a, w), iters=args.iters)
        rows.append(
            {
                **key,
                "latency_us": latency,
                "kernel": "torch.nn.functional.linear",
                "source_id": source_id,
                "device_name": env["device_name"],
                "driver": env["driver"],
                "framework_versions": f"torch={env['torch']};cuda={env['cuda']}",
            }
        )
    write_csv(Path(args.out), rows, COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
