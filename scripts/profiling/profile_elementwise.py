#!/usr/bin/env python3
"""Measure RMSNorm / RoPE / residual-add / SwiGLU latency for the ``elementwise``
keys of a shape manifest using plain PyTorch ops (fused serving kernels are
faster; record the kernel you actually use in production if different)."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import environment_record, manifest_keys, time_kernel, torch_dtype, write_csv

COLUMNS = ["op_name", "dtype", "num_tokens", "hidden_size", "latency_us", "kernel", "source_id", "device_name", "driver", "framework_versions"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import torch

    env = environment_record()
    versions = f"torch={env['torch']};cuda={env['cuda']}"
    rows = []
    for key in manifest_keys(args.manifest, "elementwise"):
        dtype = torch_dtype(key["dtype"])
        t, h = int(key["num_tokens"]), int(key["hidden_size"])
        x = torch.randn(t, h, device="cuda", dtype=dtype)
        y = torch.randn(t, h, device="cuda", dtype=dtype)
        w = torch.randn(h, device="cuda", dtype=dtype)
        op = key["op_name"]
        if op == "rmsnorm":
            fn = lambda: x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype) * w
        elif op == "rope":
            cos = torch.randn(t, h // 2, device="cuda", dtype=dtype)
            sin = torch.randn(t, h // 2, device="cuda", dtype=dtype)

            def fn():
                x1, x2 = x[..., : h // 2], x[..., h // 2 :]
                return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        elif op == "residual_add":
            fn = lambda: x + y
        elif op == "swiglu":
            fn = lambda: torch.nn.functional.silu(x) * y
        else:
            print(f"skip {op}: no reference implementation")
            continue
        rows.append({**key, "latency_us": time_kernel(fn, iters=args.iters), "kernel": f"torch-eager:{op}",
                     "source_id": f"measured:{args.device_id}:elementwise:torch-eager",
                     "device_name": env["device_name"], "driver": env["driver"], "framework_versions": versions})
    write_csv(Path(args.out), rows, COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
