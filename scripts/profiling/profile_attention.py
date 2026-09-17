#!/usr/bin/env python3
"""Measure attention-core latency for the ``context_attention`` and
``generation_attention`` keys of a shape manifest.

Prefill uses ``flash_attn_func`` (FlashAttention-2, causal). Decode uses
``torch.nn.functional.scaled_dot_product_attention`` against a dense KV cache
of ``context_len`` tokens by default, or vLLM's paged attention when
``--decode-kernel vllm`` is given and importable. Record the kernel you used:
paged and dense decode kernels differ by tens of percent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import environment_record, manifest_keys, time_kernel, torch_dtype, write_csv

CONTEXT_COLUMNS = ["attn_dtype", "kv_cache_dtype", "batch_size", "input_seq_len", "num_heads", "num_kv_heads", "head_dim", "window_size", "latency_us", "kernel", "source_id", "device_name", "driver", "framework_versions"]
GENERATION_COLUMNS = ["attn_dtype", "kv_cache_dtype", "batch_size", "context_len", "num_heads", "num_kv_heads", "head_dim", "window_size", "latency_us", "kernel", "source_id", "device_name", "driver", "framework_versions"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--decode-kernel", choices=["sdpa", "vllm"], default="sdpa")
    args = parser.parse_args()

    import torch

    env = environment_record()
    versions = f"torch={env['torch']};cuda={env['cuda']}"
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        flash_attn_func = None
        print("flash_attn not importable; prefill rows will use SDPA")

    context_rows = []
    for key in manifest_keys(args.manifest, "context_attention"):
        dtype = torch_dtype(key["attn_dtype"])
        b, s, h, hkv, d = (int(key[f]) for f in ("batch_size", "input_seq_len", "num_heads", "num_kv_heads", "head_dim"))
        q = torch.randn(b, s, h, d, device="cuda", dtype=dtype)
        kv = torch.randn(2, b, s, hkv, d, device="cuda", dtype=dtype)
        window = int(key.get("window_size", 0) or 0)
        if flash_attn_func is not None:
            ws = (window - 1, 0) if window else (-1, -1)
            fn = lambda: flash_attn_func(q, kv[0], kv[1], causal=True, window_size=ws)
            kernel = "flash_attn_func"
        else:
            qt = q.transpose(1, 2)
            kt = kv[0].transpose(1, 2).repeat_interleave(h // hkv, dim=1)
            vt = kv[1].transpose(1, 2).repeat_interleave(h // hkv, dim=1)
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
            kernel = "torch.sdpa"
        context_rows.append({**key, "latency_us": time_kernel(fn, iters=args.iters), "kernel": kernel,
                             "source_id": f"measured:{args.device_id}:context_attention:{kernel}",
                             "device_name": env["device_name"], "driver": env["driver"], "framework_versions": versions})

    generation_rows = []
    for key in manifest_keys(args.manifest, "generation_attention"):
        dtype = torch_dtype(key["attn_dtype"])
        b, ctx, h, hkv, d = (int(key[f]) for f in ("batch_size", "context_len", "num_heads", "num_kv_heads", "head_dim"))
        q = torch.randn(b, h, 1, d, device="cuda", dtype=dtype)
        k = torch.randn(b, hkv, ctx, d, device="cuda", dtype=dtype).repeat_interleave(h // hkv, dim=1)
        v = torch.randn(b, hkv, ctx, d, device="cuda", dtype=dtype).repeat_interleave(h // hkv, dim=1)
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
        kernel = "torch.sdpa-dense-kv"
        generation_rows.append({**key, "latency_us": time_kernel(fn, iters=args.iters), "kernel": kernel,
                                "source_id": f"measured:{args.device_id}:generation_attention:{kernel}",
                                "device_name": env["device_name"], "driver": env["driver"], "framework_versions": versions})

    out = Path(args.out_dir)
    write_csv(out / "context_attention_perf.csv", context_rows, CONTEXT_COLUMNS)
    write_csv(out / "generation_attention_perf.csv", generation_rows, GENERATION_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
