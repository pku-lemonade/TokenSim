#!/usr/bin/env python3
"""Measure RMSNorm / RoPE / residual-add / SwiGLU latency for the ``elementwise``
keys of a shape manifest and merge the rows into a device's operator package.

AIConfigurator's collector has no elementwise table, so this is the one
compute table TokenSim still measures with its own script.

    python scripts/collect/collect_elementwise.py \
        --manifest tmp/manifests/llama-3-8b.yaml --device a100_sxm_80g \
        --package data/operator_data/a100_sxm_80g/trtllm

By default the reference implementation is plain PyTorch eager ops. Pass
``--kernel vllm`` to time vLLM's fused ``rms_norm`` / ``fused_add_rms_norm`` /
``rotary_embedding`` / ``silu_and_mul`` kernels instead, which is what a vLLM
deployment actually runs. Record which one you used: the choice is stored in
the ``kernel`` column and in the source notes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from common import REPO_ROOT, environment_record, manifest_keys, time_kernel, torch_dtype

from TokenSim.operator_data.generator import merge_packages  # noqa: E402
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord  # noqa: E402


def _eager_ops(x, y, w, cos, sin, h, dtype, op):
    import torch

    if op == "rmsnorm":
        return lambda: x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype) * w
    if op == "rope":

        def fn():
            x1, x2 = x[..., : h // 2], x[..., h // 2 :]
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

        return fn
    if op == "residual_add":
        return lambda: x + y
    if op == "swiglu":
        return lambda: torch.nn.functional.silu(x) * y
    return None


def _vllm_ops(x, y, w, cos, sin, h, dtype, op):
    import torch
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    if op == "rmsnorm":
        return lambda: ops.rms_norm(out, x, w, 1e-6)
    if op == "residual_add":
        # vLLM fuses the residual add into the following RMSNorm.
        return lambda: ops.fused_add_rms_norm(x, y, w, 1e-6)
    if op == "swiglu":
        xy = torch.cat((x, y), dim=-1)
        return lambda: ops.silu_and_mul(out, xy)
    if op == "rope":
        positions = torch.arange(x.shape[0], device="cuda")
        cos_sin = torch.cat((cos, sin), dim=-1)[: x.shape[0]]
        q = x.clone()
        k = x.clone()
        return lambda: ops.rotary_embedding(positions, q, k, h, cos_sin, True)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="shape manifest YAML (operator_data.cli manifest)")
    parser.add_argument("--device", required=True, help="TokenSim device_id, e.g. a100_sxm_80g")
    parser.add_argument("--package", required=True, help="operator package directory to merge into (created if missing)")
    parser.add_argument("--backend", default="trtllm", help="backend label of the package (default trtllm)")
    parser.add_argument("--kernel", choices=["eager", "vllm"], default="eager")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--operator", default="", help="who collected the data (stored in the source notes)")
    args = parser.parse_args()

    import torch

    env = environment_record()
    build = _vllm_ops if args.kernel == "vllm" else _eager_ops
    kernel_label = f"vllm-custom-ops" if args.kernel == "vllm" else "torch-eager"
    source_id = f"measured:{args.device}:elementwise:{kernel_label}"
    rows = []
    skipped = []
    for key in manifest_keys(args.manifest, "elementwise"):
        dtype = torch_dtype(key["dtype"])
        t, h = int(key["num_tokens"]), int(key["hidden_size"])
        x = torch.randn(t, h, device="cuda", dtype=dtype)
        y = torch.randn(t, h, device="cuda", dtype=dtype)
        w = torch.randn(h, device="cuda", dtype=dtype)
        cos = torch.randn(t, h // 2, device="cuda", dtype=dtype)
        sin = torch.randn(t, h // 2, device="cuda", dtype=dtype)
        fn = build(x, y, w, cos, sin, h, dtype, key["op_name"])
        if fn is None:
            skipped.append(key)
            continue
        rows.append(
            {
                **key,
                "latency_us": time_kernel(fn, iters=args.iters),
                "kernel": f"{kernel_label}:{key['op_name']}",
                "source_id": source_id,
            }
        )
    if not rows:
        print("no elementwise rows measured")
        return 1

    source = SourceRecord(
        source_id=source_id,
        grade="A",
        method="measured",
        reference=f"scripts/collect/collect_elementwise.py --kernel {args.kernel}",
        notes=(
            f"{env['device_name']}; driver/clocks {env['driver']}; torch {env['torch']} cuda {env['cuda']}; "
            f"CUDA-event median of {args.iters} iterations after 5 warm-ups; "
            f"collected {_dt.date.today().isoformat()} by {args.operator or 'unknown'}"
        ),
        device=args.device,
        backend=f"{args.backend}:{kernel_label}",
    )
    meta = PackageMeta(
        dataset_version=f"{args.device}-{args.backend}-elementwise-{_dt.date.today().isoformat()}",
        device_id=args.device,
        backend=args.backend,
        sources={source_id: source},
        notes="Elementwise kernels measured with scripts/collect/collect_elementwise.py.",
    )
    new_package = OperatorDataPackage.from_rows(meta, {"elementwise": rows})
    target = Path(args.package)
    if (target / "generation_meta.yaml").is_file():
        new_package = merge_packages(OperatorDataPackage.load(target), new_package, prefer="extra")
    new_package.write(target)
    print(f"wrote {len(rows)} elementwise rows into {target}; skipped {len(skipped)} unsupported keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
