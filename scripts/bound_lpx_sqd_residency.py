"""Search native GPU ownership bounds with latency-coupled LP30 KV residency."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.config.validation import integer
from scripts.lpx.fidelity import load
from scripts.lpx.mfu_metrics import useful_matrix_flops
from scripts.lpx.replay_profiles import source_identity
from scripts.lpx.sqd_bounds import (
    ROLES,
    batch_tail_bound,
    coupled_bound,
    finite_inventory_bound,
    gpu_global_cost,
    sliding_kv_bytes,
)
from scripts.run_lpx_article_sweep import article_config
from scripts.run_lpx_context_parallel import environment


def search(config, *, maximum_batch=64, card_limits=(256, 512, 1024, 2048)):
    integer(maximum_batch, "SQD maximum explicit batch", minimum=1)
    for limit in card_limits:
        integer(limit, "SQD inventory limit", minimum=2)
    hardware, model, _, _ = environment(config)
    provider = OperatorTableAFComputeProvider(
        hardware, model, fallback="analytical_only", graph_device_kernel_us=0.2
    )
    work = useful_matrix_flops(model, 101000)
    sliding = sliding_kv_bytes(model, 102000)
    local_attention = sum(
        4
        * model.num_attention_heads
        * model.attention_geometry(i).head_dim
        * min(101000, model.attention_window(i))
        for i in range(model.num_layers)
        if model.attention_window(i)
    )
    global_matrix = 2 * sum(
        model.attention_params_per_layer(i)
        - (2 * model.attention_geometry(i).head_dim if model.qk_norm else 0)
        for i in range(model.num_layers)
        if not model.attention_window(i)
    )
    lp = hardware.device("groq_lp30")
    lp_peak = lp.peak_compute_for("fp8").value
    rows, tails, winners, witnesses = [], [], {}, {}
    devices = ("h100_sxm", "b200_sxm", "rubin")
    tps = tuple(
        tp
        for tp in range(1, model.num_attention_heads + 1)
        if not model.num_attention_heads % tp
    )
    for device in devices:
        gpu = hardware.device(device)
        for tp in tps:
            for batch in range(1, maximum_batch + 1):
                for role in ROLES:
                    cost = gpu_global_cost(provider, device, tp, batch, 101000, role)
                    linear = work["linear"] - (
                        global_matrix if role == "global_attention" else 0
                    )
                    arithmetic = linear / config["compute"][
                        "gemm_mfu"
                    ] + local_attention * lp_peak / (
                        lp.peak_compute_for("bf16").value
                        * config["compute"]["attention_mfu"]
                    )
                    settings = dict(
                        useful_flops=work["total"],
                        gpu_path_s=cost["gpu_path_s"],
                        gpu_tp=tp,
                        batch=batch,
                        gpu_peak=gpu.peak_compute_for("fp8").value,
                        lp_peak=lp_peak,
                        sliding_bytes=sliding,
                        sram_bytes=lp.memory_capacity_bytes.value,
                        lp_compute_peak_seconds=arithmetic,
                    )
                    infinite = coupled_bound(**settings)
                    finite = {
                        str(limit): finite_inventory_bound(
                            max_cards=limit, lp_weight_bytes=linear / 2, **settings
                        )
                        for limit in card_limits
                    }
                    row = dict(
                        device=device,
                        gpu_tp=tp,
                        batch=batch,
                        role=role,
                        gpu_path_s=cost["gpu_path_s"],
                        lp_matrix_weight_lower_bytes=linear // 2,
                        **infinite,
                        finite_inventory=finite,
                    )
                    rows.append(row)
                    key = f"{device}/{role}"
                    if (
                        key not in winners
                        or row["mfu_bound_percent"] > winners[key]["mfu_bound_percent"]
                    ):
                        winners[key] = row
                        witnesses[key] = cost
                    if batch == maximum_batch:
                        tail_settings = {
                            k: v
                            for k, v in settings.items()
                            if k not in {"batch", "gpu_path_s"}
                        }
                        tails.append(
                            dict(
                                device=device,
                                gpu_tp=tp,
                                role=role,
                                core_memory_s_per_request=cost[
                                    "core_memory_s_per_request"
                                ],
                                **batch_tail_bound(
                                    minimum_batch=maximum_batch + 1,
                                    core_memory_s_per_request=cost[
                                        "core_memory_s_per_request"
                                    ],
                                    **tail_settings,
                                ),
                            )
                        )
        print(json.dumps(dict(device=device, explicit_rows=len(rows))), flush=True)
    finite_winners = {
        f"{key}/{limit}": max(
            (
                row
                for row in rows
                if f"{row['device']}/{row['role']}" == key
                and row["finite_inventory"][str(limit)] is not None
            ),
            key=lambda row: row["finite_inventory"][str(limit)]["mfu_bound_percent"],
            default=None,
        )
        for key in winners
        for limit in card_limits
    }
    return dict(
        record_date="2026-11-03",
        maximum_certified=False,
        source_sha256=source_identity(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config=config,
        model=model.to_dict(),
        device_specs={
            name: hardware.device(name).to_dict() for name in (*devices, "groq_lp30")
        },
        metric=work,
        sliding_kv_payload_bytes_per_resident_request=sliding,
        explicit_batch_range=[1, maximum_batch],
        gpu_tp_values=tps,
        card_limits=card_limits,
        rows=rows,
        batch_tails=tails,
        best_explicit_bounds=winners,
        best_finite_bounds=finite_winners,
        native_component_witnesses=witnesses,
        scope="Resident independent decode, separate FP8 KV, BF16 attention, native GPU whole-operator costs and exclusive worker service; all attention communication, LP30 placement and buffer costs relaxed. No GPU context parallelism, cache sharing, KV offload or alternate GPU kernels.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--maximum-batch", type=int, default=64)
    parser.add_argument(
        "--card-limits", type=int, nargs="+", default=[256, 512, 1024, 2048]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = article_config(load(args.config))
    config["compute"]["attention_mfu"] = 0.5
    result = search(
        config, maximum_batch=args.maximum_batch, card_limits=tuple(args.card_limits)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(encoded, mtime=0))
    else:
        args.output.write_bytes(encoded)
    for key, row in result["best_explicit_bounds"].items():
        print(
            json.dumps(
                dict(
                    key=key,
                    tp=row["gpu_tp"],
                    batch=row["batch"],
                    mfu_bound_percent=row["mfu_bound_percent"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
