"""DeepSeek-V3 MLA/AF simulation on the current operator-table backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace
from typing import Any

from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.device.sourced_value import SourcedValue

MODEL_ID = "deepseek-v3-671b-mla"
_ASSUMED_DEVICE = "a100_sxm_80g"


def configuration(
    *,
    full_model: bool = False,
    a_tp: int = 1,
    f_tp: int | None = None,
    af_batch_size: int = 32,
    data_root: str | Path | None = None,
    device_id: str = _ASSUMED_DEVICE,
    network: str = "nvlink3",
    hypothetical_f_memory_gib: int | None = None,
):
    """Build a V3 ModelSpec, AF cluster and compiled runtime plan."""
    if a_tp < 1 or (f_tp is not None and f_tp < 1) or af_batch_size < 1:
        raise ValueError("parallel sizes and af_batch_size must be positive")
    hardware = HardwareContext.load(data_root or ROOT / "data")
    model = hardware.model(MODEL_ID)
    if not full_model:
        model = replace(
            model,
            num_layers=1,
            moe=replace(model.moe, num_moe_layers=1, first_k_dense_replace=0),
        )
    f_tp = f_tp or (32 if full_model else 1)
    device = hardware.device(device_id).device_id
    f_device = device
    if hypothetical_f_memory_gib is not None:
        if type(hypothetical_f_memory_gib) is not int or hypothetical_f_memory_gib < 1:
            raise ValueError("hypothetical_f_memory_gib must be a positive integer")
        base = hardware.device(device)
        f_device = f"{device}_virtual_f_{hypothetical_f_memory_gib}gib"
        hardware.devices.add(
            replace(
                base,
                device_id=f_device,
                display_name=f_device,
                aliases=(),
                memory_capacity_bytes=SourcedValue(
                    hypothetical_f_memory_gib * 2**30,
                    "assumption:virtual-f-capacity",
                    "D",
                ),
            )
        )
    groups = [
        dict(
            name="a-pool",
            capabilities=["af_attention"],
            hardware=device,
            num_workers=a_tp,
            network="af-net",
            operator_backend="vllm",
        ),
        dict(
            name="f-pool",
            capabilities=["af_ffn"],
            hardware=f_device,
            num_workers=f_tp,
            network="af-net",
            operator_backend="vllm",
        ),
    ]
    a_mappings = [
        dict(
            domain_rank=i,
            tp_rank=i,
            pp_rank=0,
            dp_rank=0,
            ep_rank=0,
            physical_worker=dict(group_name="a-pool", worker_index=i),
        )
        for i in range(a_tp)
    ]
    f_mappings = [
        dict(
            domain_rank=i,
            tp_rank=i,
            pp_rank=0,
            dp_rank=0,
            ep_rank=i,
            physical_worker=dict(group_name="f-pool", worker_index=i),
        )
        for i in range(f_tp)
    ]
    domains = [
        dict(
            name="a-domain",
            capability="af_attention",
            worker_group="a-pool",
            parallel=dict(
                tensor_parallel_size=a_tp,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                expert_parallel_size=1,
            ),
            logical_instances=[dict(name="a0", rank_mapping=a_mappings)],
            communication_plan=(
                [
                    dict(
                        operation=op,
                        after_stage=stage,
                        participants=list(range(a_tp)),
                        network="af-net",
                        resource_group="af-fabric",
                        resource_lane=lane,
                    )
                    for op, stage, lane in (
                        ("all_reduce", "a_compute", "a_to_f"),
                        ("all_gather", "residual", "f_to_a"),
                    )
                ]
                if a_tp > 1
                else []
            ),
        ),
        dict(
            name="f-domain",
            capability="af_ffn",
            worker_group="f-pool",
            parallel=dict(
                tensor_parallel_size=f_tp,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                expert_parallel_size=f_tp,
                expert_placement_strategy="linear",
            ),
            logical_instances=[dict(name="f0", rank_mapping=f_mappings)],
            communication_plan=(
                [
                    dict(
                        operation="all_reduce",
                        after_stage="expert_compute",
                        participants=list(range(f_tp)),
                        network="af-net",
                        resource_group="af-fabric",
                        resource_lane="f_to_a",
                    )
                ]
                if f_tp > 1
                else []
            ),
        ),
    ]
    doc = dict(
        num_workers=a_tp + f_tp,
        networks={"af-net": network},
        worker_groups=groups,
        af=dict(
            enabled=True,
            af_block_size=af_batch_size,
            latency_provider="operator_table",
            trace_enabled=True,
            router_location="a",
            weight_dtype="bf16",
            # Explicit eight-card inventory: legacy virtual same-domain layouts
            # must not silently become node-domain physical experiments.
            communication_domains=dict(
                node_size=8,
                node_count=(a_tp + 7) // 8 + (f_tp + 7) // 8,
                placement_strategy="dedicated",
            ),
            activation_dtype="bf16",
            kv_cache_dtype="bf16",
            execution_domains=domains,
            links=[
                dict(
                    name="af-link",
                    attention_group="a-pool",
                    ffn_group="f-pool",
                    attention_domain="a-domain",
                    ffn_domain="f-domain",
                    network="af-net",
                    resource_group="af-fabric",
                    duplex="full",
                    a_to_f_capacity=1,
                    f_to_a_capacity=1,
                )
            ],
        ),
    )
    return build_model_runtime(doc, hardware=hardware, model=model)


simulate = simulate_decode


def write_profile(path: str | Path, executor, metadata: dict[str, Any] | None = None):
    events = []
    for event in executor.trace.events:
        row = event.as_dict()
        events.append(
            dict(
                name=row.get("stage") or row.get("operation") or row.get("event_type"),
                cat=row.get("event_type"),
                ph="X",
                pid=1,
                tid=row.get("resource_id"),
                ts=row["start_at"] * 1e6,
                dur=row["service_duration"] * 1e6,
                args=row,
            )
        )
    payload = {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "metadata": metadata or {"timing": "simulated operator-table service time"},
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def run(args):
    config = configuration(
        full_model=args.full_model,
        a_tp=args.a_tp,
        f_tp=args.f_tp,
        af_batch_size=args.af_batch_size,
        data_root=args.data_root,
        device_id=args.device,
        network=args.network,
        hypothetical_f_memory_gib=args.hypothetical_f_memory_gib,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model-spec.json", config.model.to_dict())
    write_json(output / "cluster.json", config.doc)
    write_json(output / "runtime-plan.json", config.plan.summary())
    assumptions = {
        "devices": {
            worker.hardware: config.hardware.device(worker.hardware).to_dict()
            for worker in config.workers
        },
        "network": args.network,
        "link": config.hardware.link(args.network).__dict__,
        "fallback": args.fallback,
        "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
        "virtual_capacity_only": args.hypothetical_f_memory_gib is not None,
        "note": "F capacity override changes storage only; compute and memory bandwidth remain those of the base device.",
    }
    write_json(output / "hardware-assumptions.json", assumptions)
    rows = []
    for batch in args.batches:
        print(
            f"Run operator-table V3 decode: layers={config.plan.total_layers}, B={batch}",
            flush=True,
        )
        executor, measurements, compute, transfer = simulate(
            config,
            batch,
            args.context,
            steps=args.steps,
            trace=True,
            fallback=args.fallback,
        )
        step_us = sum(row["makespan_us"] for row in measurements) / args.steps
        profile = write_profile(
            output / f"decode-layers{config.plan.total_layers}-b{batch}.trace.json",
            executor,
            metadata={
                "model": MODEL_ID,
                "backend": "operator_table",
                "timing": "simulated service and queue durations; no GPU execution",
                "hardware_assumptions_file": "hardware-assumptions.json",
                "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
            },
        )
        rows.append(
            {
                "batch_size": batch,
                "context_len": args.context,
                "steps": args.steps,
                "avg_step_us": step_us,
                "tokens_per_s": batch / (step_us / 1e6),
                "tokens_per_s_per_user": 1e6 / step_us,
                "profile": str(profile.name),
                "operator_stats": compute.stats_dict(),
                "transfer": transfer.describe(),
            }
        )
    write_json(
        output / "summary.json",
        {
            "status": "operator_table_simulation",
            "model": config.model.to_dict(),
            "layers": config.plan.total_layers,
            "af_batch_size": args.af_batch_size,
            "hardware_assumptions_file": "hardware-assumptions.json",
            "hypothetical_f_memory_gib": args.hypothetical_f_memory_gib,
            "fallback": args.fallback,
            "results": rows,
            "limitations": [
                "CPU event simulation; no checkpoint loading or GPU execution",
                "BF16 architectural timing; official checkpoint FP8 is not modeled",
                "Decode phase only; prefill/MTP are not a serving benchmark",
            ],
        },
    )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/deepseek-v3-operator-table"
    )
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--a-tp", type=int, default=1)
    parser.add_argument("--f-tp", type=int)
    parser.add_argument("--af-batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="a100_sxm_80g")
    parser.add_argument("--network", default="nvlink3")
    parser.add_argument("--hypothetical-f-memory-gib", type=int)
    parser.add_argument(
        "--fallback",
        choices=("table_first", "table_only", "analytical_only"),
        default="table_first",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        min(
            *args.batches,
            args.context,
            args.steps,
            args.a_tp,
            args.f_tp or 1,
            args.af_batch_size,
        )
        < 1
    ):
        raise SystemExit("batch, context, steps and parallel sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
