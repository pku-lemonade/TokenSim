"""Shared DeepSeek-V3 runtime factory for the operator-table AF backend."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any


from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.formats import write_trace
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.device.sourced_value import SourcedValue

ROOT = Path(__file__).resolve().parents[1]
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
    options = getattr(executor.trace, "trace_config", None)
    kwargs = {"trace_config": options} if options is not None else {}
    trace_format = getattr(executor.trace, "trace_format", "json")
    writer = write_chrome_trace if trace_format == "json" else write_trace
    if trace_format != "json":
        kwargs["trace_format"] = trace_format
    return writer(
        path,
        executor.trace.events,
        metadata or {"timing": "simulated operator-table service time"},
        **kwargs,
    )
