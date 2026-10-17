"""Isolated latency probes and accounting checks for the LPX projection.

Zero launch cost and zero bandwidth ramp are diagnostic lower bounds.
The hybrid-window probe changes latency only and does not validate placement.
These probes do not calibrate devices or modify the serving experiment.
"""

from collections import defaultdict
from dataclasses import replace

from TokenSim.hardware import HardwareContext
from scripts.lpx.operators import Operators


def altered_hardware(hardware, *, zero_rubin_launch=False, zero_ramp=False):
    devices = [
        (
            replace(device, analytical={**device.analytical, "kernel_launch_us": 0.0})
            if zero_rubin_launch and device.device_id == "rubin"
            else device
        )
        for device in hardware.devices
    ]
    links = [
        replace(link, half_bandwidth_bytes=0.0) if zero_ramp else link
        for link in hardware.links
    ]
    return HardwareContext.in_memory(
        devices=devices,
        models=list(hardware.models),
        links=links,
        topologies=list(hardware.topologies),
    )


def task_metrics(tasks):
    seconds = sum(task.seconds for task in tasks)
    return {"step_us": seconds * 1e6, "tps_per_user": 1 / seconds}


def latency_probes(hardware, model, case, *, context=404200, batch=1):
    def forward(source, target=model):
        return Operators(source, target).forward(
            case.decode, "decode", batch, 1, context
        )

    baseline = forward(hardware)
    no_launch = forward(altered_hardware(hardware, zero_rubin_launch=True))
    no_ramp = forward(altered_hardware(hardware, zero_ramp=True))
    local = forward(hardware, replace(model, sliding_window=128))
    hybrid = []
    for full, windowed in zip(baseline, local, strict=True):
        if full.name != windowed.name:
            raise ValueError("hybrid-window probe requires matching task structures")
        layer = (
            int(full.name.split(":")[0][5:]) if full.name.startswith("layer") else None
        )
        hybrid.append(windowed if layer is not None and layer % 2 == 0 else full)
    breakdown = defaultdict(float)
    for task in baseline:
        if ":attention" in task.name:
            category = "attention_including_collectives"
        elif ":experts" in task.name:
            category = "ffn_including_collectives"
        elif any(
            word in task.name for word in ("dispatch", "return", "handoff", "gather")
        ):
            category = "explicit_communication"
        else:
            category = "other_compute"
        breakdown[category] += task.seconds * 1e6
    return {
        "case": case.name,
        "batch": batch,
        "context": context,
        "baseline": task_metrics(baseline),
        "zero_rubin_launch_lower_bound": task_metrics(no_launch),
        "zero_bandwidth_ramp_lower_bound": task_metrics(no_ramp),
        "alternating_local_full_latency_probe": task_metrics(hybrid),
        "breakdown_us": dict(breakdown),
        "rubin_launch_contribution_us": sum(t.seconds for t in baseline) * 1e6
        - sum(t.seconds for t in no_launch) * 1e6,
        "resource_pools": sorted({task.pool for task in baseline}),
    }


def saved_result_diagnostics(hardware, model, summary):
    rows = [row for row in summary["results"] if row["status"] == "ok"]
    baseline = max(
        (row for row in rows if row["mode"] == "rubin"),
        key=lambda row: row["tokens_per_s_per_mw"],
    )
    modes = {}
    for mode in ("rubin", "draft", "af", "pd"):
        valid = [row for row in rows if row["mode"] == mode]
        if not valid:
            continue
        efficient = max(valid, key=lambda row: row["tokens_per_s_per_mw"])
        fastest = max(valid, key=lambda row: row["tokens_per_s_per_user"])
        demand = efficient["pool_service_s"]
        turn = efficient["timings"]["turn_s"]
        modes[mode] = {
            "efficient_case": efficient["case"],
            "efficient_batch": efficient["batch"],
            "fastest_tps_per_user": fastest["tokens_per_s_per_user"],
            "output_throughput_ratio": efficient["output_tokens_per_s"]
            / baseline["output_tokens_per_s"],
            "power_ratio": efficient["power_watts"] / baseline["power_watts"],
            "efficiency_ratio": efficient["tokens_per_s_per_mw"]
            / baseline["tokens_per_s_per_mw"],
            "inventory": efficient["inventory"],
            "serialized_pool_time_fractions": {
                pool: seconds / turn for pool, seconds in demand.items()
            },
        }
    workload = summary["workload"]
    handoffs = []
    for row in rows:
        if "kv_handoff_bytes" not in row["timings"]:
            continue
        full = row["timings"]["kv_handoff_bytes"]
        delta = (
            full
            * workload["new_tokens"]
            / (workload["cached_tokens"] + workload["new_tokens"])
        )
        link = hardware.link("ethernet1600Gb" if row["mode"] == "pd" else "nvlink6")
        saving = (
            link.analytical_transfer_us(full) - link.analytical_transfer_us(delta)
        ) / 1e6
        handoffs.append(
            {
                "case": row["case"],
                "batch": row["batch"],
                "full_bytes": full,
                "delta_bytes_if_prefix_already_resident": delta,
                "conditional_saved_seconds": saving,
                "conditional_turn_speedup": row["timings"]["turn_s"]
                / (row["timings"]["turn_s"] - saving),
            }
        )
    return {
        "modes": modes,
        "handoff_policy_probes": handoffs,
        "model_scaling": {
            "total_parameters": model.total_params(),
            "experts": model.moe.num_experts,
            "activated_routed_parameters_per_token": model.expert_params()
            * model.moe.num_experts_per_tok
            * model.num_layers,
            "fp8_expert_storage_bytes": model.expert_params()
            * model.moe.num_experts
            * model.num_layers,
            "four_bit_expert_storage_lower_bound_bytes_excluding_scales": model.expert_params()
            * model.moe.num_experts
            * model.num_layers
            / 2,
        },
    }
