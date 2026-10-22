"""Isolated numerical hypotheses for the Gemma output-speed context trend."""

from dataclasses import replace

import numpy as np

from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import DeviceCatalog, LinkCatalog, SourcedValue
from scripts.lpx.aa_cost import Costs
from scripts.lpx.aa_model import Capacity, Placement
from scripts.lpx.aa_search import request_cost


MODES = ("lpx", "af", "af-global")
TARGET_RATIO = 3431 / 3382
HARDWARE_KEYS = {
    "memory_scale",
    "compute_scale",
    "link_bw_scale",
    "link_latency_scale",
    "memory_efficiency",
    "attention_mfu",
}
COST_KEYS = {
    "kernel_us",
    "cross_latency_us",
    "collective",
    "rho_in",
    "rho_out",
    "weight_dtype",
    "kv_dtype",
    "timing_window",
}


class TrendCosts(Costs):
    """An optional global-attention-only bandwidth-efficiency hypothesis."""

    global_attention_efficiency = None

    def attention(self, device, tp, full, *, context, tokens=1):
        estimate = super().attention(device, tp, full, context=context, tokens=tokens)
        if not full or self.global_attention_efficiency is None:
            return estimate
        baseline = self.devices[device].analytical["memory_efficiency"]
        memory = estimate.memory_time_us * baseline / self.global_attention_efficiency
        return replace(
            estimate,
            memory_time_us=memory,
            latency_us=max(memory, estimate.compute_time_us) + estimate.overhead_us,
            bottleneck="memory" if memory >= estimate.compute_time_us else "compute",
        )


def make_costs(hardware, model, settings=None):
    settings = dict(settings or {})
    unknown = (
        set(settings) - HARDWARE_KEYS - COST_KEYS - {"global_attention_efficiency"}
    )
    if unknown:
        raise ConfigurationError(f"unknown trend settings: {sorted(unknown)}")
    for key in HARDWARE_KEYS & settings.keys():
        number(
            settings[key],
            key,
            positive=key != "link_latency_scale",
            maximum=1 if key in {"memory_efficiency", "attention_mfu"} else None,
        )
    efficiency = settings.get("global_attention_efficiency")
    if efficiency is not None:
        number(efficiency, "global attention efficiency", positive=True, maximum=1)

    def scaled(value, factor):
        return SourcedValue(value.value * factor, "assumption:context-sensitivity", "D")

    devices = []
    for spec in hardware.devices:
        if spec.device_id not in {"rubin", "groq_lp30"}:
            devices.append(spec)
            continue
        memory = settings.get("memory_scale", 1)
        compute = settings.get("compute_scale", 1)
        analytical = {
            **spec.analytical,
            **{
                key: settings[key]
                for key in ("memory_efficiency", "attention_mfu")
                if key in settings
            },
        }
        devices.append(
            replace(
                spec,
                peak_compute={
                    k: scaled(v, compute) for k, v in spec.peak_compute.items()
                },
                memory_bandwidth_bytes_per_s=scaled(
                    spec.memory_bandwidth_bytes_per_s, memory
                ),
                on_chip_bandwidth_bytes_per_s=(
                    scaled(spec.on_chip_bandwidth_bytes_per_s, memory)
                    if spec.on_chip_bandwidth_bytes_per_s is not None
                    else None
                ),
                analytical=analytical,
                analytical_source_id="assumption:context-sensitivity",
                analytical_grade="D",
            )
        )
    altered = replace(
        hardware,
        devices=DeviceCatalog(devices),
        links=LinkCatalog(
            replace(
                link,
                bandwidth_per_direction_bytes_per_s=(
                    link.bandwidth_per_direction_bytes_per_s
                    * settings.get("link_bw_scale", 1)
                ),
                latency_us=link.latency_us * settings.get("link_latency_scale", 1),
                source_id="assumption:context-sensitivity",
                grade="D",
            )
            for link in hardware.links
        ),
    )
    cost_settings = {"collective": "scheduled-direct"}
    cost_settings.update(
        {key: value for key, value in settings.items() if key in COST_KEYS}
    )
    costs = TrendCosts(altered, model, **cost_settings)
    costs.global_attention_efficiency = efficiency
    return costs


def clear_caches():
    for name in ("estimate", "communication", "components", "decode", "prefill"):
        getattr(Costs, name).cache_clear()
    Capacity.place.cache_clear()


def point_case(point):
    def placement(raw):
        if raw is None:
            return None
        return Placement(**{**raw, "stages": tuple(tuple(s) for s in raw["stages"])})

    return dict(
        name=point["name"],
        mode=point["mode"],
        decode=tuple(placement(p) for p in point["decode"]),
        prefill=placement(point["prefill"]),
    )


def generation(costs, case, workload, *, post_first_gap_us=0):
    number(post_first_gap_us, "post-first-token gap")
    request = request_cost(costs, case, workload)
    intervals = request["durations_us"]
    if costs.timing_window == "tail80":
        raise ConfigurationError("post-first-gap probes require the full output window")
    total = float(np.sum(intervals)) + post_first_gap_us
    tokens = len(intervals) / costs.rho_out
    return dict(
        tps=tokens * 1e6 / total,
        mean_native_step_us=float(np.mean(intervals)),
        output_intervals=len(intervals),
        native_tokens_per_aa_token=costs.rho_out,
        post_first_gap_us=post_first_gap_us,
        ttft_excluded=True,
    )


def solve_bounded(function, target, low, high, *, increasing, iterations=48):
    """Bisection of a bounded monotone response; unreachable targets stay explicit."""
    low_value, high_value = function(low), function(high)
    if not min(low_value, high_value) <= target <= max(low_value, high_value):
        return None
    for _ in range(iterations):
        middle = (low + high) / 2
        value = function(middle)
        if (value < target) == increasing:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def fixed_profiles():
    profiles = [("baseline", {})]
    axes = {
        "kernel_us": [0, 0.01, 0.05, 0.1, 0.4, 1],
        "cross_latency_us": [0, 0.1, 0.5, 1, 5],
        "link_latency_scale": [0, 0.1, 0.5, 2, 5],
        "link_bw_scale": [0.25, 0.5, 2, 4, 16],
        "memory_scale": [0.25, 0.5, 2, 4, 16],
        "compute_scale": [0.25, 0.5, 2, 4, 16],
        "memory_efficiency": [0.1, 0.3, 0.6, 1],
        "attention_mfu": [0.1, 0.3, 1],
        "global_attention_efficiency": [0.01, 0.05, 0.1, 0.3, 1],
        "rho_in": [0.8, 1.2],
        "rho_out": [0.8, 1.2],
        "timing_window": ["tail80"],
        "collective": ["native-ring"],
    }
    profiles += [(f"{key}-{v}", {key: v}) for key, vs in axes.items() for v in vs]
    profiles += [
        ("fp8-weights", {"weight_dtype": "fp8"}),
        ("fp8-kv", {"kv_dtype": "fp8"}),
        ("fp8-both", {"weight_dtype": "fp8", "kv_dtype": "fp8"}),
        (
            "idealized-fast",
            dict(kernel_us=0, link_latency_scale=0, memory_scale=16, compute_scale=16),
        ),
        (
            "slow-constant",
            dict(
                kernel_us=1, link_latency_scale=5, memory_scale=0.5, compute_scale=0.5
            ),
        ),
    ]
    return profiles
