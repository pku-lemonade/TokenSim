"""Explicit, uncalibrated LP30 execution-unit and mixed-precision accounting.

MXM rates come from the device catalog; VXM and SXM rates are independent
assumptions because published FP8 matrix peaks do not specify those units.
"""

from dataclasses import asdict, dataclass, replace
import math

from TokenSim.config.validation import integer, number
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.analytical.base import AnalyticalEstimate, Calibration
from TokenSim.operator_data.workload import work_for_table


@dataclass(frozen=True)
class LP30Pipeline:
    activation_dtype: str = "bf16"
    vector_ops_per_s: float = 20e12
    exp_ops_per_s: float = 1e12
    shuffle_bytes_per_s: float = 10e12
    vector_efficiency: float = 0.5
    shuffle_efficiency: float = 0.8
    vector_bytes: int = 320
    attention_tile_keys: int = 256
    fill_drain_us: float = 0.05
    schedule: str = "tiled"

    def __post_init__(self):
        dtype_size_bytes(self.activation_dtype, role="activation_dtype")
        for name in ("vector_ops_per_s", "exp_ops_per_s", "shuffle_bytes_per_s"):
            number(getattr(self, name), f"LP30 {name}", positive=True)
        for name in ("vector_efficiency", "shuffle_efficiency"):
            number(getattr(self, name), f"LP30 {name}", positive=True, maximum=1)
        for name in ("vector_bytes", "attention_tile_keys"):
            integer(getattr(self, name), f"LP30 {name}", minimum=1)
        number(self.fill_drain_us, "LP30 fill/drain", minimum=0)
        if self.schedule not in {"tiled", "serialized"}:
            raise ConfigurationError("LP30 schedule must be tiled or serialized")


def _rounded(elements, element_bytes, vector_bytes):
    return math.ceil(elements * element_bytes / vector_bytes) * vector_bytes


def _attention_units(table, key, work, policy):
    """Count softmax, online state updates, cache conversions and score movement."""
    heads, dim = int(key["num_heads"]), int(key["head_dim"])
    pairs = work.flops / (4 * dim)
    queries = int(key.get("input_seq_len", 1))
    rows = int(key["batch_size"]) * heads * queries
    # max, subtract, sum, normalize and scale; exponentials have a separate rate.
    scalar_ops = 5 * pairs
    cache_elements = work.kv_bytes / dtype_size_bytes(
        key["kv_cache_dtype"], role="kv_cache_dtype"
    )
    conversions = cache_elements if key["kv_cache_dtype"] != key["attn_dtype"] else 0
    mean_keys = pairs / max(rows, 1)
    tiles = max(1, math.ceil(mean_keys / policy.attention_tile_keys))
    # Running max/sum plus output rescaling for tiled online softmax.
    scalar_ops += max(0, tiles - 1) * rows * (2 * dim + 5)
    shuffle = _rounded(2 * pairs * 4, 1, policy.vector_bytes)
    return scalar_ops + conversions, pairs, shuffle, tiles, conversions


def estimate_lp30(table, key, device, calibration=None):
    policy = LP30Pipeline(**device.analytical["lpx_pipeline"])
    params = device.analytical
    work = work_for_table(table, key)
    matrix_ops, vector_ops, exps, shuffle, tiles = 0.0, 0.0, 0.0, 0.0, 1
    conversions = 0.0
    dtype = str(key.get("attn_dtype", key.get("dtype", "bf16")))
    efficiency = float(params.get("gemm_mfu", 0.8))
    if table in {"gemm", "grouped_gemm"}:
        matrix_ops = work.flops
        groups = int(key.get("batch_size", 1)) if table == "grouped_gemm" else 1
        m, n, k = (int(key[name]) for name in ("m", "n", "k"))
        external = dtype_size_bytes(policy.activation_dtype, role="activation_dtype")
        out_dtype = key.get("out_dtype", policy.activation_dtype)
        out_bytes = dtype_size_bytes(out_dtype, role="activation_dtype")
        matrix_dtype = dtype.removesuffix("_wo")
        input_elements, output_elements = groups * m * k, groups * m * n
        conversions = (
            input_elements if policy.activation_dtype != matrix_dtype else 0
        ) + (output_elements if out_dtype != matrix_dtype else 0)
        vector_ops = conversions
        shuffle = _rounded(
            input_elements * external + output_elements * out_bytes,
            1,
            policy.vector_bytes,
        )
        work = replace(
            work,
            activation_bytes=input_elements * external + output_elements * out_bytes,
        )
        tiles = max(1, math.ceil(output_elements * out_bytes / policy.vector_bytes))
    elif table in {
        "generation_attention",
        "context_attention",
        "ragged_generation_attention",
    }:
        matrix_ops = work.flops
        efficiency = float(params.get("attention_mfu", 0.6))
        vector_ops, exps, shuffle, tiles, conversions = _attention_units(
            table, key, work, policy
        )
    elif table == "elementwise":
        vector_ops = work.flops
        if key["op_name"] in {"swiglu", "gelu", "geglu"}:
            exps = int(key["num_tokens"]) * int(key["hidden_size"])
        if key.get("lpx_vector_operation") == "logit_softcap":
            elements = int(key["num_tokens"]) * int(key["hidden_size"])
            vector_ops, exps = 4 * elements, elements
            work = replace(work, flops=vector_ops)
        elif key.get("lpx_vector_operation") == "lse_merge":
            # One row per fragment holds d partial outputs and one FP32 LSE:
            # rescale and accumulate 2 ops per element, one exponential per row.
            fragments, width = int(key["num_tokens"]), int(key["hidden_size"])
            vector_ops, exps = fragments * (2 * width + 1), fragments
            output = fragments // int(key["kv_parallel_size"]) * (width - 1)
            work = replace(
                work,
                flops=vector_ops,
                activation_bytes=fragments * width * 4
                + output * dtype_size_bytes(dtype, role="activation_dtype"),
            )
        shuffle = work.activation_bytes
        tiles = max(1, math.ceil(work.activation_bytes / policy.vector_bytes))
    else:
        raise ConfigurationError(f"LP30 unit accounting does not cover {table!r}")
    for name, value in (
        ("matrix efficiency", efficiency),
        ("memory efficiency", params.get("memory_efficiency", 0.9)),
    ):
        number(value, f"LP30 {name}", positive=True, maximum=1)
    matrix_s = matrix_ops / (device.peak_compute_for(dtype).value * efficiency)
    # Exponential service occupies the same VXM pipeline as ordinary vector work.
    vector_s = (
        vector_ops / policy.vector_ops_per_s + exps / policy.exp_ops_per_s
    ) / policy.vector_efficiency
    shuffle_s = shuffle / (policy.shuffle_bytes_per_s * policy.shuffle_efficiency)
    memory_s = work.total_bytes / (
        device.on_chip_bandwidth_bytes_per_s.value
        * float(params.get("memory_efficiency", 0.9))
    )
    services = dict(mxm=matrix_s, vxm=vector_s, sxm=shuffle_s, sram=memory_s)
    if policy.schedule == "serialized":
        service_s = sum(services.values())
    else:
        # A leading tile traverses all stages; subsequent tiles share pipelines.
        service_s = (
            max(services.values())
            + (sum(services.values()) - max(services.values())) / tiles
        )
    overhead = float(params.get("kernel_launch_us", 0.2)) + policy.fill_drain_us
    calibration = calibration or Calibration()
    details = dict(
        policy=asdict(policy),
        unit_service_us={name: value * 1e6 for name, value in services.items()},
        vector_operations=vector_ops,
        exponential_operations=exps,
        conversion_elements=conversions,
        shuffle_bytes=shuffle,
        pipeline_tiles=tiles,
        fits_on_chip=work.weight_bytes <= device.on_chip_memory_bytes.value,
        status="Uncalibrated LP30 unit-throughput and pipeline assumptions.",
    )
    work = replace(work, details={**work.details, **details})
    compute_us = max(0.0, service_s * 1e6)
    return AnalyticalEstimate(
        latency_us=calibration.apply(compute_us + overhead),
        compute_time_us=compute_us,
        memory_time_us=memory_s * 1e6,
        overhead_us=overhead,
        bottleneck=max(services, key=services.get),
        work=work,
        calibration_id=calibration.calibration_id,
        source_id=f"analytical:groq_tsp:lp30-units-v1:{calibration.calibration_id}",
        details=details,
    )
