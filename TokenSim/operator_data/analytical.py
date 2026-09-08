from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "int8": 1,
}


@dataclass(frozen=True)
class HardwareProfile:
    device: str
    backend: str
    peak_compute_by_dtype: Mapping[str, float]
    memory_bandwidth_bytes_per_s: float
    memory_capacity_bytes: int | None = None
    interconnect_bandwidth_bytes_per_s: float | None = None

    def peak_compute(self, dtype: str) -> float:
        normalized = dtype.lower()
        try:
            value = float(self.peak_compute_by_dtype[normalized])
        except KeyError as exc:
            raise ValueError(
                f"hardware {self.device!r} has no peak compute for dtype={dtype!r}"
            ) from exc
        if value <= 0 or self.memory_bandwidth_bytes_per_s <= 0:
            raise ValueError("hardware throughput and bandwidth must be positive")
        return value


@dataclass(frozen=True)
class Calibration:
    multiplier: float = 1.0
    offset_us: float = 0.0
    calibration_id: str = "uncalibrated"

    def apply(self, roofline_us: float) -> float:
        value = roofline_us * self.multiplier + self.offset_us
        if value < 0:
            raise ValueError("calibration produced a negative latency")
        return value


@dataclass(frozen=True)
class AnalyticalEstimate:
    latency_us: float
    flops: float
    weight_bytes: float
    activation_bytes: float
    kv_bytes: float
    compute_time_us: float
    memory_time_us: float
    bottleneck: str
    calibration_id: str
    details: Mapping[str, Any] = field(default_factory=dict)


def estimate_operator(
    table_name: str,
    key: Mapping[str, Any],
    hardware: HardwareProfile,
    calibration: Calibration | None = None,
) -> AnalyticalEstimate:
    calibration = calibration or Calibration()
    if key.get("device", hardware.device) != hardware.device:
        raise ValueError("shape device does not match hardware profile")
    if key.get("backend", hardware.backend) != hardware.backend:
        raise ValueError("shape backend does not match hardware profile")

    if table_name == "gemm":
        dtype = str(key["gemm_dtype"]).lower()
        m, n, k = (int(key[field]) for field in ("m", "n", "k"))
        element_bytes = _dtype_bytes(dtype)
        flops = 2.0 * m * n * k
        weight_bytes = k * n * element_bytes
        activation_bytes = (m * k + m * n) * element_bytes
        kv_bytes = 0.0
    elif table_name in {"context_attention", "generation_attention"}:
        dtype = str(key["attn_dtype"]).lower()
        element_bytes = _dtype_bytes(dtype)
        kv_element_bytes = _dtype_bytes(str(key["kv_cache_dtype"]).lower())
        batch = int(key["batch_size"])
        heads = int(key["num_heads"])
        kv_heads = int(key["num_kv_heads"])
        head_dim = int(key["head_dim"])
        h_q = heads * head_dim
        h_kv = kv_heads * head_dim
        if table_name == "context_attention":
            seq = int(key["input_seq_len"])
            flops = 4.0 * batch * seq * seq * h_q
            kv_bytes = 2.0 * batch * seq * h_kv * kv_element_bytes
            activation_bytes = 2.0 * batch * seq * h_q * element_bytes
        else:
            seq = int(key["context_len"])
            flops = 4.0 * batch * seq * h_q
            kv_bytes = 2.0 * batch * seq * h_kv * kv_element_bytes
            activation_bytes = 2.0 * batch * h_q * element_bytes
        weight_bytes = 0.0
    elif table_name == "moe":
        dtype = str(key["moe_dtype"]).lower()
        element_bytes = _dtype_bytes(dtype)
        tokens = int(key["num_tokens"])
        hidden = int(key["hidden_size"])
        inter = int(key["inter_size"])
        _ = int(key["top_k"])
        experts = int(key["num_experts"])
        tp_size = int(key["tp_size"])
        if tp_size < 1:
            raise ValueError("moe tp_size must be positive")
        local_inter = (inter + tp_size - 1) // tp_size
        # gate/up (two projections) plus down, only for activated experts.
        # num_tokens is already the number of routed token-expert assignments.
        flops = 6.0 * tokens * hidden * local_inter
        active_experts = min(experts, max(1, tokens))
        weight_bytes = 3.0 * active_experts * hidden * local_inter * element_bytes
        activation_bytes = (
            tokens * hidden + 2 * tokens * local_inter + tokens * hidden
        ) * element_bytes
        kv_bytes = 0.0
    elif table_name == "elementwise":
        dtype = str(key["dtype"]).lower()
        element_bytes = _dtype_bytes(dtype)
        tokens = int(key["num_tokens"])
        hidden = int(key["hidden_size"])
        op_name = str(key["op_name"]).lower()
        flop_factor = {
            "rmsnorm": 5.0,
            "rope": 4.0,
            "residual_add": 1.0,
            "data_reorder": 0.0,
            "other_residual": 1.0,
            "swiglu": 4.0,
        }.get(op_name)
        if flop_factor is None:
            raise ValueError(f"unsupported elementwise op_name={op_name!r}")
        flops = flop_factor * tokens * hidden
        weight_bytes = 0.0
        activation_bytes = (2.0 if op_name != "residual_add" else 3.0) * (
            tokens * hidden * element_bytes
        )
        kv_bytes = 0.0
    else:
        raise ValueError(f"analytical model does not support table={table_name!r}")

    peak_compute = hardware.peak_compute(dtype)
    total_bytes = weight_bytes + activation_bytes + kv_bytes
    compute_time_us = flops / peak_compute * 1e6
    memory_time_us = total_bytes / hardware.memory_bandwidth_bytes_per_s * 1e6
    roofline_us = max(compute_time_us, memory_time_us)
    bottleneck = "compute" if compute_time_us >= memory_time_us else "memory"
    return AnalyticalEstimate(
        latency_us=calibration.apply(roofline_us),
        flops=flops,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        kv_bytes=kv_bytes,
        compute_time_us=compute_time_us,
        memory_time_us=memory_time_us,
        bottleneck=bottleneck,
        calibration_id=calibration.calibration_id,
        details={"roofline_time_us": roofline_us},
    )


def _dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype={dtype!r}") from exc
