from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from TokenSim.dtypes import canonicalize_dtype, dtype_size_bytes
from TokenSim.operator_data.schema import decode_positive_vector, validate_batch_key


@dataclass(frozen=True)
class HardwareProfile:
    device: str
    backend: str
    peak_compute_by_dtype: Mapping[str, float]
    memory_bandwidth_bytes_per_s: float
    memory_capacity_bytes: int | None = None
    interconnect_bandwidth_bytes_per_s: float | None = None

    def peak_compute(self, dtype: str) -> float:
        normalized = canonicalize_dtype(dtype, role="activation_dtype")
        peak_compute_by_dtype = {
            canonicalize_dtype(name, role="activation_dtype"): value
            for name, value in self.peak_compute_by_dtype.items()
        }
        try:
            value = float(peak_compute_by_dtype[normalized])
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("calibration multiplier must be finite and positive")
        if not math.isfinite(self.offset_us) or self.offset_us < 0:
            raise ValueError("fixed overhead offset_us must be finite and non-negative")

    def apply(self, roofline_us: float) -> float:
        value = roofline_us * self.multiplier + self.offset_us
        if not math.isfinite(value) or value < 0:
            raise ValueError("calibration produced an invalid latency")
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
    validate_batch_key(table_name, key)
    if key.get("device", hardware.device) != hardware.device:
        raise ValueError("shape device does not match hardware profile")
    if key.get("backend", hardware.backend) != hardware.backend:
        raise ValueError("shape backend does not match hardware profile")

    formula_inputs: dict[str, Any]
    if table_name in {"gemm", "batched_gemm"}:
        weight_dtype = canonicalize_dtype(
            str(key["weight_dtype"]), role="weight_dtype"
        )
        dtype = canonicalize_dtype(
            str(key["activation_dtype"]), role="activation_dtype"
        )
        m, n, k = (int(key[field]) for field in ("m", "n", "k"))
        weight_element_bytes = dtype_size_bytes(
            weight_dtype, role="weight_dtype"
        )
        activation_element_bytes = dtype_size_bytes(
            dtype, role="activation_dtype"
        )
        flops = 2.0 * m * n * k
        weight_bytes = k * n * weight_element_bytes
        activation_bytes = (m * k + m * n) * activation_element_bytes
        kv_bytes = 0.0
        formula = "flops=2*m*n*k; weights=k*n; activations=m*k+m*n"
        formula_inputs = {
            "m": m,
            "n": n,
            "k": k,
            "weight_dtype": weight_dtype,
            "activation_dtype": dtype,
            "weight_element_bytes": weight_element_bytes,
            "activation_element_bytes": activation_element_bytes,
        }
        if table_name == "batched_gemm":
            groups = int(key["groups"])
            flops *= groups
            weight_bytes *= groups
            activation_bytes *= groups
            formula_inputs["groups"] = groups
            formula += "; multiply work and bytes by independent groups; one launch"
    elif table_name == "mla_attention":
        dtype = canonicalize_dtype(str(key["activation_dtype"]), role="activation_dtype")
        a_bytes = dtype_size_bytes(dtype, role="activation_dtype")
        c_bytes = dtype_size_bytes(str(key["kv_cache_dtype"]), role="kv_cache_dtype")
        contexts = decode_positive_vector(key["context_lengths"])
        b, q, h, r, dr, dn, dv = (int(key[n]) for n in
            ("batch_size", "query_len", "num_heads", "kv_lora_rank", "rope_head_dim", "nope_head_dim", "v_head_dim"))
        tokens = b * q
        cache_write = tokens * (r + dr) * c_bytes
        if key["phase"] == "decode":
            pairs = sum(contexts)
            matmul_flops = 2.0 * h * pairs * (2 * r + dr)
            cache_read = pairs * (r + dr) * c_bytes
            kv_bytes = cache_read + cache_write
            activation_bytes = b * h * (2 * r + dr) * a_bytes
            formula = "absorbed decode: 2*H*sum(S)*(2*r+dr); compressed cache shared across heads, replicated across TP ranks"
        else:
            # Causal useful work; no cached prefix. Expanded K/V are ephemeral,
            # while the persistent cache still contains only latent+RoPE.
            pairs = sum(s * (s + 1) // 2 for s in contexts)
            matmul_flops = 2.0 * h * pairs * (dn + dr + dv)
            cache_read = 0
            kv_bytes = cache_write
            activation_bytes = 2 * tokens * h * (dn + dr + dv) * a_bytes
            formula = "expanded causal prefill: 2*H*sum(S*(S+1)/2)*(dn+dr+dv); compressed cache write"
        softmax_flops = 5.0 * h * pairs
        flops = matmul_flops + softmax_flops
        weight_bytes = 0.0
        formula_inputs = dict(key, matmul_flops=matmul_flops, softmax_flops=softmax_flops,
            cache_read_bytes=cache_read, cache_write_bytes=cache_write,
            resident_cache_bytes=sum(contexts)*(r+dr)*c_bytes,
            assumptions="one logical cache read; fused softmax without score materialization; softmax uses activation peak approximation")
    elif table_name in {"context_attention", "generation_attention", "generation_attention_batch"}:
        attention_type = _validate_attention_type(key)
        dtype = canonicalize_dtype(
            str(key["activation_dtype"]), role="activation_dtype"
        )
        element_bytes = dtype_size_bytes(dtype, role="activation_dtype")
        kv_element_bytes = dtype_size_bytes(
            str(key["kv_cache_dtype"]), role="kv_cache_dtype"
        )
        batch = int(key["batch_size"])
        heads = int(key["num_heads"])
        kv_heads = int(key["num_kv_heads"])
        head_dim = int(key["head_dim"])
        window_size = int(key["window_size"])
        h_q = heads * head_dim
        h_kv = kv_heads * head_dim
        if table_name == "context_attention":
            seq = int(key["input_seq_len"])
            flops = 4.0 * batch * seq * seq * h_q
            kv_bytes = 2.0 * batch * seq * h_kv * kv_element_bytes
            activation_bytes = 2.0 * batch * seq * h_q * element_bytes
            effective_context = seq
        else:
            contexts = (decode_positive_vector(key["context_lengths"])
                        if table_name == "generation_attention_batch"
                        else (int(key["context_len"]),) * batch)
            effective = tuple(min(seq, window_size) if window_size > 0 else seq for seq in contexts)
            seq = max(contexts)
            effective_context = sum(effective) / batch
            flops = 4.0 * sum(effective) * h_q
            kv_bytes = (
                2.0 * batch * effective_context * h_kv * kv_element_bytes
            )
            activation_bytes = 2.0 * batch * h_q * element_bytes
        weight_bytes = 0.0
        formula = (
            "attention_core: flops=4*batch*query_tokens*effective_context*Q_width; "
            "kv_bytes=2*batch*effective_context*KV_width*kv_element_bytes"
        )
        formula_inputs = {
            "attention_type": attention_type,
            "batch_size": batch,
            "context_length": seq,
            "effective_context_length": effective_context,
            "window_size": window_size,
            "query_heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "query_width": h_q,
            "kv_width": h_kv,
            "activation_dtype": dtype,
            "kv_cache_dtype": canonicalize_dtype(
                str(key["kv_cache_dtype"]), role="kv_cache_dtype"
            ),
        }
        if table_name != "context_attention":
            formula_inputs["context_lengths"] = contexts
            formula_inputs["effective_context_lengths"] = effective
    elif table_name in {"moe", "moe_group"}:
        operation = str(key["operation"]).strip().lower()
        if operation not in {"routed_expert", "shared_expert"}:
            raise ValueError(
                "analytical MoE operation must be routed_expert or shared_expert; "
                f"got {operation!r}"
            )
        weight_dtype = canonicalize_dtype(
            str(key["weight_dtype"]), role="weight_dtype"
        )
        dtype = canonicalize_dtype(
            str(key["activation_dtype"]), role="activation_dtype"
        )
        weight_element_bytes = dtype_size_bytes(
            weight_dtype, role="weight_dtype"
        )
        activation_element_bytes = dtype_size_bytes(
            dtype, role="activation_dtype"
        )
        tokens = int(key["num_tokens"])
        hidden = int(key["hidden_size"])
        inter = int(key["inter_size"])
        _ = int(key["top_k"])
        experts = int(key["num_experts"])
        tp_size = int(key["tp_size"])
        if tp_size < 1:
            raise ValueError("moe tp_size must be positive")
        if inter % tp_size:
            raise ValueError(
                f"MoE inter_size={inter} is not divisible by tp_size={tp_size}; "
                "implicit uneven TP shards are unsupported"
            )
        local_inter = inter // tp_size
        # gate/up (two projections) plus down, only for activated experts.
        # num_tokens is already the number of routed token-expert assignments.
        flops = 6.0 * tokens * hidden * local_inter
        active_experts = min(experts, tokens)
        weight_bytes = (
            3.0
            * active_experts
            * hidden
            * local_inter
            * weight_element_bytes
        )
        activation_bytes = (
            tokens * hidden + 2 * tokens * local_inter + tokens * hidden
        ) * activation_element_bytes
        kv_bytes = 0.0
        formula = (
            "SwiGLU gate/up/down: flops=6*tokens*hidden*local_inter; "
            "weights=3*active_experts*hidden*local_inter"
        )
        formula_inputs = {
            "operation": operation,
            "num_tokens": tokens,
            "hidden_size": hidden,
            "global_intermediate_size": inter,
            "local_intermediate_size": local_inter,
            "tp_size": tp_size,
            "active_experts": active_experts,
            "weight_dtype": weight_dtype,
            "activation_dtype": dtype,
            "projection_count": 3,
        }
        if table_name == "moe_group":
            formula_inputs["expert_token_counts"] = decode_positive_vector(key["expert_token_counts"])
            formula_inputs["execution_boundary"] = key["execution_boundary"]
    elif table_name == "elementwise":
        dtype = canonicalize_dtype(
            str(key["activation_dtype"]), role="activation_dtype"
        )
        source_dtype = canonicalize_dtype(
            str(key["source_dtype"]), role="activation_dtype"
        )
        target_dtype = canonicalize_dtype(
            str(key["target_dtype"]), role="activation_dtype"
        )
        source_element_bytes = dtype_size_bytes(
            source_dtype, role="activation_dtype"
        )
        target_element_bytes = dtype_size_bytes(
            target_dtype, role="activation_dtype"
        )
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
            "routing_weight_multiply": 1.0,
            "routed_reduce": 1.0,
            "shared_sum": 1.0,
            "encode": 1.0,
            "decode": 1.0,
            "embedding_lookup": 0.0,
            "argmax": 1.0,
        }.get(op_name)
        if flop_factor is None:
            raise ValueError(f"unsupported elementwise op_name={op_name!r}")
        is_conversion = op_name in {"encode", "decode"}
        if not is_conversion and (
            source_dtype != dtype or target_dtype != dtype
        ):
            raise ValueError(
                f"elementwise {op_name!r} requires source_dtype and "
                "target_dtype to equal activation_dtype"
            )
        flops = flop_factor * tokens * hidden
        weight_bytes = 0.0
        element_count = tokens * hidden
        if is_conversion:
            activation_bytes = element_count * (
                source_element_bytes + target_element_bytes
            )
        else:
            tensor_count = 3.0 if op_name == "residual_add" else 2.0
            activation_bytes = tensor_count * element_count * source_element_bytes
            if op_name == "embedding_lookup":
                # Read selected weight rows, int32 token IDs; write activations.
                activation_bytes += tokens * 4
            elif op_name == "argmax":
                # One read of BF16 logits, one int32 output token per request.
                activation_bytes = element_count * source_element_bytes + tokens * 4
        kv_bytes = 0.0
        formula = (
            "elementwise flops=op_factor*num_tokens*hidden_size; "
            "bytes=input/output tensor traffic"
        )
        formula_inputs = {
            "op_name": op_name,
            "num_tokens": tokens,
            "hidden_size": hidden,
            "activation_dtype": dtype,
            "source_dtype": source_dtype,
            "target_dtype": target_dtype,
            "source_element_bytes": source_element_bytes,
            "target_element_bytes": target_element_bytes,
            "flop_factor": flop_factor,
        }
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
        details={
            "formula_version": "tokensim-analytical-v2",
            "formula": formula,
            "formula_inputs": formula_inputs,
            "roofline_time_us": roofline_us,
            "fixed_overhead_us": calibration.offset_us,
            "throughput_multiplier": calibration.multiplier,
            "calibrated_variable_time_us": roofline_us * calibration.multiplier,
        },
    )


def _validate_attention_type(key: Mapping[str, Any]) -> str:
    attention_type = str(key["attention_type"]).strip().lower()
    if attention_type in {"mla", "mfa"}:
        raise ValueError("MLA/MFA require a dedicated formula; use mla_attention for MLA")
    if attention_type not in {"mha", "gqa", "mqa"}:
        raise ValueError(
            f"analytical attention_type must be mha, gqa, or mqa; "
            f"got {attention_type!r}"
        )
    heads = int(key["num_heads"])
    kv_heads = int(key["num_kv_heads"])
    if heads < 1 or kv_heads < 1 or kv_heads > heads:
        raise ValueError(
            f"invalid {attention_type} head shape: query_heads={heads}, "
            f"kv_heads={kv_heads}"
        )
    if attention_type == "mha" and kv_heads != heads:
        raise ValueError(
            f"MHA requires equal query and KV heads; got {heads} and {kv_heads}"
        )
    if attention_type == "mqa" and kv_heads != 1:
        raise ValueError(f"MQA requires one KV head; got {kv_heads}")
    if attention_type == "gqa" and not (1 < kv_heads < heads):
        raise ValueError(
            f"GQA requires 1 < KV heads < query heads; got {kv_heads} and {heads}"
        )
    return attention_type
