"""FLOP and byte accounting for the operators TokenSim schedules.

Every function returns an :class:`OperatorWork`. These formulas are the single
source of truth for the analytical estimators, the ``*_analysis`` tables, and
the documentation; the numbers below intentionally match
``docs/task-TokenSim-Groq-Operator-Data.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from TokenSim.hardware.device import dtype_bytes, normalize_dtype

ACTIVATION_DTYPE_FOR_WEIGHT_ONLY = "fp16"


@dataclass(frozen=True)
class OperatorWork:
    flops: float
    weight_bytes: float = 0.0
    activation_bytes: float = 0.0
    kv_bytes: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> float:
        return self.weight_bytes + self.activation_bytes + self.kv_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes if self.total_bytes > 0 else float("inf")


def activation_bytes_for(dtype: str) -> float:
    """Bytes per activation element given the storage dtype of the weights."""
    canonical = normalize_dtype(dtype)
    if canonical.endswith("_wo"):
        return dtype_bytes(ACTIVATION_DTYPE_FOR_WEIGHT_ONLY)
    if canonical in {"int4", "nvfp4"}:
        # 4-bit activations are not used for LLM inference; assume fp8 activations.
        return 1.0
    return dtype_bytes(canonical)


def gemm_work(m: int, n: int, k: int, dtype: str) -> OperatorWork:
    b_w = dtype_bytes(dtype)
    b_a = activation_bytes_for(dtype)
    return OperatorWork(
        flops=2.0 * m * n * k,
        weight_bytes=float(k) * n * b_w,
        activation_bytes=(float(m) * k + float(m) * n) * b_a,
        details={"m": m, "n": n, "k": k, "dtype": normalize_dtype(dtype)},
    )


def context_attention_work(
    batch_size: int,
    input_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    attn_dtype: str,
    kv_cache_dtype: str,
    window_size: int = 0,
    causal: bool = True,
) -> OperatorWork:
    """Prefill attention core: QK^T, softmax, and PV for one padded batch.

    Softmax FLOPs are folded into the two matmuls (they are a small fraction
    and always fused). Causal masking halves the useful key range; a sliding
    window bounds the keys attended by each query.
    """
    h_q = num_heads * head_dim
    h_kv = num_kv_heads * head_dim
    b_a = dtype_bytes(attn_dtype)
    b_kv = dtype_bytes(kv_cache_dtype)
    keys_per_query = float(input_seq_len)
    if window_size and window_size > 0:
        keys_per_query = min(keys_per_query, float(window_size))
        causal_factor = 1.0 if window_size < input_seq_len else 0.5
    else:
        causal_factor = 0.5 if causal else 1.0
    flops = 4.0 * batch_size * input_seq_len * keys_per_query * h_q * causal_factor
    tokens = float(batch_size) * input_seq_len
    return OperatorWork(
        flops=flops,
        activation_bytes=tokens * 2 * h_q * b_a,  # read Q, write O
        kv_bytes=tokens * 4 * h_kv * b_kv,  # read K,V and write them to the cache
        details={
            "batch_size": batch_size,
            "input_seq_len": input_seq_len,
            "keys_per_query": keys_per_query,
            "causal_factor": causal_factor,
        },
    )


def generation_attention_work(
    batch_size: int,
    context_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    attn_dtype: str,
    kv_cache_dtype: str,
    window_size: int = 0,
) -> OperatorWork:
    """Decode attention core: one query token per sequence against the KV cache."""
    h_q = num_heads * head_dim
    h_kv = num_kv_heads * head_dim
    b_a = dtype_bytes(attn_dtype)
    b_kv = dtype_bytes(kv_cache_dtype)
    keys = float(context_len)
    if window_size and window_size > 0:
        keys = min(keys, float(window_size))
    return OperatorWork(
        flops=4.0 * batch_size * keys * h_q,
        activation_bytes=float(batch_size) * 2 * h_q * b_a,
        kv_bytes=float(batch_size) * (2 * keys * h_kv * b_kv + 2 * h_kv * b_kv),
        details={"batch_size": batch_size, "context_len": context_len, "keys": keys},
    )


def moe_work(
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    top_k: int,
    num_experts: int,
    tp_size: int,
    ep_size: int,
    dtype: str,
    gated: bool = True,
    imbalance: float = 1.0,
) -> OperatorWork:
    """Expert FFN work on one rank for ``num_tokens`` tokens entering the layer.

    ``imbalance`` scales the balanced per-rank routed load (1.0 = uniform);
    callers derive it from the routing histogram or the distribution label.
    """
    g = 2 if gated else 1
    b_w = dtype_bytes(dtype)
    b_a = activation_bytes_for(dtype)
    local_experts = max(1, -(-num_experts // ep_size))
    local_inter = max(1, -(-inter_size // tp_size))
    routed_pairs = float(num_tokens) * top_k
    local_pairs = routed_pairs / ep_size * imbalance
    active_local_experts = min(float(local_experts), max(1.0, local_pairs))
    flops = local_pairs * 2.0 * hidden_size * local_inter * (g + 1)
    weight_bytes = active_local_experts * (g + 1) * hidden_size * local_inter * b_w
    activation_bytes = local_pairs * (2 * hidden_size + (g + 1) * local_inter) * b_a
    return OperatorWork(
        flops=flops,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        details={
            "local_experts": local_experts,
            "local_inter": local_inter,
            "local_pairs": local_pairs,
            "active_local_experts": active_local_experts,
            "imbalance": imbalance,
        },
    )


_ELEMENTWISE_FLOPS_PER_ELEMENT = {
    "rmsnorm": 4.0,
    "layernorm": 5.0,
    "rope": 3.0,
    "residual_add": 1.0,
    "swiglu": 4.0,
    "gelu": 6.0,
    "data_reorder": 0.0,
    "other_residual": 1.0,
}

_ELEMENTWISE_BYTES_PER_ELEMENT = {
    # (reads + writes) expressed in element units of the token x hidden tensor
    "rmsnorm": 2.0,
    "layernorm": 2.0,
    "rope": 2.0,
    "residual_add": 3.0,
    "swiglu": 3.0,  # read gate and up, write one
    "gelu": 2.0,
    "data_reorder": 2.0,
    "other_residual": 2.0,
}


def elementwise_work(op_name: str, num_tokens: int, hidden_size: int, dtype: str) -> OperatorWork:
    op = op_name.lower()
    if op not in _ELEMENTWISE_FLOPS_PER_ELEMENT:
        raise ValueError(f"unsupported elementwise op {op_name!r}")
    elements = float(num_tokens) * hidden_size
    b = activation_bytes_for(dtype)
    extra_bytes = hidden_size * b if op in {"rmsnorm", "layernorm"} else 0.0  # weight vector
    return OperatorWork(
        flops=_ELEMENTWISE_FLOPS_PER_ELEMENT[op] * elements,
        activation_bytes=_ELEMENTWISE_BYTES_PER_ELEMENT[op] * elements * b + extra_bytes,
        details={"op_name": op, "num_tokens": num_tokens, "hidden_size": hidden_size},
    )


def work_for_table(table: str, key: Mapping[str, Any], *, gated: bool = True, imbalance: float = 1.0) -> OperatorWork:
    """Dispatch a table query key to the matching work formula."""
    if table == "gemm":
        return gemm_work(int(key["m"]), int(key["n"]), int(key["k"]), str(key["dtype"]))
    if table == "context_attention":
        return context_attention_work(
            int(key["batch_size"]),
            int(key["input_seq_len"]),
            int(key["num_heads"]),
            int(key["num_kv_heads"]),
            int(key["head_dim"]),
            str(key["attn_dtype"]),
            str(key["kv_cache_dtype"]),
            int(key.get("window_size", 0)),
        )
    if table == "generation_attention":
        return generation_attention_work(
            int(key["batch_size"]),
            int(key["context_len"]),
            int(key["num_heads"]),
            int(key["num_kv_heads"]),
            int(key["head_dim"]),
            str(key["attn_dtype"]),
            str(key["kv_cache_dtype"]),
            int(key.get("window_size", 0)),
        )
    if table == "moe":
        return moe_work(
            int(key["num_tokens"]),
            int(key["hidden_size"]),
            int(key["inter_size"]),
            int(key["top_k"]),
            int(key["num_experts"]),
            int(key["tp_size"]),
            int(key["ep_size"]),
            str(key["dtype"]),
            gated=gated,
            imbalance=imbalance,
        )
    if table == "elementwise":
        return elementwise_work(str(key["op_name"]), int(key["num_tokens"]), int(key["hidden_size"]), str(key["dtype"]))
    raise ValueError(f"no work formula for table {table!r}")
