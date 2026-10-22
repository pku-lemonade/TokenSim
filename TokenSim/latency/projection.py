"""Tensor-parallel projection plans for dense and grouped low-rank attention."""

from dataclasses import dataclass

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class Projection:
    name: str
    table: str
    dtype: str
    n: int
    k: int
    groups: int = 1

    def key(self, tokens):
        result = {"dtype": self.dtype, "m": tokens, "n": self.n, "k": self.k}
        if self.table == "grouped_gemm":
            result.update(batch_size=self.groups, out_dtype="bf16")
        return result


def attention_projections(model, tp):
    spec = model.components
    if spec is None:
        from TokenSim.config.cache_config import local_kv_heads
        from math import ceil

        q = ceil(model.num_attention_heads / tp) * model.head_dim
        kv = (
            model.kv_dim // tp
            if model.kv_cache_dim is not None
            else local_kv_heads(model.num_key_value_heads, tp) * model.head_dim
        )
        return (
            Projection(
                "qkv",
                "gemm",
                model.operator_dtype("attention"),
                q + (1 if model.k_eq_v else 2) * kv,
                model.hidden_size,
            ),
            Projection(
                "output",
                "gemm",
                model.operator_dtype("attention"),
                model.hidden_size,
                q,
            ),
        )
    if model.num_attention_heads % tp or spec.o_groups % tp:
        raise ConfigurationError(
            "compressed attention TP must evenly divide heads and output groups"
        )
    dtype = model.operator_dtype("attention_gemm")
    return (
        # This merged projection is replicated, even under TP.
        Projection(
            "fused_wqa_wkv",
            "gemm",
            dtype,
            spec.q_lora_rank + model.head_dim,
            model.hidden_size,
        ),
        Projection(
            "wq_b",
            "gemm",
            dtype,
            model.num_attention_heads * model.head_dim // tp,
            spec.q_lora_rank,
        ),
        Projection(
            "wo_a",
            "grouped_gemm",
            dtype,
            spec.o_lora_rank,
            model.num_attention_heads * model.head_dim // spec.o_groups,
            groups=spec.o_groups // tp,
        ),
        Projection(
            "wo_b",
            "gemm",
            dtype,
            model.hidden_size,
            spec.o_groups * spec.o_lora_rank // tp,
        ),
    )
