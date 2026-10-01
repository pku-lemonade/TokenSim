"""DSV4 projection geometry from vLLM attention.py / nvidia/ops/o_proj.py."""

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
    spec = model.deepseek_v4
    if spec is None or model.num_attention_heads % tp or spec.o_groups % tp:
        raise ConfigurationError("DSV4 TP must evenly divide heads and output groups")
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
