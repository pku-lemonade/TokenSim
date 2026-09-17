from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SCHEMA_VERSION = 1
LATENCY_UNIT = "us"
VALUE_FIELDS = ("latency_us", "source_id")

COLLECTIVE_OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send_recv")
ELEMENTWISE_OPS = (
    "rmsnorm",
    "layernorm",
    "rope",
    "residual_add",
    "swiglu",
    "gelu",
    "data_reorder",
    "other_residual",
)


@dataclass(frozen=True)
class AxisSpec:
    """A key field along which controlled interpolation is permitted."""

    name: str
    scale: str = "log"  # "log": geometric interpolation (sizes); "linear": arithmetic

    def __post_init__(self) -> None:
        if self.scale not in {"log", "linear"}:
            raise ValueError(f"axis {self.name!r}: scale must be log or linear")


@dataclass(frozen=True)
class TableSpec:
    name: str
    key_fields: tuple[str, ...]
    axes: tuple[AxisSpec, ...]
    dtype_fields: tuple[str, ...]
    description: str

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    @property
    def discrete_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self.key_fields if field not in self.axis_names)

    @property
    def required_fields(self) -> tuple[str, ...]:
        return self.key_fields + VALUE_FIELDS

    @property
    def perf_filename(self) -> str:
        return f"{self.name}_perf.parquet"

    @property
    def analysis_filename(self) -> str:
        return f"{self.name}_analysis.parquet"


TABLE_SPECS: Mapping[str, TableSpec] = {
    "gemm": TableSpec(
        name="gemm",
        key_fields=("dtype", "m", "n", "k"),
        # Outer axes are resolved first; ``m`` (tokens) varies most at runtime.
        axes=(AxisSpec("k"), AxisSpec("n"), AxisSpec("m")),
        dtype_fields=("dtype",),
        description="C[m,n] = A[m,k] x B[k,n]; dtype is the weight/storage dtype (e.g. fp16, fp8, int8_wo).",
    ),
    # for prefill
    "context_attention": TableSpec(
        name="context_attention",
        key_fields=(
            "attn_dtype",
            "kv_cache_dtype",
            "batch_size",
            "input_seq_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
        ),
        axes=(AxisSpec("num_heads", "linear"), AxisSpec("batch_size"), AxisSpec("input_seq_len")),
        dtype_fields=("attn_dtype", "kv_cache_dtype"),
        description="Prefill attention core (QK^T, softmax, PV) for batch_size sequences padded to input_seq_len.",
    ),
    # for decode
    "generation_attention": TableSpec(
        name="generation_attention",
        key_fields=(
            "attn_dtype",
            "kv_cache_dtype",
            "batch_size",
            "context_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
        ),
        axes=(AxisSpec("num_heads", "linear"), AxisSpec("batch_size"), AxisSpec("context_len")),
        dtype_fields=("attn_dtype", "kv_cache_dtype"),
        description="Decode attention core for batch_size single-token queries against context_len cached tokens each.",
    ),
    "moe": TableSpec(
        name="moe",
        key_fields=(
            "dtype",
            "distribution",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "top_k",
            "num_experts",
            "tp_size",
            "ep_size",
        ),
        axes=(AxisSpec("num_tokens"),),
        dtype_fields=("dtype",),
        description=(
            "Fused/grouped expert FFN time on one rank. num_tokens is the number of tokens "
            "entering the MoE layer (before top-k expansion) for the whole EP group; the row "
            "gives the time of one rank holding num_experts/ep_size experts sharded tp_size ways."
        ),
    ),
    "elementwise": TableSpec(
        name="elementwise",
        key_fields=("op_name", "dtype", "num_tokens", "hidden_size"),
        axes=(AxisSpec("hidden_size"), AxisSpec("num_tokens")),
        dtype_fields=("dtype",),
        description="Memory-bound point-wise kernels (rmsnorm, rope, residual_add, swiglu, data_reorder).",
    ),
    "collective": TableSpec(
        name="collective",
        key_fields=("dtype", "operation", "group_size", "nodes", "message_bytes"),
        axes=(AxisSpec("message_bytes"),),
        dtype_fields=("dtype",),
        description=(
            "Measured collective time (nccl-tests convention: message_bytes is the full buffer, "
            "time is one out-of-place iteration) for group_size ranks spread over nodes."
        ),
    ),
}

INTEGER_FIELDS = frozenset(
    {
        "m",
        "n",
        "k",
        "batch_size",
        "input_seq_len",
        "context_len",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "window_size",
        "num_tokens",
        "hidden_size",
        "inter_size",
        "top_k",
        "num_experts",
        "tp_size",
        "ep_size",
        "group_size",
        "nodes",
        "message_bytes",
    }
)

STRING_FIELDS = frozenset(
    field
    for spec in TABLE_SPECS.values()
    for field in spec.key_fields
    if field not in INTEGER_FIELDS
) | {"source_id"}


def table_spec(name: str) -> TableSpec:
    try:
        return TABLE_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unknown operator table {name!r}; known: {sorted(TABLE_SPECS)}") from exc
