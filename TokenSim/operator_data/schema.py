from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SCHEMA_VERSION = 1
LATENCY_UNIT = "us"


@dataclass(frozen=True)
class TableSpec:
    name: str
    key_fields: tuple[str, ...]
    interpolation_axis: str | None
    value_fields: tuple[str, ...] = ("latency_us", "source_id")

    @property
    def required_fields(self) -> tuple[str, ...]:
        return self.key_fields + self.value_fields

    @property
    def discrete_fields(self) -> tuple[str, ...]:
        if self.interpolation_axis is None:
            return self.key_fields
        return tuple(
            field for field in self.key_fields if field != self.interpolation_axis
        )


TABLE_SPECS: Mapping[str, TableSpec] = {
    "gemm": TableSpec(
        "gemm",
        ("device", "backend", "gemm_dtype", "m", "n", "k"),
        "m",
    ),
    "context_attention": TableSpec(
        "context_attention",
        (
            "device",
            "backend",
            "attn_dtype",
            "kv_cache_dtype",
            "batch_size",
            "input_seq_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
        ),
        "input_seq_len",
    ),
    "generation_attention": TableSpec(
        "generation_attention",
        (
            "device",
            "backend",
            "attn_dtype",
            "kv_cache_dtype",
            "batch_size",
            "context_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
        ),
        "context_len",
    ),
    "moe": TableSpec(
        "moe",
        (
            "device",
            "backend",
            "moe_dtype",
            "phase",
            "distribution",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "top_k",
            "num_experts",
            "tp_size",
            "ep_size",
        ),
        "num_tokens",
    ),
    "elementwise": TableSpec(
        "elementwise",
        ("device", "backend", "op_name", "dtype", "num_tokens", "hidden_size"),
        None,
    ),
    "communication": TableSpec(
        "communication",
        (
            "source_device",
            "target_device",
            "backend",
            "network",
            "link_group",
            "operation",
            "direction",
            "group_size",
        ),
        None,
        (
            "fixed_latency_us",
            "effective_bandwidth_bytes_per_s",
            "source_id",
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
    }
)

STRING_FIELDS = frozenset(
    field
    for spec in TABLE_SPECS.values()
    for field in spec.key_fields
    if field not in INTEGER_FIELDS
) | {"source_id"}


def perf_filename(table_name: str) -> str:
    return f"{table_name}_perf.parquet"


def analysis_filename(table_name: str) -> str:
    return f"{table_name}_analysis.parquet"
