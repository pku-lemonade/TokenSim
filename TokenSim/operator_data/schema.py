from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SCHEMA_VERSION = 2
LATENCY_UNIT = "us"
OPTIONAL_TABLES = frozenset({"generation_attention_batch", "moe_group", "mla_attention", "batched_gemm"})


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
        (
            "device",
            "backend",
            "weight_dtype",
            "activation_dtype",
            "m",
            "n",
            "k",
        ),
        "m",
    ),
    "context_attention": TableSpec(
        "context_attention",
        (
            "device",
            "backend",
            "attention_type",
            "activation_dtype",
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
            "attention_type",
            "activation_dtype",
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
            "operation",
            "weight_dtype",
            "activation_dtype",
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
        (
            "device",
            "backend",
            "op_name",
            "activation_dtype",
            "source_dtype",
            "target_dtype",
            "num_tokens",
            "hidden_size",
        ),
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
            "resource_lane",
            "operation",
            "direction",
            "transfer_dtype",
            "source_endpoint",
            "target_endpoint",
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


# Ragged contexts and per-expert loads are exact execution shapes. Do not
# interpolate their encoded vectors or substitute a sum of singleton timings.
TABLE_SPECS = dict(TABLE_SPECS)
TABLE_SPECS["generation_attention_batch"] = TableSpec(
    "generation_attention_batch",
    tuple("context_lengths" if name == "context_len" else name
          for name in TABLE_SPECS["generation_attention"].key_fields),
    None,
)
TABLE_SPECS["moe_group"] = TableSpec(
    "moe_group",
    TABLE_SPECS["moe"].key_fields + ("expert_token_counts", "execution_boundary"),
    None,
)
TABLE_SPECS["batched_gemm"] = TableSpec(
    "batched_gemm", TABLE_SPECS["gemm"].key_fields + ("groups",), "m",
)
# Separate optional table: old Attention keys and packages remain unchanged.
# Ragged contexts are exact keys; no interpolation across execution modes.
TABLE_SPECS["mla_attention"] = TableSpec(
    "mla_attention",
    ("device", "backend", "activation_dtype", "kv_cache_dtype", "phase",
     "execution_mode", "batch_size", "query_len", "context_lengths", "num_heads",
     "kv_lora_rank", "rope_head_dim", "nope_head_dim", "v_head_dim"),
    None,
)


def encode_positive_vector(values) -> str:
    values = tuple(values)
    if not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError("batch shape must contain positive integers")
    return ",".join(str(value) for value in values)


def decode_positive_vector(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid encoded batch shape") from exc
    if encode_positive_vector(values) != value:
        raise ValueError("batch shape must use canonical comma-separated integers")
    return values


def validate_batch_key(table_name: str, key: Mapping) -> None:
    if table_name in {"mla_attention", "batched_gemm"}:
        for name in TABLE_SPECS[table_name].key_fields:
            if name in INTEGER_FIELDS and (type(key[name]) is not int or key[name] < 1):
                raise ValueError(f"{table_name} {name} must be a positive integer")
    if table_name == "mla_attention":
        contexts = decode_positive_vector(key["context_lengths"])
        if len(contexts) != key["batch_size"]:
            raise ValueError("MLA context_lengths must match batch_size")
        if key["phase"] == "decode":
            if key["execution_mode"] != "absorbed" or key["query_len"] != 1:
                raise ValueError("MLA decode requires absorbed mode and query_len=1")
        elif key["phase"] == "prefill":
            if key["execution_mode"] != "expanded" or any(s != key["query_len"] for s in contexts):
                raise ValueError("MLA prefill requires expanded mode and equal full sequences; cached/chunked prefill unsupported")
        else:
            raise ValueError("MLA phase must be prefill or decode")
        if key["rope_head_dim"] % 2:
            raise ValueError("MLA rope_head_dim must be even")
    if table_name == "generation_attention_batch":
        if len(decode_positive_vector(key["context_lengths"])) != key["batch_size"]:
            raise ValueError("context_lengths must match batch_size")
    if table_name == "moe_group":
        counts = decode_positive_vector(key["expert_token_counts"])
        if len(counts) != key["num_experts"] or sum(counts) != key["num_tokens"]:
            raise ValueError("expert_token_counts must match experts and assignments")
        if key["execution_boundary"] != "unweighted_partials":
            raise ValueError("moe_group requires unweighted_partials; combine is timed separately")
        if key["top_k"] != 1:
            raise ValueError("moe_group num_tokens already counts routed assignments")


DTYPE_FIELD_ROLES: Mapping[str, str] = {
    "weight_dtype": "weight_dtype",
    "activation_dtype": "activation_dtype",
    "kv_cache_dtype": "kv_cache_dtype",
    "source_dtype": "activation_dtype",
    "target_dtype": "activation_dtype",
}


def dtype_role_for_field(field: str, row: Mapping[str, object]) -> str | None:
    role = DTYPE_FIELD_ROLES.get(field)
    if role is not None:
        return role
    if field != "transfer_dtype":
        return None
    direction = str(row.get("direction", "")).strip().lower()
    if direction == "a_to_f":
        return "a_to_f_dtype"
    if direction == "f_to_a":
        return "f_to_a_dtype"
    return "activation_dtype"


INTEGER_FIELDS = frozenset(
    {
        "groups", "query_len", "kv_lora_rank", "rope_head_dim", "nope_head_dim", "v_head_dim",
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
