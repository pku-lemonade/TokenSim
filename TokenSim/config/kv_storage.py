"""Explicit experimental storage geometry for Gemma's tied global K/V states."""

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError

GLOBAL_KV_LAYOUTS = ("separate", "value_rope_fp8")


def rotary_indices(head_dim):
    """Gemma proportional RoPE rotates a prefix within each half of the head."""
    integer(head_dim, "shared KV head dimension", minimum=8)
    if head_dim % 8:
        raise ConfigurationError("shared KV head dimension must divide into eighths")
    quarter_half = head_dim // 8
    return (*range(quarter_half), *range(head_dim // 2, head_dim // 2 + quarter_half))


def validate_global_kv_layout(model):
    mode = model.global_kv_layout
    if mode not in GLOBAL_KV_LAYOUTS:
        raise ConfigurationError(f"global_kv_layout must be one of {GLOBAL_KV_LAYOUTS}")
    if mode == "separate":
        return
    if (
        model.architecture != "gemma4_text"
        or model.kv_cache_dtype != "fp8"
        or model.activation_dtype != "bf16"
        or not model.qk_norm
        or model.components is not None
        or model.attention_type == "mla"
    ):
        raise ConfigurationError(
            "value_rope_fp8 requires Gemma 4, BF16 activations and FP8 KV"
        )
    selected = [
        i
        for i in range(model.num_layers)
        if not model.attention_window(i) and model.attention_geometry(i).k_eq_v
    ]
    if not selected:
        raise ConfigurationError(
            "shared global KV layout needs tied global projections"
        )
    for layer in selected:
        rotary_indices(model.attention_geometry(layer).head_dim)


def selected_layout(model, layer_id):
    if model.global_kv_layout == "separate":
        return "separate"
    return (
        model.global_kv_layout
        if not model.attention_window(layer_id)
        and model.attention_geometry(layer_id).k_eq_v
        else "separate"
    )


def shared_head_bytes(head_dim):
    # Full FP8 values, the rotated FP8 key coordinates and two FP32 scales.
    return head_dim + len(rotary_indices(head_dim)) + 8


def head_storage_bytes(model, layer_id):
    """Override bytes per owned token and KV head only for the selected layers."""
    if selected_layout(model, layer_id) == "separate":
        return None
    return shared_head_bytes(model.attention_geometry(layer_id).head_dim)


def attention_storage_key(model, layer_id):
    if selected_layout(model, layer_id) == "separate":
        return {}
    return dict(kv_cache_layout="value_rope_fp8")


def extra_workspace(*, head_dim, kv_heads, query_heads, queries, tile_keys, buffers):
    """Preserve baseline scratch and additionally reserve assembled K and Q tiles."""
    rotary_indices(head_dim)
    return (
        buffers * tile_keys * kv_heads * head_dim * 2
        + queries * query_heads * head_dim * 2
        + buffers * tile_keys * kv_heads * 8
        + queries * kv_heads * (head_dim + head_dim // 4) * 4
    )
