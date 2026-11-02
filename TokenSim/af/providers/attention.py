"""Uniform and variable-length attention invocation estimates."""

from collections import Counter
import json

from TokenSim.af.providers.compute import prefill_components
from TokenSim.config.kv_storage import attention_storage_key
from TokenSim.errors import ConfigurationError


def attention_core(
    provider, resolver, query, tokens, heads, kv_heads, dtype, lengths=None
):
    model = provider.model
    common = {
        "attn_dtype": dtype,
        "kv_cache_dtype": query.kv_cache_dtype or model.kv_cache_dtype,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": model.attention_geometry(query.layer_id).head_dim,
        "window_size": model.attention_window(query.layer_id),
    }
    storage = attention_storage_key(model, query.layer_id)
    if storage and (
        resolver.device.device_id != "groq_lp30"
        or "lpx_pipeline" not in resolver.device.analytical
    ):
        raise ConfigurationError(
            "shared global KV requires the LP30 analytical pipeline"
        )
    if storage and resolver.package is not None:
        raise ConfigurationError(
            "shared global KV needs analytical_only operator resolution"
        )
    common.update(storage)
    lengths = tuple(query.context_lengths if lengths is None else lengths)
    lengths = lengths or (1,) * tokens
    if any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("attention contexts must be positive integers")
    if query.phase == "prefill" or any(length > 1 for length in query.query_lengths):
        return prefill_components(provider, resolver, query, tokens, common, lengths)
    if len(lengths) == 1:
        lengths = lengths * tokens
    if len(lengths) != tokens:
        raise ValueError("decode attention requires one context per query token")
    histogram = Counter(lengths)
    if len(histogram) == 1:
        table = "generation_attention"
        key = {**common, "batch_size": tokens, "context_len": lengths[0]}
    else:
        table = "ragged_generation_attention"
        key = {
            **common,
            "batch_size": tokens,
            "context_lengths": json.dumps(
                sorted(histogram.items()), separators=(",", ":")
            ),
        }
    return [provider._component(resolver, "attention_core", table, key)]
