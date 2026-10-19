"""Uniform and variable-length attention invocation estimates."""

from collections import Counter
import json

from TokenSim.af.providers.compute import prefill_components


def attention_core(provider, resolver, query, tokens, heads, kv_heads, dtype):
    model = provider.model
    common = {
        "attn_dtype": dtype,
        "kv_cache_dtype": query.kv_cache_dtype or model.kv_cache_dtype,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": model.head_dim,
        "window_size": model.attention_window(query.layer_id),
    }
    lengths = tuple(query.context_lengths) or (1,) * tokens
    if any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("attention contexts must be positive integers")
    if query.phase == "prefill" or any(length > 1 for length in query.query_lengths):
        return prefill_components(provider, resolver, query, tokens, common)
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
