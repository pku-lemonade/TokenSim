"""Uniform and variable-length attention invocation estimates."""

from collections import Counter
import json

from dataclasses import replace
from TokenSim.errors import ConfigurationError


def attention_core(provider, resolver, query, tokens, heads, kv_heads, dtype):
    model = provider.model
    common = {
        "attn_dtype": dtype,
        "kv_cache_dtype": query.kv_cache_dtype or model.kv_cache_dtype,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": model.attention_geometry(query.layer_id).head_dim,
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


def prefill_components(provider, resolver, query, tokens, common):
    lengths = tuple(query.context_lengths) or (tokens,)
    queries = tuple(query.query_lengths)
    if not queries:
        if sum(lengths) == tokens:
            queries = lengths
        elif tokens % len(lengths) == 0:
            queries = (tokens // len(lengths),) * len(lengths)
        else:
            raise ValueError("uneven prefill requires per-request query_lengths")
    if (
        len(queries) != len(lengths)
        or sum(queries) != tokens
        or any(
            type(q) is not int or q < 1 or type(k) is not int or k < q
            for q, k in zip(queries, lengths)
        )
    ):
        raise ValueError("prefill query lengths must fit their per-request KV contexts")
    components = []
    for (q, k), count in sorted(Counter(zip(queries, lengths)).items()):
        key = {**common, "batch_size": count, "input_seq_len": q}
        if k > q:
            key["context_len"] = k
        component = provider._component(
            resolver, "attention_core", "context_attention", key
        )
        if k > q and component.match_type != "analytical":
            base = resolver.analytical_reference_us(
                "context_attention",
                {name: value for name, value in key.items() if name != "context_len"},
            )
            full = resolver.analytical_reference_us("context_attention", key)
            if base is None or full is None or base <= 0:
                raise ConfigurationError(
                    "cached prefill requires an analytical growth reference"
                )
            previous = component.match_type
            for counter in (
                resolver.stats.match_type_counts,
                resolver.stats.table_match_counts["context_attention"],
            ):
                counter[previous] -= 1
                if counter[previous] == 0:
                    del counter[previous]
            resolver.stats.record("context_attention", "extrapolated")
            component = replace(
                component,
                latency_us=component.latency_us * max(1.0, full / base),
                match_type="extrapolated",
                source_id=f"prefix_scaled:{component.source_id}",
            )
        components.append(component)
    return components
