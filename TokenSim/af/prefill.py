"""Price per-request prefill queries against the context already in KV."""

from collections import Counter
from dataclasses import replace

from TokenSim.errors import ConfigurationError


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
