"""Local attention contexts and combine work on KV-parallel attention layers.

A KV-parallel rank attends over the positions it owns and emits partial
outputs plus log-sum-exp (LSE) values; the ``attention_output`` node merges
the received fragments and applies the output projection.  See
docs/lpx-context-parallel.md.
"""

from __future__ import annotations

from TokenSim.af.runtime.kv_sequence import visible_tokens
from TokenSim.errors import ConfigurationError

# Partial outputs and LSE values cross the combine all-to-all in FP32.
COMBINE_ELEMENT_BYTES = 4


def kv_parallel_size(query) -> int:
    return int(query.local_shape.get("kv_parallel_size", 1))


def local_attention_lengths(query, window):
    """Return the KV lengths this KVP rank reads for each request.

    A prefill chunk keeps its query tokens and attends to the owned earlier
    positions plus at most its own ``q`` new positions, which bounds the causal
    part without modeling which new positions the rank owns.
    """
    shape = query.local_shape
    group = (
        shape["kv_parallel_size"],
        shape["kv_parallel_rank"],
        shape["kv_append_block"],
    )
    queries = tuple(query.query_lengths)
    if query.phase == "prefill" or any(q > 1 for q in queries):
        if len(queries) != len(query.context_lengths):
            raise ConfigurationError(
                "KV-parallel prefill needs one query length per KV context"
            )
        lengths = tuple(
            min(context, q + visible_tokens(context - q, window, *group))
            for q, context in zip(queries, query.context_lengths)
        )
    else:
        # A rank that owns no visible key still runs the step, so keep one key.
        lengths = tuple(
            max(1, visible_tokens(context, window, *group))
            for context in query.context_lengths
        )
    return lengths


def combine_payload_bytes(tokens, output_heads, head_dim, kvp) -> int:
    """Per-rank all-to-all buffer: one partial slice for every group member."""
    return tokens * output_heads * kvp * (head_dim + 1) * COMBINE_ELEMENT_BYTES


def attention_output_components(provider, resolver, query, tokens, dtype):
    """Merge ``kvp`` partial fragments per retained head, then project."""
    model, shape = provider.model, query.local_shape
    kvp, heads, head_dim = (
        shape["kv_parallel_size"],
        shape["output_heads"],
        shape["head_dim"],
    )
    weight_dtype = model.operator_dtype("attention", query.weight_dtype or dtype)
    key = {
        "op_name": "weighted_combine",
        "dtype": dtype,
        "num_tokens": tokens * heads * kvp,
        "hidden_size": head_dim + 1,
    }
    if resolver.device.analytical.get("lpx_pipeline") is not None:
        key.update(lpx_vector_operation="lse_merge", kv_parallel_size=kvp)
    return [
        provider._component(resolver, "lse_merge", "elementwise", key),
        provider._gemm(
            resolver,
            "output_projection",
            tokens,
            model.hidden_size,
            heads * head_dim,
            weight_dtype,
        ),
    ]
