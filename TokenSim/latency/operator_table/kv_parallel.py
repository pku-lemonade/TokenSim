"""KV-parallel attention for complete-model prefill on colocated AF ranks.

Selected layer groups keep ``tp / kvp`` query heads, read only the positions
their KVP rank owns, merge partial outputs after an all-to-all inside the KVP
group and project ``heads / tp`` heads.  Ownership and combine volume follow
docs/lpx-context-parallel.md and the decode graph in
``TokenSim.af.planning.context_parallel``.
"""

from __future__ import annotations

import math

from TokenSim.af.runtime.kv_sequence import layer_kv_parallel_size, visible_tokens
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.latency.base import query_tokens
from TokenSim.latency.projection import Projection

# Partial outputs and LSE values cross the combine all-to-all in FP32.
_COMBINE_ELEMENT_BYTES = 4


def apply_kv_parallel(
    backend, workers, *, kv_parallel_size, kv_parallel_layers, kv_append_block
):
    """Shard selected layer groups of ``backend`` by sequence position."""
    if kv_parallel_size == 1:
        return
    me = backend.rank_info
    group = tuple(
        sorted(
            worker.id
            for worker in workers
            if (worker.dp_rank, worker.pp_rank) == (me.dp_rank, me.pp_rank)
            and worker.tp_rank // kv_parallel_size == me.tp_rank // kv_parallel_size
        )
    )
    for child in backend._layer_backends or (backend,):
        kvp = layer_kv_parallel_size(
            child.model.sliding_window, kv_parallel_size, kv_parallel_layers
        )
        if kvp > 1:
            _shard(child, kvp, me.tp_rank % kvp, kv_append_block, group)


def _shard(child, kvp, rank, block, group):
    model, tp = child.model, child.tp
    attention_tp = tp // kvp
    child.heads_local = math.ceil(model.num_attention_heads / attention_tp)
    child.kv_heads_local = local_kv_heads(model.num_key_value_heads, attention_tp)
    child.q_dim_local = child.heads_local * model.head_dim
    child.kv_dim_local = child.kv_heads_local * model.head_dim
    heads = model.num_attention_heads // tp
    dtype = model.operator_dtype("attention")
    child.native_attention_projections = (
        Projection(
            "qkv",
            "gemm",
            dtype,
            child.q_dim_local + (1 if model.k_eq_v else 2) * child.kv_dim_local,
            model.hidden_size,
        ),
        Projection("output", "gemm", dtype, model.hidden_size, heads * model.head_dim),
    )
    window, owner = model.sliding_window, (kvp, rank, block)
    prefill, decode = child._attention_prefill, child._attention_decode
    step = child.estimate_step_latency

    def local_prefill(batch, q_len, kv_len):
        # Owned earlier positions plus at most this chunk's own positions.
        earlier = visible_tokens(kv_len - q_len, window, *owner)
        return prefill(batch, q_len, min(kv_len, q_len + earlier))

    def local_decode(batch, context, query_width=1):
        return decode(
            batch, max(1, visible_tokens(context, window, *owner)), query_width
        )

    def combined_step(requests):
        latency = step(requests)
        tokens = sum(query_tokens(request) for request in requests)
        if not tokens:
            return latency
        key = {
            "op_name": "weighted_combine",
            "dtype": model.activation_dtype,
            "num_tokens": tokens * heads * kvp,
            "hidden_size": model.head_dim + 1,
        }
        if child.device.analytical.get("lpx_pipeline") is not None:
            key.update(lpx_vector_operation="lse_merge", kv_parallel_size=kvp)
        merge_s = child.query("elementwise", key).latency_us * 1e-6
        child.stats.record_component("elementwise", merge_s * child.layers_local)
        latency += merge_s * child.layers_local
        communicator = child.communicator
        if communicator is None or len(group) < 2:
            return latency
        payload = tokens * heads * kvp * (model.head_dim + 1) * _COMBINE_ELEMENT_BYTES
        estimate = communicator._collective("all_to_all", payload, group)
        combine_s = estimate.latency_s * child.layers_local
        communicator._record(
            "kv_combine",
            None,
            payload * child.layers_local,
            combine_s,
            group,
            estimate,
        )
        child.stats.record_component("comm", combine_s)
        return latency + combine_s

    child._attention_prefill = local_prefill
    child._attention_decode = local_decode
    child.estimate_step_latency = combined_step
