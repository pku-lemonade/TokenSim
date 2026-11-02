"""Compile aggregate cache reservations from each layer's exact KV heads."""

from dataclasses import replace

from TokenSim.af.runtime.kv_sequence import layer_kv_parallel_size
from TokenSim.af.runtime.topology import kv_head_partitions
from TokenSim.config.kv_storage import shared_head_bytes
from TokenSim.errors import ConfigurationError


def _layer_partitions(
    layer, tp, block_size, dtype_size, kvp, append_block, head_bytes=None
):
    """Shard KV heads over ``tp / kvp`` and positions over ``kvp`` ranks."""
    tpa = tp // kvp
    heads = kv_head_partitions(
        num_kv_heads=layer.num_kv_heads,
        tensor_parallel_size=tpa,
        head_dim=layer.head_dim,
        num_layers=1,
        block_size=block_size,
        dtype_size_bytes=dtype_size,
    )
    if head_bytes is not None:
        heads = tuple(
            replace(
                part,
                bytes_per_token=len(part.global_head_ids) * head_bytes,
                bytes_per_block=len(part.global_head_ids) * head_bytes * block_size,
            )
            for part in heads
        )
    if kvp == 1:
        return heads
    # A logical block of block_size positions leaves ceil(block_size / kvp) here.
    owned_per_block = -(-block_size // kvp)
    return tuple(
        replace(
            heads[rank // kvp],
            tp_rank=rank,
            bytes_per_block=heads[rank // kvp].bytes_per_token * owned_per_block,
            kv_parallel_size=kvp,
            kv_parallel_rank=rank % kvp,
            kv_append_block=append_block,
        )
        for rank in range(tp)
    )


def layer_kv_partitions(
    layers,
    tp,
    block_size,
    dtype_size,
    kv_parallel_size=1,
    kv_parallel_layers="full_attention",
    kv_append_block=16,
):
    head_bytes_by_layer = {}
    for layer in layers:
        if layer.kv_cache_layout == "separate":
            continue
        if (
            layer.kv_cache_layout != "value_rope_fp8"
            or dtype_size != 1
            or layer.attention_window
            or not layer.k_eq_v
            or layer.mla
        ):
            raise ConfigurationError("invalid native shared KV storage geometry")
        head_bytes_by_layer[layer.layer_id] = shared_head_bytes(layer.head_dim)
    first = layers[0]
    geometries = {(l.num_kv_heads, l.head_dim) for l in layers}
    kvps = {
        layer.layer_id: layer_kv_parallel_size(
            layer.attention_window, kv_parallel_size, kv_parallel_layers
        )
        for layer in layers
    }
    if not head_bytes_by_layer and len(geometries) == 1 and set(kvps.values()) == {1}:
        return kv_head_partitions(
            num_kv_heads=first.num_kv_heads,
            tensor_parallel_size=tp,
            head_dim=first.head_dim,
            num_layers=len(layers),
            block_size=block_size,
            dtype_size_bytes=dtype_size,
            mla_cache_width=first.mla.cache_width if first.mla else 0,
        )
    partitions = [
        (
            layer.layer_id,
            _layer_partitions(
                layer,
                tp,
                block_size,
                dtype_size,
                kvps[layer.layer_id],
                kv_append_block,
                head_bytes_by_layer.get(layer.layer_id),
            ),
        )
        for layer in layers
    ]
    return tuple(
        replace(
            partitions[0][1][rank],
            bytes_per_token=sum(p[rank].bytes_per_token for _, p in partitions),
            bytes_per_block=sum(p[rank].bytes_per_block for _, p in partitions),
            layer_partitions=tuple((i, p[rank]) for i, p in partitions),
            kv_parallel_size=1,
            kv_parallel_rank=0,
        )
        for rank in range(tp)
    )
