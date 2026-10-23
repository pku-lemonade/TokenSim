"""Compile aggregate cache reservations from each layer's exact KV heads."""

from dataclasses import replace

from TokenSim.af.runtime.kv_sequence import layer_kv_parallel_size
from TokenSim.af.runtime.topology import kv_head_partitions


def _layer_partitions(layer, tp, block_size, dtype_size, kvp, append_block):
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
    first = layers[0]
    geometries = {(l.num_kv_heads, l.head_dim) for l in layers}
    kvps = {
        layer.layer_id: layer_kv_parallel_size(
            layer.attention_window, kv_parallel_size, kv_parallel_layers
        )
        for layer in layers
    }
    if len(geometries) == 1 and set(kvps.values()) == {1}:
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
