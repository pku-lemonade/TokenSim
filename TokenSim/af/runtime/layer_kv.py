"""Compile aggregate cache reservations from each layer's exact KV heads."""

from dataclasses import replace

from TokenSim.af.runtime.topology import kv_head_partitions


def layer_kv_partitions(layers, tp, block_size, dtype_size):
    first = layers[0]
    geometries = {(l.num_kv_heads, l.head_dim) for l in layers}
    if len(geometries) == 1:
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
            kv_head_partitions(
                num_kv_heads=layer.num_kv_heads,
                tensor_parallel_size=tp,
                head_dim=layer.head_dim,
                num_layers=1,
                block_size=block_size,
                dtype_size_bytes=dtype_size,
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
        )
        for rank in range(tp)
    )
