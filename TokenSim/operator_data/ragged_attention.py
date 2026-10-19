"""Work for one attention kernel over an exact sequence-length histogram."""

import json

from TokenSim.operator_data.schema import canonical_histogram
from TokenSim.operator_data.workload import OperatorWork, generation_attention_work


def ragged_generation_work(key):
    histogram = json.loads(
        canonical_histogram(key["context_lengths"], field="context_lengths")
    )
    batch = key["batch_size"]
    if type(batch) is not int or batch < 1:
        raise ValueError("ragged attention batch_size must be a positive integer")
    if (
        any(count != int(count) for _, count in histogram)
        or sum(count for _, count in histogram) != batch
    ):
        raise ValueError(
            "ragged attention histogram must contain exactly batch_size requests"
        )
    parts = [
        generation_attention_work(
            int(count),
            length,
            int(key["num_heads"]),
            int(key["num_kv_heads"]),
            int(key["head_dim"]),
            str(key["attn_dtype"]),
            str(key["kv_cache_dtype"]),
            int(key.get("window_size", 0)),
        )
        for length, count in histogram
    ]
    return OperatorWork(
        flops=sum(part.flops for part in parts),
        activation_bytes=sum(part.activation_bytes for part in parts),
        kv_bytes=sum(part.kv_bytes for part in parts),
        details={**key, "kernel_invocations": 1},
    )
