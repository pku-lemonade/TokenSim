"""Shared global FP8 KV traffic and reconstruction for LP30 attention kernels."""

from dataclasses import replace

from TokenSim.config.kv_storage import rotary_indices, shared_head_bytes
from TokenSim.errors import ConfigurationError


def shared_attention_work(key, work, *, kernel):
    """Return explicit physical traffic and additional scalar/shuffle operations."""
    if key.get("kv_cache_layout", "separate") == "separate":
        return work, None
    if (
        key["kv_cache_layout"] != "value_rope_fp8"
        or key["kv_cache_dtype"] != "fp8"
        or key["attn_dtype"] != "bf16"
        or key.get("window_size", 0)
    ):
        raise ConfigurationError("unsupported shared global KV operator layout")
    dim = int(key["head_dim"])
    rotary = len(rotary_indices(dim))
    queries = int(key["batch_size"]) * int(key.get("input_seq_len", 1))
    appended = queries * int(key["num_kv_heads"])
    vectors = work.kv_bytes / (2 * dim)
    read = vectors - appended
    if read < 0 or not vectors.is_integer():
        raise ConfigurationError("shared KV traffic has invalid token/head geometry")
    payload = vectors * (dim + rotary)
    # One cast and one scale multiply per read payload coordinate; append
    # additionally needs abs/max, scale formation and normalization before cast.
    extra_vector = read * (dim + rotary) + appended * (3 * (dim + rotary) + 8)
    # BF16 decoded tiles and FP32 scales are explicit additional data movement.
    shuffle = read * (2 * (dim + rotary) + 8)
    pairs = work.flops / (4 * dim)
    if kernel == "assemble":
        shuffle += read * dim * 4  # BF16 source reads and assembled-K writes.
    elif kernel == "split_dot":
        extra_vector += pairs  # Sum the rotary and nonrotary score matrices.
        shuffle += pairs * 8  # Read and write FP32 partial scores.
    else:
        raise ConfigurationError("unknown LP30 shared KV kernel")
    extra = dict(
        layout="value_rope_fp8",
        kernel=kernel,
        rotary_coordinates=rotary,
        read_head_vectors=read,
        appended_head_vectors=appended,
        cache_payload_elements=payload,
        scale_bytes_per_head_vector=8,
        extra_vector_operations=extra_vector,
        extra_shuffle_bytes=shuffle,
        query_scale_elements=queries * int(key["num_heads"]) * (dim - rotary),
        split_dot_extra_fill=kernel == "split_dot",
    )
    # Matrix FLOPs are unchanged; cache traffic includes both FP32 scales.
    return (
        replace(
            work,
            kv_bytes=vectors * shared_head_bytes(dim),
            activation_bytes=work.activation_bytes + shuffle,
            details={**work.details, "shared_kv": extra},
        ),
        extra,
    )


def query_scale_seconds(extra, policy, bandwidth, memory_efficiency):
    """Price the dependent BF16 Q*gamma transform before any QK matrix work."""
    if extra is None:
        return 0.0
    elements = extra["query_scale_elements"]
    traffic = elements * 6  # BF16 query, gamma and output.
    return (
        elements / (policy.vector_ops_per_s * policy.vector_efficiency)
        + traffic / (policy.shuffle_bytes_per_s * policy.shuffle_efficiency)
        + traffic / (bandwidth * memory_efficiency)
        + policy.fill_drain_us * 1e-6 * (1 + extra["split_dot_extra_fill"])
    )
