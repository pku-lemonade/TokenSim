"""Persistent device buffers of fused MoE kernels.

Sizes follow the allocations of vLLM 30f5c01d4ebb and DeepGEMM e1f418c2a4f.
They are allocation-shape estimates; streaming-multiprocessor counts,
allocator rounding and CUDA-graph driver memory remain explicit assumptions.
"""

from __future__ import annotations

from TokenSim.errors import ConfigurationError
from TokenSim.latency.moe_spectrum import BLOCK_MS


MIB = 1024**2
# Row tile heights (BLOCK_M) DeepGEMM MegaMoE may select for a rank's tokens.
BLOCK_M_CANDIDATES = (8, 16, 32, 64, 96, 128, 192, 240)
# MegaMoE pads its per-rank token capacity to a common multiple of the
# candidate BLOCK_M values so every tiling divides the buffer.
TOKEN_ALIGNMENT = 1920
SIGNAL_BYTES = 21140864  # Host sizeof(alignas(128) MegaMoESignals), max ranks 72.


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def align(value, alignment):
    return ceil_div(value, alignment) * alignment


def mega_moe_storage(model, parallel, tokens, *, shared=True, sms):
    """One class-cached symmetric allocation per matching target/draft key.

    Persistent capacity takes the unsharded scheduler budget. Shared L1 X is
    a view of input X; shared scales/L2 and routed rings are distinct regions.
    """
    ep, experts = parallel.expert_parallel_group_size, model.moe.num_experts
    hidden, inter, topk = (
        model.hidden_size,
        model.moe_intermediate_size,
        model.moe.num_experts_per_tok,
    )
    if (
        not parallel.enable_expert_parallel
        or experts % ep
        or not 1 <= ep <= 72
        or experts > 2048
        or experts // ep > 512
        or sms < 2
        or hidden % 256
        or inter % 128
        or tokens <= 0
    ):
        raise ConfigurationError("unsupported native MegaMoE symmetric-buffer geometry")
    capacity = align(tokens, TOKEN_ALIGNMENT)
    if capacity > (1 << 15) * 8:
        raise ConfigurationError("native MegaMoE capacity exceeds shared signal pool")
    local, clusters = experts // ep, sms // 2
    l1, l2 = 2 * inter // 256, hidden // 256
    candidate_rings = []
    for block in BLOCK_M_CANDIDATES:
        pool_blocks = ceil_div(capacity * ep * min(topk, local), block) + local
        first_l2 = ceil_div(ceil_div(clusters, l2) * l1, clusters)
        interleave = ceil_div(l1 + (pool_blocks - 1) * max(0, l1 - l2), clusters) + 1
        warmup = min(max(first_l2, interleave), ceil_div(pool_blocks * l1, clusters))
        live = min(
            pool_blocks,
            ceil_div(min(warmup * clusters, pool_blocks * l1), l1)
            + ceil_div(pool_blocks * max(0, l2 - l1), l2)
            + ceil_div(clusters, min(l1, l2)),
        )
        candidate_rings.append(live * block)
    ring = align(max(candidate_rings), TOKEN_ALIGNMENT)
    if ring > (1 << 20) * min(BLOCK_M_CANDIDATES):
        raise ConfigurationError("native MegaMoE ring exceeds signal pool")
    sf_ring = max(ring // block * align(block, 128) for block in BLOCK_M_CANDIDATES)
    # Each expert may leave a tail shorter than the largest BLOCK_M.
    full_pool = align(
        capacity * ep * min(topk, local) + local * (max(BLOCK_MS) - 1), TOKEN_ALIGNMENT
    )
    shared_count = model.moe.num_shared_experts if shared else 0
    shared_inter = inter * shared_count
    shared_sf = ceil_div(capacity, 8) * 128
    components = {
        "workspace": align(SIGNAL_BYTES + experts * capacity * 4 + full_pool * 12, 16),
        "input_x": capacity * hidden,
        "input_scales": capacity * hidden // 32,
        "topk_ids": capacity * topk * 8,
        "topk_weights": capacity * topk * 4,
        "shared_l1_scales": shared_sf * hidden // 32 if shared_count else 0,
        "shared_l2_x": capacity * shared_inter,
        "shared_l2_scales": shared_sf * shared_inter // 32,
        "ring_l1_x": ring * hidden,
        "ring_l1_scales": sf_ring * hidden // 32,
        "ring_l1_weights": ring * 4,
        "ring_l2_x": ring * inter,
        "ring_l2_scales": sf_ring * inter // 32,
        "combine_output": capacity * (topk + bool(shared_count)) * hidden * 2,
    }
    offset, intervals = 0, []
    for name, size in components.items():
        intervals.append(
            {
                "storage": name,
                "start_byte": offset,
                "end_byte": offset + size,
                "bytes": size,
            }
        )
        offset += size
    return {
        "bytes": offset,
        "components": components,
        "intervals": intervals,
        "aliases": {"shared_l1_x": "input_x"},
        "cache_instances_per_rank": 1,
        "unsharded_scheduler_tokens": tokens,
        "aligned_tokens": capacity,
        "ring_tokens": ring,
        "scale_ring_tokens": sf_ring,
        "sms_assumption": sms,
        "sms_measured_for_reference": False,
        "cache_key_geometry": {
            "ep": ep,
            "experts": experts,
            "capacity": capacity,
            "topk": topk,
            "hidden": hidden,
            "intermediate": inter,
            "shared_experts": shared_count,
        },
        "scope": "pinned native allocation; borrowed timing kernel revision remains unverified",
    }
