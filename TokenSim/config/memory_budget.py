"""Explicit geometry-based memory estimates for native DSV4 configurations.

These are storage-shape estimates, not GPU profiler measurements. Keep every
assumption visible; historical free/KV/non-torch values are comparison only.
"""

from __future__ import annotations

from TokenSim.errors import ConfigurationError


def deepseek_v4_memory_budget(
    model,
    device,
    weight_bytes,
    parallel,
    *,
    utilization,
    max_num_batched_tokens,
    max_num_seqs,
    speculative_tokens=3,
):
    if not 0 < utilization <= 1:
        raise ConfigurationError("gpu_memory_utilization must be in (0, 1]")
    if max_num_batched_tokens <= 0 or max_num_seqs <= 0 or speculative_tokens < 0:
        raise ConfigurationError(
            "DSV4 memory budgets need positive token/sequence limits and nonnegative lookahead"
        )
    spec = model.deepseek_v4
    tokens = max_num_batched_tokens
    query_width = (
        model.num_attention_heads * model.head_dim // parallel.tensor_parallel_size
    )
    # Two BF16 hyper-residuals, attention BF16 query/output and FP8 query,
    # replicated input projection and grouped output staging, shared top-k
    # indices, and BF16 gate/up activations for the selected experts.
    activation = {
        "hyper_residuals": 2 * tokens * spec.hc_mult * model.hidden_size * 2,
        "attention_query_output": tokens * query_width * (2 + 2 + 1),
        "projection_staging": tokens
        * (
            spec.q_lora_rank
            + model.head_dim
            + spec.o_groups * spec.o_lora_rank // parallel.tensor_parallel_size
        )
        * 2,
        "shared_topk_indices": tokens * spec.index_topk * 4,
        "selected_expert_gate_up": tokens
        * model.moe.num_experts_per_tok
        * model.moe_intermediate_size
        * 2
        * 2,
    }
    # Geometry upper envelope for MegaMoE's dispatch buffers. This is separate
    # from the device's generic NCCL/framework reserve and from local FFN
    # activations above. Exact symmetric packing/lifetimes remain uncertain.
    dispatch_tokens = (
        tokens * parallel.expert_parallel_group_size * model.moe.num_experts_per_tok
    )
    communication = {
        "mega_dispatch_fp8_and_scales": dispatch_tokens
        * (model.hidden_size + (model.hidden_size // 128) * 4),
        "mega_route_ids_weights": dispatch_tokens * (8 + 4),
        "local_combine_output": tokens * model.hidden_size * 2,
    }
    capture_tokens = max_num_seqs * (1 + speculative_tokens)
    # Persistent decode graph envelope across layer sites, assuming one live
    # workspace per layer site in the largest capture, with smaller captures
    # sharing the pool. Runtime has not measured these allocations.
    graph = (
        capture_tokens
        * (model.num_layers + spec.num_nextn_predict_layers)
        * (
            query_width * 2
            + spec.hc_mult * model.hidden_size * 2
            + model.moe.num_experts_per_tok * model.moe_intermediate_size * 2
        )
    )
    hardware_budget = int(device.memory_capacity_bytes.value * utilization)
    deductions = {
        "weights": int(weight_bytes),
        "runtime_and_nccl_reserve": int(device.memory_reserved_bytes.value),
        "peak_activation_estimate": sum(activation.values()),
        "mega_communication_estimate": sum(communication.values()),
        "decode_graph_estimate": graph,
    }
    remaining = hardware_budget - sum(deductions.values())
    if remaining <= 0:
        raise ConfigurationError(
            "DSV4 weights and explicit runtime estimates exhaust the configured HBM budget"
        )
    return {
        "hardware_capacity_bytes": int(device.memory_capacity_bytes.value),
        "utilization": utilization,
        "hardware_budget_bytes": hardware_budget,
        "deductions": deductions,
        "activation_components": activation,
        "communication_components": communication,
        "available_kv_bytes": remaining,
        "estimation_grade": "D",
        "assumptions": [
            "activation buffers conservatively sum live shape envelopes; overlap not measured",
            "MegaMoE dispatch buffer envelope uses EP*top_k*local token budget; packing not measured",
            "decode graph uses per-layer persistent workspace envelope; smaller captures share a pool",
            "runtime/NCCL reserve comes from hardware catalog; not historical profiler increments",
        ],
        "comparison_policy": "report differences and sensitivity; never fit KV capacity or use negative profiler increments",
    }
