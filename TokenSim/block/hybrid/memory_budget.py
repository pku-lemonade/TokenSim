"""Explicit geometry-based memory estimates for native compressed attention configurations.

These are storage-shape estimates, not GPU profiler measurements. Keep every
assumption visible; historical free/KV/non-torch values are comparison only.
"""

from __future__ import annotations

from TokenSim.errors import ConfigurationError


def _generic_memory_budget(
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
    """Estimate runtime envelopes for the generic attention backend."""
    if not 0 < utilization <= 1:
        raise ConfigurationError("gpu_memory_utilization must be in (0, 1]")
    if max_num_batched_tokens <= 0 or max_num_seqs <= 0 or speculative_tokens < 0:
        raise ConfigurationError(
            "compressed attention memory budgets need positive token/sequence limits and nonnegative lookahead"
        )
    spec = model.components
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
        * (
            model.num_layers
            + (spec.num_nextn_predict_layers if speculative_tokens else 0)
        )
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
            "compressed attention weights and explicit runtime estimates exhaust the configured HBM budget"
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


def hybrid_memory_budget(
    model,
    device,
    weight_bytes,
    parallel,
    *,
    utilization,
    max_num_batched_tokens,
    max_num_seqs,
    speculative_tokens=3,
    moe_execution=None,
    attention_backend="module",
):
    """Budget named persistent allocations and a source live-phase upper bound.

    Generic attention retains its previous estimate. Native FlashInfer storage
    is independent of borrowed FlashMLA timing. Unverified generic MoE buffers,
    SM count and graph driver increments stay explicit grade-D estimates.
    """
    from .live_buffers import PHASES, live_phases
    from .persistent_buffers import (
        MIB,
        align,
        mega_moe_storage,
    )
    from TokenSim.config.moe_execution import MoEExecutionConfig

    for value in (max_num_batched_tokens, max_num_seqs, speculative_tokens):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(
                "compressed attention token, sequence and speculative limits must be integers"
            )
    if (
        not 0 < utilization <= 1
        or max_num_batched_tokens <= 0
        or max_num_seqs <= 0
        or speculative_tokens < 0
    ):
        raise ConfigurationError(
            "compressed attention memory budget requires valid utilization and token/sequence limits"
        )
    if attention_backend != "module":
        return _generic_memory_budget(
            model,
            device,
            weight_bytes,
            parallel,
            utilization=utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            speculative_tokens=speculative_tokens,
        )
    if parallel.pipeline_parallel_size != 1:
        raise ConfigurationError(
            "native compressed attention memory geometry currently requires PP1"
        )
    spec, tokens, tp = (
        model.components,
        max_num_batched_tokens,
        parallel.tensor_parallel_size,
    )
    if spec is None or spec.o_groups % tp or model.num_attention_heads % tp:
        raise ConfigurationError(
            "native compressed attention memory geometry requires complete local attention groups"
        )
    if moe_execution is None:
        execution = MoEExecutionConfig(
            backend=(
                "deep_gemm_amxf4_mega_moe"
                if parallel.enable_expert_parallel
                else "generic"
            )
        )
    else:
        execution = MoEExecutionConfig.from_value(moe_execution)
    mtp = bool(speculative_tokens and spec.num_nextn_predict_layers)
    kinds = sorted(
        {layer.attention_kind for layer in model.layer_specs(include_mtp=False)}
    )
    persistent = {
        "target_topk": tokens * spec.index_topk * 4,
        "draft_topk": tokens * spec.index_topk * 4 if mtp else 0,
        "target_mtp_hidden": (
            tokens * spec.hc_mult * model.hidden_size * 2 if mtp else 0
        ),
        "drafter_hidden": tokens * spec.hc_mult * model.hidden_size * 2 if mtp else 0,
        "flashinfer_workspace": 128 * MIB,
        # CSA/HCA share one factory key; SWA uses the other. In the saved
        # Pro config original_position 65536 * factor 16 == maximum_position.
        "compressed_rope_cache": (
            model.max_position_embeddings * spec.qk_rope_head_dim * 4
            if any(kind != "swa" for kind in kinds)
            else 0
        ),
        "swa_rope_cache": (
            model.max_position_embeddings * spec.qk_rope_head_dim * 4
            if mtp or "swa" in kinds
            else 0
        ),
        "first_mhc_broadcast_weight": spec.hc_mult
        * (spec.hc_mult + 2)
        * model.hidden_size
        * 4,
        "runner_and_drafter_input_buffers": (
            tokens * 13 + (max_num_seqs + 1) * 4 + max_num_seqs * 8
        )
        * (2 if mtp else 1),
    }
    indexer_workspace = None
    if "csa" in kinds:
        gather_rows = model.max_position_embeddings * 40 // 4
        values_width = (
            spec.index_head_dim // 2
            if spec.indexer_cache_dtype == "mxfp4"
            else spec.index_head_dim
        )
        scale_width = (
            spec.index_head_dim // 32 if spec.indexer_cache_dtype == "mxfp4" else 4
        )
        # get_simultaneous reserves the gather pair and radix workspace together
        # during profiling, then retains this one lane throughout inference.
        indexer_workspace = {
            "gather_rows": gather_rows,
            "values_bytes": align(gather_rows * values_width, 256),
            "scales_bytes": align(gather_rows * scale_width, 256),
            "radix_bytes": MIB,
            "workspace_lanes": 1,
        }
        persistent["indexer_shared_workspace"] = sum(
            indexer_workspace[name]
            for name in ("values_bytes", "scales_bytes", "radix_bytes")
        )
    value = device.analytical.get("multiprocessor_count")
    if value is None or value < 2 or int(value) != value:
        raise ConfigurationError(
            "persistent-kernel memory estimates require a configured device multiprocessor_count"
        )
    sms = int(value)
    mega = (
        mega_moe_storage(
            model, parallel, tokens, shared=execution.shared_expert_fusion, sms=sms
        )
        if execution.native
        else None
    )
    communication = {}
    if mega:
        persistent["mega_symmetric"] = mega["bytes"]
        communication = dict(mega["components"])
    elif parallel.enable_expert_parallel:
        # The generic fused/collective path needs a separate source audit.
        # Preserve its previous communication envelope until then.
        dispatch = (
            tokens * parallel.expert_parallel_group_size * model.moe.num_experts_per_tok
        )
        persistent["generic_ep_communication_envelope"] = (
            dispatch * (model.hidden_size + model.hidden_size // 128 * 4 + 12)
            + tokens * model.hidden_size * 2
        )
    query_width = 1 + speculative_tokens

    def plans(rows, graph):
        result = [
            live_phases(
                model,
                parallel,
                rows,
                max_num_seqs,
                kind=kind,
                native_moe=execution.native,
                shared_fusion=execution.shared_expert_fusion,
                first_layer=first,
                graph=graph,
                query_width=query_width,
                sms=sms,
            )
            for kind in kinds
            for first in (True, False)
        ]
        if mtp:
            result.append(
                live_phases(
                    model,
                    parallel,
                    rows,
                    max_num_seqs,
                    kind="swa",
                    draft=True,
                    native_moe=execution.native,
                    shared_fusion=execution.shared_expert_fusion,
                    graph=graph,
                    query_width=query_width,
                    sms=sms,
                )
            )
        return result

    eager = plans(tokens, False)
    eager_peak = max(eager, key=lambda row: row["peak_bytes"])
    capture_seqs = min(max_num_seqs, tokens // query_width)
    capture_rows = capture_seqs * query_width
    graph_plans = plans(capture_rows, True) if capture_rows else []
    graph_count = capture_seqs  # target uniform FULL descriptors
    draft_prefill_graphs = capture_seqs if mtp else 0
    # MRV2 reuses the same capture-size list for q=1 draft decode. Descriptors
    # beyond max_num_seqs are discarded rather than expanded into new graphs.
    draft_decode_graphs = (
        min(capture_seqs, max_num_seqs // query_width)
        if mtp and speculative_tokens > 1
        else 0
    )
    graph_count += draft_prefill_graphs + draft_decode_graphs
    graph_pool_peak = max((row["peak_bytes"] for row in graph_plans), default=0)
    graph_output = capture_rows * model.hidden_size * 2
    driver_proxy = max(0, graph_count - 1) * MIB
    graph_bytes = graph_pool_peak + graph_output + driver_proxy
    hardware_budget = int(device.memory_capacity_bytes.value * utilization)
    deductions = {
        "weights": int(weight_bytes),
        "runtime_and_nccl_reserve": int(device.memory_reserved_bytes.value),
        "native_persistent_buffers": sum(persistent.values()),
        "peak_activation_estimate": eager_peak["peak_bytes"],
        "decode_graph_estimate": graph_bytes,
    }
    remaining = hardware_budget - sum(deductions.values())
    if remaining <= 0:
        raise ConfigurationError(
            "compressed attention native storage and runtime estimates exhaust the configured HBM budget"
        )
    peak_index = PHASES.index(eager_peak["peak_phase"])
    return {
        "hardware_capacity_bytes": int(device.memory_capacity_bytes.value),
        "utilization": utilization,
        "hardware_budget_bytes": hardware_budget,
        "deductions": deductions,
        "persistent_components": persistent,
        "activation_components": {
            r["storage"]: r["bytes"]
            for r in eager_peak["storages"]
            if r["begin_phase"] <= peak_index < r["end_phase"]
        },
        "activation_live_plans": eager,
        "communication_components": communication,
        "native_mega_storage": mega,
        "indexer_workspace": indexer_workspace,
        "graph_storage": {
            "capture_rows": capture_rows,
            "target_descriptors": capture_seqs,
            "draft_prefill_descriptors": draft_prefill_graphs,
            "draft_decode_descriptors": draft_decode_graphs,
            "total_descriptors": graph_count,
            "shared_pool_live_peak_bytes": graph_pool_peak,
            "persistent_target_output_bytes": graph_output,
            "driver_increment_proxy_bytes": driver_proxy,
            "bytes": graph_bytes,
            "plans": graph_plans,
            "scope": "one shared pool across layer sites and descriptors; target output has a separate retained owner",
            "driver_grade": "D: target profiler1MiB per-descriptor floor used for unmeasured target/draft driver increments",
        },
        "available_kv_bytes": remaining,
        "estimation_grade": "D",
        "storage_model": "native named persistent allocations plus source live-phase upper bounds",
        "vllm_source_revision": "30f5c01d4ebb5c493ed86195db24d0ec5b3ce3d7",
        "deepgemm_source_revision": "e1f418c2a4f20818221f6b0e578b4c2f634d4c3f",
        "assumptions": [
            f"{sms} SM from the device's analytical configuration; no reference hardware query or timing-version match",
            "BF16 activation and plain-FP8 FlashInfer source allocations; source-config RoPE extent equals model maximum position",
            "parallel query/quant branches conservatively co-live until join; CUDA allocator reserved-memory/fragmentation unmeasured",
            "context indexer loop may retain old logits during RHS; profiling cap512MiB and one shared gather/radix lane follow pinned defaults",
            "native expert intermediates/shared L1 input are symmetric-buffer regions/aliases; standalone BF16 y is a separate transient",
            "graph scratch reused across layer sites; largest target/draft live bound used once, plus separate fixed output and driver-floor proxy",
            "generic fused-MoE activation/EP envelopes remain unverified when selected; no native ring claimed for that path",
            "runtime/NCCL reserve retained in full; no overlap credit or reference-capacity fit",
            "sampler FP32 processing copy is a conservative optional-path bound; cached probabilistic draft logits not assumed for synthetic sampling",
        ],
        "comparison_policy": "report differences and sensitivity; never fit KV capacity or use negative profiler increments",
    }
