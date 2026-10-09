"""Pinned compressed attention native storage geometry and conservative Python live phases.

Source: vLLM 30f5c01d4ebb and its DeepGEMM e1f418c2a4f allocation ABI.
These are allocation-shape estimates. SM/allocator/graph driver costs remain
explicit assumptions, independent of serving throughput or reference KV size.
"""

from __future__ import annotations

from TokenSim.errors import ConfigurationError


MIB = 1024**2
SM_COUNT_ASSUMPTION = 148
BLOCK_M_CANDIDATES = (8, 16, 32, 64, 96, 128, 192, 240)
TOKEN_ALIGNMENT = 1920
SIGNAL_BYTES = 21140864  # Host sizeof(alignas(128) MegaMoESignals), max ranks72.


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def align(value, alignment):
    return ceil_div(value, alignment) * alignment


def mega_moe_storage(model, parallel, tokens, *, shared=True, sms=SM_COUNT_ASSUMPTION):
    """One class-cached symmetric allocation per matching target/draft key.

    Persistent capacity takes the unsharded scheduler budget. Shared L1 X is
    a view of input X; shared scales/L2 and routed rings are distinct regions.
    """
    ep, experts = parallel.expert_parallel_group_size, model.moe.num_experts
    hidden, inter, topk = model.hidden_size, model.moe_intermediate_size, model.moe.num_experts_per_tok
    if (not parallel.enable_expert_parallel or experts % ep or not 1 <= ep <= 72
            or experts > 2048 or experts // ep > 512 or sms < 2
            or hidden % 256 or inter % 128 or tokens <= 0):
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
        live = min(pool_blocks,
                   ceil_div(min(warmup * clusters, pool_blocks * l1), l1)
                   + ceil_div(pool_blocks * max(0, l2 - l1), l2)
                   + ceil_div(clusters, min(l1, l2)))
        candidate_rings.append(live * block)
    ring = align(max(candidate_rings), TOKEN_ALIGNMENT)
    if ring > (1 << 20) * min(BLOCK_M_CANDIDATES):
        raise ConfigurationError("native MegaMoE ring exceeds signal pool")
    sf_ring = max(ring // block * align(block, 128) for block in BLOCK_M_CANDIDATES)
    full_pool = align(capacity * ep * min(topk, local) + local * 239, TOKEN_ALIGNMENT)
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
        intervals.append({"storage": name, "start_byte": offset, "end_byte": offset + size, "bytes": size})
        offset += size
    return {
        "bytes": offset, "components": components, "intervals": intervals,
        "aliases": {"shared_l1_x": "input_x"},
        "cache_instances_per_rank": 1,
        "unsharded_scheduler_tokens": tokens, "aligned_tokens": capacity,
        "ring_tokens": ring, "scale_ring_tokens": sf_ring,
        "sms_assumption": sms, "sms_measured_for_reference": False,
        "cache_key_geometry": {"ep": ep, "experts": experts, "capacity": capacity,
                               "topk": topk, "hidden": hidden, "intermediate": inter,
                               "shared_experts": shared_count},
        "scope": "pinned native allocation; borrowed timing kernel revision remains unverified",
    }


PHASES = ("mtp_input", "mhc_attention", "input_projections", "parallel_query_quant",
          "sparse_indexer_attention", "output_projection", "mhc_ffn", "expert",
          "final_residual", "head_norm", "logits_sampler")


def live_phases(model, parallel, rows, max_seqs, *, kind, native_moe=True,
                shared_fusion=True, draft=False, first_layer=False, graph=False, query_width=4):
    """Upper bound from source owners and half-open allocation intervals.

    Parallel projection/quant branches are conservatively co-live until their
    join. Context indexer loop RHS may retain the previous logits allocation.
    Values are logical live storage, not measured CUDA reserved-memory peaks.
    """
    spec, hidden, tp = model.components, model.hidden_size, parallel.tensor_parallel_size
    sp = parallel.pipeline_parallel_size == 1 and parallel.enable_expert_parallel and tp > 1 and (
        native_moe or parallel.data_parallel_size > 1
    )
    shard = ceil_div(rows, tp) if sp else rows
    gathered = shard * tp if sp else rows
    heads = model.num_attention_heads // tp
    groups = spec.o_groups // tp
    hc, vector = shard * spec.hc_mult * hidden * 2, shard * hidden * 2
    full_vector, full_hc = rows * hidden * 2, rows * spec.hc_mult * hidden * 2
    compressed, indexer = kind != "swa", kind == "csa"
    records = []

    def storage(name, size, begin, end, owner):
        if size:
            records.append({"storage": name, "bytes": int(size), "begin_phase": begin,
                            "end_phase": end, "owner": owner})

    if draft:
        # MTP normalization precedes SP: shard views retain the full allocation.
        storage("mtp_normalized_hidden", full_hc, 0, 10, "MTP.forward normalized previous_hidden_states")
        storage("mtp_normalized_embedding", full_vector, 0, 8, "MTP.forward normalized inputs_embeds")
        storage("target_hidden_carry", full_vector, 0, 11, "model runner target output during drafting")
        storage("mtp_input_embedding", full_vector, 0, 10, "outer MTP wrapper retains embedding through layer RHS")
        storage("mtp_h_projection", hc, 0, 1, "MTP h_proj before broadcast add")
        storage("mtp_e_projection", vector, 0, 1, "MTP e_proj before broadcast add")
        storage("attention_residual", hc, 0, 8, "MTP projected input aliases first-layer residual")
    elif not first_layer:
        storage("caller_residual", hc, 1, 8, "model caller holds previous residual through layer RHS")
        storage("caller_hidden", vector, 1, 8, "model caller holds previous FFN output")
        storage("attention_residual", hc, 1, 7, "first mHC transition residual")
    else:
        storage("caller_embedding", full_vector, 1, 8, "first-layer embedding, including SP view owner")
        storage("attention_residual", hc, 1, 7, "first-layer broadcast residual")
    storage("attention_input", gathered * hidden * 2, 1 if not sp else 2, 6,
            "mHC input or full all-gather owner sliced to positions")
    if sp:
        storage("shard_norm_input", vector, 1, 3, "SP mHC normalized shard before gather RHS")
    storage("ffn_residual", hc, 6, 9, "second mHC transition; original caller still holds prior residual")
    storage("ffn_input", vector, 6, 8, "mHC FFN normalized shard")
    storage("mix_coefficients", shard * (spec.hc_mult + spec.hc_mult**2) * 4 * 3, 1, 9,
            "caller/attention/FFN post and residual mix upper bound")
    # Both fused8-way and fallback split-k scratch use <=148 splits. The actual
    # source dispatch is retained below rather than allocating layer copies.
    splits = 8 if shard <= 32 else max(1, min(SM_COUNT_ASSUMPTION // ceil_div(shard, 64),
                                             spec.hc_mult * hidden // 256))
    scratch = shard * (spec.hc_mult * (spec.hc_mult + 2) + 1) * 4 * splits
    storage("mhc_scratch_pre", scratch, 1, 2, "mHC GEMM mixes/squared sums")
    storage("mhc_scratch_ffn", scratch, 6, 7, "mHC GEMM mixes/squared sums")
    storage("qkv_raw", rows * (spec.q_lora_rank + model.head_dim) * 2, 2, 6, "fused input GEMM output")
    storage("qkv_normalized", rows * (spec.q_lora_rank + model.head_dim) * 2, 2, 6, "fused q/kv RMSNorm outputs")
    storage("projection_fp8_input", rows * (hidden + ceil_div(hidden, 128)), 2, 3,
            "input FP8 quantization and packed scales")
    if compressed:
        storage("compressor_kv_score", rows * (2 if indexer else 1) * model.head_dim * 2 * 4,
                2, 6, "FP32 compressor projection retained by attention.forward")
    if indexer:
        storage("indexer_compressor_kv_score", rows * spec.index_head_dim * 4 * 4, 2, 6,
                "FP32 indexer compressor projection")
        storage("indexer_input_weights", rows * spec.index_n_heads * 2, 2, 6, "BF16 indexer weights projection")
        storage("indexer_query_bf16", rows * spec.index_n_heads * spec.index_head_dim * 2, 3, 4,
                "parallel indexer query projection before quantization join")
        entry = (spec.index_head_dim // 2 + spec.index_head_dim // 32
                 if spec.indexer_cache_dtype == "mxfp4" else spec.index_head_dim)
        storage("indexer_query_quant", rows * spec.index_n_heads * entry, 3, 5, "indexer quantized values/scales")
        storage("indexer_output_weights", rows * spec.index_n_heads * 4, 3, 5, "FP32 indexer weights")
        logits = rows * ceil_div(model.max_position_embeddings, 4) * 4 if graph else 512 * MIB
        storage("indexer_logits", logits, 4, 5, "decode padded logits or context/profiling cap")
        if not graph:
            storage("previous_indexer_logits", logits, 4, 5, "context chunk-loop RHS retains previous logits")
    storage("query_bf16", rows * heads * model.head_dim * 2, 3, 4 if model.kv_cache_dtype == "fp8" else 5,
            "query before plain-FP8 insert or in-place BF16 query")
    if model.kv_cache_dtype == "fp8":
        storage("query_fp8", rows * heads * model.head_dim, 3, 5, "plain FP8 query returned to attention")
    # SM100 TP1's fused epilogue outputs FP8. Other TP sizes allocate BF16
    # attention output, followed by distinct inverse-RoPE FP8 quant storage.
    fused_output = heads == 128 and model.kv_cache_dtype == "fp8" and model.operator_dtype("attention_gemm") == "fp8_block"
    storage("attention_output", rows * heads * model.head_dim * (1 if fused_output else 2), 2, 6,
            "caller-owned FP8 epilogue or BF16 sparse-attention output")
    scale_bytes = align(rows, 4) * heads * ceil_div(model.head_dim, 128)
    if fused_output:
        storage("output_fp8_scales", scale_bytes, 2, 6, "fused epilogue packed TMA scales")
    else:
        storage("output_quantized_values", rows * heads * model.head_dim, 5, 6, "inverse-RoPE FP8 staging")
        storage("output_fp8_scales", scale_bytes, 5, 6, "inverse-RoPE packed TMA scales")
    storage("output_grouped_z", rows * groups * spec.o_lora_rank * 2, 5, 6, "grouped wo_a BF16 output")
    storage("output_linear_quant", rows * groups * spec.o_lora_rank, 5, 6, "wo_b FP8 input staging")
    storage("output_projection_result", full_vector, 5, 7, "wo_b output retained during next mHC RHS")
    if sp:
        storage("reduced_attention_shard", vector, 5, 7, "reduce-scatter result")
        if gathered != rows:
            storage("reduce_scatter_padding", gathered * hidden * 2, 5, 6, "padded full input before collective")
    storage("expert_bf16_y", vector, 7, 8, "distinct MegaMoE output outside symmetric buffer")
    storage("expert_topk", shard * model.moe.num_experts_per_tok * 12, 7, 8, "int64 IDs and FP32 weights")
    if not native_moe:
        storage("generic_expert_envelope", shard * model.moe.num_experts_per_tok
                * model.moe_intermediate_size * 4, 7, 8, "unverified generic fused-MoE activation envelope")
    if not shared_fusion and model.moe.num_shared_experts:
        storage("unfused_shared_activations", shard * model.moe_intermediate_size
                * model.moe.num_shared_experts * 6, 7, 8, "shared gate/up and activated FFN intermediates")
        storage("unfused_shared_output", vector, 7, 8, "shared output added to routed y")
    storage("final_reconstructed_residual", hc, 8, 9, "mhc_post final reconstruction")
    if sp:
        storage("final_gathered_residual", gathered * spec.hc_mult * hidden * 2, 8, 10,
                "final SP gather owner, sliced to full positions")
    else:
        # Reconstruction is the same tensor passed to hc_head, not a second copy.
        records[-1]["end_phase"] = 10
    storage("head_output", full_vector, 9, 11, "hc_head BF16 output")
    storage("norm_output", full_vector, 9, 11, "final RMSNorm BF16 output")
    selected = min(rows, max_seqs if draft else max_seqs * query_width)
    storage("selected_hidden", selected * hidden * 2, 10, 11, "logits selected hidden rows")
    storage("local_logits", selected * ceil_div(model.vocab_size, tp) * 2, 10, 11, "BF16 vocabulary projection")
    if tp > 1:
        storage("gathered_logits", selected * model.vocab_size * 2, 10, 11, "TP logits gather")
    storage("processed_logits", selected * model.vocab_size * 4, 10, 11, "optional sampler FP32 copy upper bound")
    storage("gumbel_scratch", selected * ceil_div(model.vocab_size, 1024) * 12, 10, 11,
            "local argmax and max before sample reduction")
    live = [{"phase": phase, "bytes": sum(r["bytes"] for r in records if r["begin_phase"] <= i < r["end_phase"])}
            for i, phase in enumerate(PHASES)]
    peak = max(live, key=lambda row: row["bytes"])
    return {"rows": rows, "sp_rows": shard, "gather_owner_rows": gathered,
            "kind": kind, "draft": draft, "first_layer": first_layer, "graph": graph,
            "peak_bytes": peak["bytes"], "peak_phase": peak["phase"],
            "phases": live, "storages": records,
            "grade": "D source live-phase upper bound; allocator and parallel-branch simultaneity unmeasured"}
