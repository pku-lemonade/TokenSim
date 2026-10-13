"""Transient compressed-attention and MoE storage by execution phase."""

from __future__ import annotations

from .persistent_buffers import MIB, align, ceil_div


PHASES = (
    "mtp_input",
    "mhc_attention",
    "input_projections",
    "parallel_query_quant",
    "sparse_indexer_attention",
    "output_projection",
    "mhc_ffn",
    "expert",
    "final_residual",
    "head_norm",
    "logits_sampler",
)


def live_phases(
    model,
    parallel,
    rows,
    max_seqs,
    *,
    kind,
    native_moe=True,
    shared_fusion=True,
    draft=False,
    first_layer=False,
    graph=False,
    query_width=4,
    sms,
):
    """Upper bound from source owners and half-open allocation intervals.

    Parallel projection/quant branches are conservatively co-live until their
    join. Context indexer loop RHS may retain the previous logits allocation.
    Values are logical live storage, not measured CUDA reserved-memory peaks.
    """
    spec, hidden, tp = (
        model.components,
        model.hidden_size,
        parallel.tensor_parallel_size,
    )
    sp = parallel.pipeline_parallel_size == 1 and parallel.sequence_parallel_moe(
        bool(native_moe)
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
            records.append(
                {
                    "storage": name,
                    "bytes": int(size),
                    "begin_phase": begin,
                    "end_phase": end,
                    "owner": owner,
                }
            )

    if draft:
        # MTP normalization precedes SP: shard views retain the full allocation.
        storage(
            "mtp_normalized_hidden",
            full_hc,
            0,
            10,
            "MTP.forward normalized previous_hidden_states",
        )
        storage(
            "mtp_normalized_embedding",
            full_vector,
            0,
            8,
            "MTP.forward normalized inputs_embeds",
        )
        storage(
            "target_hidden_carry",
            full_vector,
            0,
            11,
            "model runner target output during drafting",
        )
        storage(
            "mtp_input_embedding",
            full_vector,
            0,
            10,
            "outer MTP wrapper retains embedding through layer RHS",
        )
        storage("mtp_h_projection", hc, 0, 1, "MTP h_proj before broadcast add")
        storage("mtp_e_projection", vector, 0, 1, "MTP e_proj before broadcast add")
        storage(
            "attention_residual",
            hc,
            0,
            8,
            "MTP projected input aliases first-layer residual",
        )
    elif not first_layer:
        storage(
            "caller_residual",
            hc,
            1,
            8,
            "model caller holds previous residual through layer RHS",
        )
        storage("caller_hidden", vector, 1, 8, "model caller holds previous FFN output")
        storage("attention_residual", hc, 1, 7, "first mHC transition residual")
    else:
        storage(
            "caller_embedding",
            full_vector,
            1,
            8,
            "first-layer embedding, including SP view owner",
        )
        storage("attention_residual", hc, 1, 7, "first-layer broadcast residual")
    storage(
        "attention_input",
        gathered * hidden * 2,
        1 if not sp else 2,
        6,
        "mHC input or full all-gather owner sliced to positions",
    )
    if sp:
        storage(
            "shard_norm_input",
            vector,
            1,
            3,
            "SP mHC normalized shard before gather RHS",
        )
    storage(
        "ffn_residual",
        hc,
        6,
        9,
        "second mHC transition; original caller still holds prior residual",
    )
    storage("ffn_input", vector, 6, 8, "mHC FFN normalized shard")
    storage(
        "mix_coefficients",
        shard * (spec.hc_mult + spec.hc_mult**2) * 4 * 3,
        1,
        9,
        "caller/attention/FFN post and residual mix upper bound",
    )
    # Split-k scratch is bounded by the device-configured SM count. The actual
    # source dispatch is retained below rather than allocating layer copies.
    splits = (
        8
        if shard <= 32
        else max(1, min(sms // ceil_div(shard, 64), spec.hc_mult * hidden // 256))
    )
    scratch = shard * (spec.hc_mult * (spec.hc_mult + 2) + 1) * 4 * splits
    storage("mhc_scratch_pre", scratch, 1, 2, "mHC GEMM mixes/squared sums")
    storage("mhc_scratch_ffn", scratch, 6, 7, "mHC GEMM mixes/squared sums")
    storage(
        "qkv_raw",
        rows * (spec.q_lora_rank + model.head_dim) * 2,
        2,
        6,
        "fused input GEMM output",
    )
    storage(
        "qkv_normalized",
        rows * (spec.q_lora_rank + model.head_dim) * 2,
        2,
        6,
        "fused q/kv RMSNorm outputs",
    )
    storage(
        "projection_fp8_input",
        rows * (hidden + ceil_div(hidden, 128)),
        2,
        3,
        "input FP8 quantization and packed scales",
    )
    if compressed:
        storage(
            "compressor_kv_score",
            rows * (2 if indexer else 1) * model.head_dim * 2 * 4,
            2,
            6,
            "FP32 compressor projection retained by attention.forward",
        )
    if indexer:
        storage(
            "indexer_compressor_kv_score",
            rows * spec.index_head_dim * 4 * 4,
            2,
            6,
            "FP32 indexer compressor projection",
        )
        storage(
            "indexer_input_weights",
            rows * spec.index_n_heads * 2,
            2,
            6,
            "BF16 indexer weights projection",
        )
        storage(
            "indexer_query_bf16",
            rows * spec.index_n_heads * spec.index_head_dim * 2,
            3,
            4,
            "parallel indexer query projection before quantization join",
        )
        entry = (
            spec.index_head_dim // 2 + spec.index_head_dim // 32
            if spec.indexer_cache_dtype == "mxfp4"
            else spec.index_head_dim
        )
        storage(
            "indexer_query_quant",
            rows * spec.index_n_heads * entry,
            3,
            5,
            "indexer quantized values/scales",
        )
        storage(
            "indexer_output_weights",
            rows * spec.index_n_heads * 4,
            3,
            5,
            "FP32 indexer weights",
        )
        logits = (
            rows * ceil_div(model.max_position_embeddings, 4) * 4
            if graph
            else 512 * MIB
        )
        storage(
            "indexer_logits",
            logits,
            4,
            5,
            "decode padded logits or context/profiling cap",
        )
        if not graph:
            storage(
                "previous_indexer_logits",
                logits,
                4,
                5,
                "context chunk-loop RHS retains previous logits",
            )
    storage(
        "query_bf16",
        rows * heads * model.head_dim * 2,
        3,
        4 if model.kv_cache_dtype == "fp8" else 5,
        "query before plain-FP8 insert or in-place BF16 query",
    )
    if model.kv_cache_dtype == "fp8":
        storage(
            "query_fp8",
            rows * heads * model.head_dim,
            3,
            5,
            "plain FP8 query returned to attention",
        )
    # SM100 TP1's fused epilogue outputs FP8. Other TP sizes allocate BF16
    # attention output, followed by distinct inverse-RoPE FP8 quant storage.
    # The fused epilogue exists only for 128 local heads (all heads at TP1).
    fused_output = (
        heads == 128
        and model.kv_cache_dtype == "fp8"
        and model.operator_dtype("attention_gemm") == "fp8_block"
    )
    storage(
        "attention_output",
        rows * heads * model.head_dim * (1 if fused_output else 2),
        2,
        6,
        "caller-owned FP8 epilogue or BF16 sparse-attention output",
    )
    scale_bytes = align(rows, 4) * heads * ceil_div(model.head_dim, 128)
    if fused_output:
        storage(
            "output_fp8_scales", scale_bytes, 2, 6, "fused epilogue packed TMA scales"
        )
    else:
        storage(
            "output_quantized_values",
            rows * heads * model.head_dim,
            5,
            6,
            "inverse-RoPE FP8 staging",
        )
        storage(
            "output_fp8_scales", scale_bytes, 5, 6, "inverse-RoPE packed TMA scales"
        )
    storage(
        "output_grouped_z",
        rows * groups * spec.o_lora_rank * 2,
        5,
        6,
        "grouped wo_a BF16 output",
    )
    storage(
        "output_linear_quant",
        rows * groups * spec.o_lora_rank,
        5,
        6,
        "wo_b FP8 input staging",
    )
    storage(
        "output_projection_result",
        full_vector,
        5,
        7,
        "wo_b output retained during next mHC RHS",
    )
    if sp:
        storage("reduced_attention_shard", vector, 5, 7, "reduce-scatter result")
        if gathered != rows:
            storage(
                "reduce_scatter_padding",
                gathered * hidden * 2,
                5,
                6,
                "padded full input before collective",
            )
    storage(
        "expert_bf16_y",
        vector,
        7,
        8,
        "distinct MegaMoE output outside symmetric buffer",
    )
    storage(
        "expert_topk",
        shard * model.moe.num_experts_per_tok * 12,
        7,
        8,
        "int64 IDs and FP32 weights",
    )
    if not native_moe:
        storage(
            "generic_expert_envelope",
            shard * model.moe.num_experts_per_tok * model.moe_intermediate_size * 4,
            7,
            8,
            "unverified generic fused-MoE activation envelope",
        )
    if not shared_fusion and model.moe.num_shared_experts:
        storage(
            "unfused_shared_activations",
            shard * model.moe_intermediate_size * model.moe.num_shared_experts * 6,
            7,
            8,
            "shared gate/up and activated FFN intermediates",
        )
        storage(
            "unfused_shared_output", vector, 7, 8, "shared output added to routed y"
        )
    storage("final_reconstructed_residual", hc, 8, 9, "mhc_post final reconstruction")
    if sp:
        storage(
            "final_gathered_residual",
            gathered * spec.hc_mult * hidden * 2,
            8,
            10,
            "final SP gather owner, sliced to full positions",
        )
    else:
        # Reconstruction is the same tensor passed to hc_head, not a second copy.
        records[-1]["end_phase"] = 10
    storage("head_output", full_vector, 9, 11, "hc_head BF16 output")
    storage("norm_output", full_vector, 9, 11, "final RMSNorm BF16 output")
    selected = min(rows, max_seqs if draft else max_seqs * query_width)
    storage(
        "selected_hidden", selected * hidden * 2, 10, 11, "logits selected hidden rows"
    )
    storage(
        "local_logits",
        selected * ceil_div(model.vocab_size, tp) * 2,
        10,
        11,
        "BF16 vocabulary projection",
    )
    if tp > 1:
        storage(
            "gathered_logits",
            selected * model.vocab_size * 2,
            10,
            11,
            "TP logits gather",
        )
    storage(
        "processed_logits",
        selected * model.vocab_size * 4,
        10,
        11,
        "optional sampler FP32 copy upper bound",
    )
    storage(
        "gumbel_scratch",
        selected * ceil_div(model.vocab_size, 1024) * 12,
        10,
        11,
        "local argmax and max before sample reduction",
    )
    live = [
        {
            "phase": phase,
            "bytes": sum(
                r["bytes"] for r in records if r["begin_phase"] <= i < r["end_phase"]
            ),
        }
        for i, phase in enumerate(PHASES)
    ]
    peak = max(live, key=lambda row: row["bytes"])
    return {
        "rows": rows,
        "sp_rows": shard,
        "gather_owner_rows": gathered,
        "kind": kind,
        "draft": draft,
        "first_layer": first_layer,
        "graph": graph,
        "peak_bytes": peak["bytes"],
        "peak_phase": peak["phase"],
        "phases": live,
        "storages": records,
        "grade": "D source live-phase upper bound; allocator and parallel-branch simultaneity unmeasured",
    }
