"""MLA decomposition into existing GEMM/elementwise and dedicated core keys."""

from TokenSim.operator_data.schema import encode_positive_vector


def attention_components(
    provider,
    query,
    model,
    tokens,
    hidden,
    tp,
    weight_dtype,
    activation_dtype,
    kv_cache_dtype,
    include_tail_elementwise,
):
    dims = model.mla_dimensions
    h = model.num_attention_heads // tp
    common = dict(device=query.device_type.strip().lower(), backend=provider.backend)
    results = []

    def gemm(name, m, n, k, groups=None):
        key = dict(
            common,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            m=m,
            n=n,
            k=k,
        )
        table = "gemm"
        if groups is not None:
            table = "batched_gemm"
            key["groups"] = groups
        results.append(provider._component(name, provider.lookup.lookup(table, key)))

    def element(name, op, width):
        results.append(provider._element_component(name, op, query, tokens, width))

    contexts = query.context_lengths
    encode_positive_vector(contexts)
    if query.phase == "decode":
        if len(contexts) != tokens:
            raise ValueError("MLA decode requires one context length per token")
        query_len, mode = 1, "absorbed"
    elif query.phase == "prefill":
        if len(set(contexts)) != 1 or sum(contexts) != tokens:
            raise ValueError(
                "MLA prefill requires equal full sequences; cached/chunked prefill unsupported"
            )
        query_len, mode = contexts[0], "expanded"
    else:
        raise ValueError("MLA phase must be prefill or decode")

    if "rmsnorm" in model.a_elementwise:
        element("a_rmsnorm", "rmsnorm", hidden)
    gemm("mla_q_down", tokens, dims.q_lora_rank, hidden)
    element("mla_q_norm", "rmsnorm", dims.q_lora_rank)
    gemm("mla_q_up", tokens, h * dims.head_dim, dims.q_lora_rank)
    gemm("mla_kv_down", tokens, dims.cache_width, hidden)
    element("mla_kv_norm", "rmsnorm", dims.kv_lora_rank)
    element("mla_rope", "rope", (h + 1) * dims.rope_head_dim)
    if query.phase == "decode":
        gemm("mla_q_absorb", tokens, dims.kv_lora_rank, dims.nope_head_dim, h)
    else:
        gemm(
            "mla_kv_expand",
            tokens,
            h * (dims.nope_head_dim + dims.v_head_dim),
            dims.kv_lora_rank,
        )

    key = dict(
        common,
        activation_dtype=activation_dtype,
        kv_cache_dtype=kv_cache_dtype,
        phase=query.phase,
        execution_mode=mode,
        batch_size=len(contexts),
        query_len=query_len,
        context_lengths=encode_positive_vector(contexts),
        num_heads=h,
        kv_lora_rank=dims.kv_lora_rank,
        rope_head_dim=dims.rope_head_dim,
        nope_head_dim=dims.nope_head_dim,
        v_head_dim=dims.v_head_dim,
    )
    core = provider._component(
        "mla_attention_core", provider.lookup.lookup("mla_attention", key)
    )
    core.update(
        table="mla_attention",
        execution_unit="attention_batch",
        execution_mode=mode,
        cache_layout="latent_plus_rope_replicated_per_A_TP_rank",
    )
    results.append(core)
    if query.phase == "decode":
        gemm("mla_value_expand", tokens, dims.v_head_dim, dims.kv_lora_rank, h)
    gemm("output_projection", tokens, hidden, h * dims.v_head_dim)
    for op in model.a_elementwise:
        if include_tail_elementwise and op in {"residual_add", "data_reorder"}:
            element("a_" + op, op, hidden)
    if (
        model.post_attention_norm_on_a
        and query.stage == "attention"
        and query.layer_type == "moe"
    ):
        element("post_attention_rmsnorm", "rmsnorm", hidden)
    return results
