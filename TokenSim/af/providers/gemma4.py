"""Gemma text normalization, scaling and optional LP30 softcap accounting."""


def components_for(provider, resolver, query, components, tokens, tp_size, dtype):
    if provider.model.architecture != "gemma4_text":
        return components
    h = provider.model.hidden_size
    if query.stage == "layout" and query.operation == "post_attention_norm":
        components = [
            provider._elementwise(resolver, name, op, tokens, h, dtype)
            for name, op in (
                ("post_attn_norm", "rmsnorm"),
                ("attn_residual", "residual_add"),
                ("pre_ffn_norm", "rmsnorm"),
            )
        ]
    elif query.stage == "residual":
        components.insert(
            0,
            provider._elementwise(
                resolver, "post_ffn_norm", "rmsnorm", tokens, h, dtype
            ),
        )
    if resolver.device.analytical.get("lpx_pipeline") is None:
        return components
    kind, width = None, h
    if query.stage == "residual":
        kind = "layer_scalar"
    elif query.stage == "layout" and query.operation == "embedding_lookup":
        kind = "embedding_scale"
    elif query.stage == "layout" and query.operation == "lm_head":
        kind, width = "logit_softcap", provider.model.vocab_size // tp_size
    if kind:
        components.append(
            provider._component(
                resolver,
                kind,
                "elementwise",
                dict(
                    op_name="other_residual",
                    dtype=dtype,
                    num_tokens=tokens,
                    hidden_size=width,
                    lpx_vector_operation=kind,
                ),
            )
        )
    return components
