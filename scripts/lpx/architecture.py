"""Compare total storage parameters and activated arithmetic across proxies."""


def describe_model(model):
    moe_layers = len(model.moe_layer_indices())
    dense_layers = model.num_layers - moe_layers
    routed = moe_layers * model.moe.num_experts_per_tok * model.expert_params()
    shared = moe_layers * model.moe.num_shared_experts * model.expert_params()
    dense = dense_layers * model.dense_ffn_params_per_layer()
    attention = model.num_layers * model.attention_params_per_layer()
    router = moe_layers * model.hidden_size * model.moe.num_experts
    head = model.hidden_size * model.vocab_size
    return {
        "model_id": model.model_id,
        "total_parameters": model.total_params(),
        "activated_routed_ffn_parameters_per_token": routed,
        "activated_linear_parameters_per_token": routed
        + shared
        + dense
        + attention
        + router
        + head,
        "linear_flops_per_token": 2
        * (routed + shared + dense + attention + router + head),
        "attention_kv_dependent_flops_included": False,
        "full_attention_layers": sum(
            model.attention_window(i) == 0 for i in range(model.num_layers)
        ),
        "bounded_attention_layers": sum(
            model.attention_window(i) > 0 for i in range(model.num_layers)
        ),
        "expert_storage_dtype": model.operator_dtype("routed_experts", model.dtype),
        "nonexpert_weight_dtype": model.dtype,
        "activation_dtype": model.activation_dtype,
        "kv_cache_dtype": model.kv_cache_dtype,
        "uniform_expert_selection": model.moe.routing.uniform_selection,
        "architecture_assumption": model.sources.get("assumption-scaled-gpt-oss", {}),
    }
