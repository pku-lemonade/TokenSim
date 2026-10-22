"""Small native model fixtures independent of historical study catalogs."""

from dataclasses import replace


def tiny_moe(hardware):
    model = hardware.model("moe-toy")
    return replace(
        model,
        hidden_size=32,
        intermediate_size=64,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=128,
        dtype="bf16",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        moe=replace(
            model.moe,
            num_experts=8,
            num_moe_layers=4,
            first_k_dense_replace=0,
            moe_intermediate_size=64,
        ),
    )
