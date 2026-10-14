"""Runtime components."""

from __future__ import annotations

from typing import Any

from TokenSim.af.runtime_records import (
    AFKVHeadPartition,
    AFLayerRuntime,
    AFModelComponent,
)
from TokenSim.errors import ConfigurationError
from TokenSim.moe.config import MoEModelConfig


def _components_for_binding(
    *,
    capability: str,
    router_location: str,
    tp_rank: int,
    tp_size: int,
    dp_rank: int,
    is_leader: bool,
    expert_ids: tuple[int, ...],
    layers: tuple[AFLayerRuntime, ...],
    hidden: int,
    intermediate_source: str,
    vocab_size: int,
    weight_dtype_size: int,
    kv_partition: AFKVHeadPartition | None,
    expert_tp_size: int,
    expert_tp_rank: int,
) -> tuple[AFModelComponent, ...]:
    _ = dp_rank
    components: list[AFModelComponent] = []
    if is_leader and capability == (
        "af_attention" if router_location == "a" else "af_ffn"
    ):
        for layer in layers:
            if layer.layer_type != "moe":
                continue
            components.append(
                AFModelComponent(
                    component_id=f"layer.{layer.layer_id}.router.leader",
                    category="router",
                    bytes=hidden * layer.num_experts * weight_dtype_size,
                    formula=f"hidden * experts * weight_dtype ({router_location.upper()} leader only)",
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("num_experts", layer.num_experts),
                    ),
                )
            )
    if capability == "af_attention":
        if kv_partition is None:
            raise ConfigurationError("A binding requires a KV-head partition")
        q_heads = layers[0].num_attention_heads // tp_size
        q_width = q_heads * layers[0].head_dim
        kv_width = kv_partition.local_head_count * layers[0].head_dim
        attention_elements = 2 * hidden * q_width + 2 * hidden * kv_width
        mla = layers[0].mla
        if mla:
            attention_elements = mla.weight_elements(q_heads, hidden)
        for layer in layers:
            components.append(
                AFModelComponent(
                    component_id=f"layer.{layer.layer_id}.attention.tp{tp_rank}-of-{tp_size}",
                    category="attention",
                    bytes=attention_elements * weight_dtype_size,
                    formula=(
                        "MLA replicated Q/KV down + head-sharded Q/KV up and O + latent norms"
                        if mla
                        else "H*(Q_local+2*KV_local)+Q_local*H, using exact Q/K/V/O "
                        "dimensions"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("q_heads_local", q_heads),
                        ("kv_heads_local", kv_partition.local_head_count),
                        ("head_dim", layer.head_dim),
                    ),
                )
            )
        components.append(
            AFModelComponent(
                component_id=f"embedding_lm_head.{'tied' if layers[0].tie_word_embeddings else 'untied'}.tp{tp_rank}-of-{tp_size}",
                category="embedding_lm_head",
                bytes=(vocab_size * hidden // tp_size)
                * weight_dtype_size
                * (1 if layers[0].tie_word_embeddings else 2),
                formula="vocab * hidden * weight_dtype / A_TP * (1 if tied else 2)",
                shape=(
                    ("vocab_size", vocab_size),
                    ("hidden_size", hidden),
                    ("tp_size", tp_size),
                ),
            )
        )
        return tuple(components)

    if capability != "af_ffn":
        raise ConfigurationError(f"unsupported AF component capability {capability!r}")
    for layer in layers:
        if layer.layer_type == "dense":
            local_intermediate = layer.dense_intermediate_size // tp_size
            components.append(
                AFModelComponent(
                    component_id=(
                        f"layer.{layer.layer_id}.dense_ffn.tp{tp_rank}-of-{tp_size}"
                    ),
                    category="dense_ffn",
                    bytes=3 * hidden * local_intermediate * weight_dtype_size,
                    formula=(
                        "3 * hidden * local_intermediate * weight_dtype (SwiGLU "
                        f"gate/up/down); intermediate_source={intermediate_source}"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("intermediate_size", layer.dense_intermediate_size),
                        ("local_intermediate_size", local_intermediate),
                    ),
                )
            )
            continue
        local_expert_intermediate = layer.routed_intermediate_size // expert_tp_size
        for expert_id in expert_ids:
            components.append(
                AFModelComponent(
                    component_id=(
                        f"layer.{layer.layer_id}.routed_expert.{expert_id}."
                        f"tp{expert_tp_rank}-of-{expert_tp_size}"
                    ),
                    category="routed_expert",
                    bytes=(3 * hidden * local_expert_intermediate * weight_dtype_size),
                    formula=(
                        "3 * hidden * local_expert_intermediate * weight_dtype "
                        "(SwiGLU gate/up/down)"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("intermediate_size", layer.routed_intermediate_size),
                        ("local_intermediate_size", local_expert_intermediate),
                        ("expert_id", expert_id),
                    ),
                )
            )
        if layer.shared_intermediate_size:
            local_shared_intermediate = layer.shared_intermediate_size // tp_size
            components.append(
                AFModelComponent(
                    component_id=(
                        f"layer.{layer.layer_id}.shared_expert.tp{tp_rank}-of-{tp_size}"
                    ),
                    category="shared_expert",
                    bytes=(3 * hidden * local_shared_intermediate * weight_dtype_size),
                    formula=(
                        "3 * hidden * local_shared_intermediate * weight_dtype "
                        "for fused shared experts"
                    ),
                    layer_id=layer.layer_id,
                    shape=(
                        ("hidden_size", hidden),
                        ("intermediate_size", layer.shared_intermediate_size),
                        ("local_intermediate_size", local_shared_intermediate),
                        ("num_shared_experts", layer.num_shared_experts),
                    ),
                )
            )
    return tuple(components)


def _experts_by_ep(
    num_experts: int, ep_size: int, strategy: str
) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {rank: [] for rank in range(ep_size)}
    if num_experts <= 0:
        return {rank: () for rank in range(ep_size)}
    if strategy == "round_robin":
        for expert_id in range(num_experts):
            result[expert_id % ep_size].append(expert_id)
    elif strategy == "linear":
        base, remainder = divmod(num_experts, ep_size)
        cursor = 0
        for rank in range(ep_size):
            count = base + (1 if rank < remainder else 0)
            result[rank].extend(range(cursor, cursor + count))
            cursor += count
    else:
        raise ConfigurationError(f"unsupported expert placement strategy {strategy!r}")
    return {rank: tuple(experts) for rank, experts in result.items()}


def _positive_model_int(model_conf: Any, name: str) -> int:
    value = getattr(model_conf, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"AF runtime model metadata {name} is missing")
    value = int(value)
    if value < 1:
        raise ConfigurationError(f"AF runtime model metadata {name} must be positive")
    return value


def _num_kv_heads(model_conf: Any) -> int:
    if _as_bool(getattr(model_conf, "Multi_Query", False)):
        return 1
    if _as_bool(getattr(model_conf, "Grouped_Query", False)):
        grouped = int(getattr(model_conf, "Grouped_Num", 0))
        attention_heads = _positive_model_int(model_conf, "Nhead")
        if grouped < 1 or attention_heads % grouped:
            raise ConfigurationError(
                f"AF runtime invalid grouped-query metadata: Nhead={attention_heads}, "
                f"Grouped_Num={grouped}"
            )
        return attention_heads // grouped
    return _positive_model_int(model_conf, "Nhead")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _intermediate_size(
    model_conf: Any, moe_config: MoEModelConfig, hidden: int
) -> tuple[int, str]:
    if moe_config.intermediate_size is not None:
        return int(moe_config.intermediate_size), "moe_config.intermediate_size"
    model_value = getattr(model_conf, "FFN_Hidden", None)
    if model_value is not None:
        value = int(model_value)
        if value < 1:
            raise ConfigurationError("AF runtime FFN_Hidden must be positive")
        return value, "model.FFN_Hidden"
    return 4 * hidden, "analytical_default_4x_hidden"
