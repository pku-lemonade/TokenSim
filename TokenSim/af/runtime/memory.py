"""Compile component memory, shared physical ledgers and protocol buffers."""

from __future__ import annotations


from TokenSim.af.runtime.records import (
    AFKVHeadPartition,
    AFLayerRuntime,
    AFModelComponent,
    AFPhysicalRuntimeState,
    RankKey,
)
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


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


def _compile_physical_states(
    *,
    preliminary,
    kv_by_rank,
    worker_by_physical,
    hardware,
    usable_memory_fraction,
    cluster_config,
    moe_config,
    hidden,
    effective_dtypes,
    model,
):
    physical_components: dict[str, dict[str, AFModelComponent]] = {}
    physical_bindings: dict[str, list[RankKey]] = {}
    physical_kv_bytes: dict[str, list[int]] = {}
    for item in preliminary:
        domain = item["domain"]
        mapping = item["mapping"]
        rank_key = (domain.name, mapping.domain_rank)
        physical = mapping.physical_worker.identity
        physical_bindings.setdefault(physical, []).append(rank_key)
        ledger = physical_components.setdefault(physical, {})
        for component in item["components"]:
            ledger.setdefault(component.component_id, component)
        partition = kv_by_rank.get(rank_key)
        if partition is not None:
            physical_kv_bytes.setdefault(physical, []).append(partition.bytes_per_block)

    physical_states: list[AFPhysicalRuntimeState] = []
    for physical, rank_keys in sorted(physical_bindings.items()):
        worker = worker_by_physical[physical]
        try:
            hardware_conf = hardware.device(worker.hardware)
            capacity_bytes = int(
                hardware_conf.usable_memory_bytes * usable_memory_fraction
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"AF runtime hardware capacity metadata missing for "
                f"worker={physical!r}, hardware={worker.hardware!r}"
            ) from exc
        components = tuple(
            physical_components[physical][component_id]
            for component_id in sorted(physical_components[physical])
        )
        model_bytes = sum(component.bytes for component in components)
        buffers = ()
        if cluster_config.af.transfer_topology_mode == "physical":
            # Explicit conservative workspace assumption: one input and one
            # output buffer for the largest activation/logit boundary in a
            # configured AF block. Reused across layers, not silently zero.
            routing_expansion = (
                max(1, moe_config.num_experts_per_tok) if moe_config is not None else 1
            )
            workspace = (
                2
                * cluster_config.af.af_block_size
                * max(hidden * routing_expansion, model.vocab_size)
                * dtype_size_bytes(
                    effective_dtypes.activation_dtype, role="activation_dtype"
                )
            )
            buffers += (("communication_workspace", workspace),)
        if (
            cluster_config.af.execution_protocol == "vllm_afd"
            and cluster_config.af.communication_timing.native_control_enabled
        ):
            from TokenSim.af.execution.control import encode_eager_control

            tokens = cluster_config.af.af_block_size
            hidden_bytes = (
                tokens
                * hidden
                * dtype_size_bytes(
                    effective_dtypes.activation_dtype, role="activation_dtype"
                )
            )
            roles = {
                item["domain"].capability
                for item in preliminary
                if item["mapping"].physical_worker.identity == physical
            }
            # Dedicated logical protocol buffers for one eager iteration, reused
            # across layers. Kernel workspaces/allocator fragmentation are separate.
            if "af_attention" in roles:
                buffers += (
                    ("a_residual", hidden_bytes),
                    ("a_send_hidden", hidden_bytes),
                    ("a_return_hidden", hidden_bytes),
                )
            if "af_ffn" in roles:
                buffers += (
                    ("f_input_hidden", hidden_bytes),
                    ("f_output_hidden", hidden_bytes),
                    ("f_router_logits", tokens * moe_config.num_experts * 2),
                    ("f_topk_ids_weights", tokens * moe_config.num_experts_per_tok * 8),
                )
            buffers += (("control_staging", 8 + len(encode_eager_control(tokens))),)
        required_bytes = model_bytes + sum(size for _, size in buffers)
        if required_bytes >= capacity_bytes:
            domains_and_ranks = ",".join(
                f"{domain}:{rank}" for domain, rank in rank_keys
            )
            component_bytes: dict[str, int] = {}
            for component in components:
                component_bytes[component.category] = (
                    component_bytes.get(component.category, 0) + component.bytes
                )
            raise ConfigurationError(
                f"AF model does not fit before execution: domain/rank={domains_and_ranks}, "
                f"physical_worker={physical}, hardware={worker.hardware}, "
                f"component_bytes={component_bytes}, model_bytes={model_bytes}, "
                f"capacity_bytes={capacity_bytes}, required_bytes={required_bytes}, "
                f"available_bytes={capacity_bytes}"
            )
        physical_states.append(
            AFPhysicalRuntimeState(
                physical_identity=physical,
                hardware=worker.hardware,
                capacity_bytes=capacity_bytes,
                bindings=tuple(sorted(rank_keys)),
                components=components,
                model_bytes=model_bytes,
                kv_bytes_per_block=max(physical_kv_bytes.get(physical, [0])),
                protocol_buffers=buffers,
            )
        )
    return tuple(physical_states)
