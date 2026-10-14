"""Runtime compile."""

from __future__ import annotations

from typing import Any

from TokenSim.af.runtime.memory import (
    _compile_physical_states,
    _components_for_binding,
    _experts_by_ep,
)
from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.runtime.records import (
    AFDomainRuntime,
    AFKVHeadPartition,
    AFLayerRuntime,
    AFModelComponent,
    AFRankBinding,
    AFReplicaRuntime,
    AFTensorOwnership,
    RankKey,
)
from TokenSim.af.runtime.topology import (
    _compile_collective_plans,
    _compile_dtypes,
    _compile_partial_destinations,
    _compile_request_owners,
    _tensor_interval,
    _validate_parallel_shapes,
    kv_head_partitions,
)
from TokenSim.config.cluster_config import ClusterConfig, WorkerConfig
from TokenSim.config.constants import _GB
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


def compile_af_runtime_plan(
    *,
    cluster_config: ClusterConfig,
    worker_configs: list[WorkerConfig],
    model: Any,
    hardware: Any,
    block_size: int,
    dtype: str = "bf16",
    watermark: float = 0.01,
    usable_memory_fraction: float = 1.0,
    kv_cache_capacity_tokens_per_dp_rank: int | None = None,
    kv_cache_capacity_tokens_total: int | None = None,
) -> AFRuntimePlan:
    from TokenSim.config.validation import integer
    from TokenSim.config.validation import number

    number(usable_memory_fraction, "gpu_memory_utilization", positive=True, maximum=1)
    if (
        kv_cache_capacity_tokens_per_dp_rank is None
        and kv_cache_capacity_tokens_total is None
    ):
        kv_cache_capacity_tokens_per_dp_rank = (
            cluster_config.kv_cache_capacity_tokens_per_dp_rank
        )
        kv_cache_capacity_tokens_total = cluster_config.kv_cache_capacity_tokens_total
    if (
        kv_cache_capacity_tokens_per_dp_rank is not None
        and kv_cache_capacity_tokens_total is not None
    ):
        raise ConfigurationError(
            "AF KV total and per-replica limits are mutually exclusive"
        )
    if kv_cache_capacity_tokens_total is not None:
        integer(
            kv_cache_capacity_tokens_total,
            "KV cache total capacity",
            minimum=block_size,
        )
    if kv_cache_capacity_tokens_per_dp_rank is not None:
        integer(
            kv_cache_capacity_tokens_per_dp_rank,
            "KV cache capacity",
            minimum=block_size,
        )
    if not cluster_config.af.enabled:
        raise ConfigurationError("AF runtime plan requires af.enabled=true")
    from TokenSim.hardware.domains import physical_domain_config

    physical_domain_config(cluster_config.af)
    if not 0 <= watermark < 1:
        raise ConfigurationError("AF runtime watermark must be in [0, 1)")
    from TokenSim.config.model_config import ModelSpec

    if not isinstance(model, ModelSpec):
        raise ConfigurationError(
            f"AF runtime plan requires a ModelSpec, got {type(model).__name__}"
        )
    model_id = model.model_id
    moe_config = model.moe
    if cluster_config.af.execution_protocol == "vllm_afd":
        if (
            "qwen3" not in model_id.lower()
            or not moe_config.enabled
            or moe_config.num_shared_experts
            or moe_config.num_moe_layers != model.num_layers
        ):
            raise ConfigurationError(
                "vllm_afd supports all-MoE Qwen3 without shared experts"
            )
        domains = tuple(cluster_config.execution_domains.values())
        if len(domains) != 2 or any(
            domain.parallel.tensor_parallel_size != 1
            or domain.parallel.data_parallel_size != 1
            or domain.parallel.pipeline_parallel_size != 1
            or domain.parallel.expert_parallel_size != 1
            for domain in domains
        ):
            raise ConfigurationError("vllm_afd supports 1A1F with TP=DP=PP=EP=1")
    effective_dtypes, link_dtypes = _compile_dtypes(cluster_config, model, dtype)
    weight_bytes = dtype_size_bytes(effective_dtypes.weight_dtype, role="weight_dtype")
    kv_bytes = dtype_size_bytes(effective_dtypes.kv_cache_dtype, role="kv_cache_dtype")

    hidden = model.hidden_size
    total_layers = model.num_layers
    attention_heads = model.num_attention_heads
    mla = model.mla_dimensions if model.attention_type == "mla" else None
    head_dim = model.head_dim
    kv_heads = 1 if mla else model.num_key_value_heads
    intermediate, intermediate_source = (
        model.intermediate_size,
        "ModelSpec.intermediate_size",
    )
    moe_intermediate = moe_config.moe_intermediate_size or intermediate
    shared_intermediate = moe_intermediate * (
        moe_config.num_shared_experts if moe_config.enabled else 0
    )
    vocab_size = model.vocab_size
    if vocab_size < 1:
        raise ConfigurationError("AF runtime vocabulary size must be positive")

    moe_layer_ids = set(moe_config.moe_layer_indices(total_layers))
    if moe_config.enabled and len(moe_layer_ids) != moe_config.num_moe_layers:
        raise ConfigurationError(
            "AF runtime MoE layer metadata exceeds the model layer range: "
            f"declared={moe_config.num_moe_layers}, resolved={len(moe_layer_ids)}, "
            f"total_layers={total_layers}"
        )
    layers = tuple(
        AFLayerRuntime(
            layer_id=layer_id,
            mla=mla,
            tie_word_embeddings=model.tie_word_embeddings,
            layer_type="moe" if layer_id in moe_layer_ids else "dense",
            hidden_size=hidden,
            num_attention_heads=attention_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            dense_intermediate_size=intermediate,
            routed_intermediate_size=moe_intermediate,
            shared_intermediate_size=(
                shared_intermediate if layer_id in moe_layer_ids else 0
            ),
            num_experts=(moe_config.num_experts if layer_id in moe_layer_ids else 0),
            top_k=(moe_config.num_experts_per_tok if layer_id in moe_layer_ids else 1),
            num_shared_experts=(
                moe_config.num_shared_experts if layer_id in moe_layer_ids else 0
            ),
        )
        for layer_id in range(total_layers)
    )
    _validate_parallel_shapes(
        cluster_config=cluster_config,
        hidden=hidden,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        mla_cache_width=mla.cache_width if mla else 0,
        intermediate=intermediate,
        moe_intermediate=moe_intermediate,
        shared_intermediate=shared_intermediate,
        has_dense=any(layer.layer_type == "dense" for layer in layers),
        has_moe=bool(moe_layer_ids),
    )

    worker_by_physical = {
        worker.physical_worker_ref.identity: worker
        for worker in worker_configs
        if worker.physical_worker_ref is not None
    }
    domain_configs = tuple(
        domain
        for domain in cluster_config.af.execution_domains
        if domain.capability in {"af_attention", "af_ffn"}
    )
    domains = tuple(
        AFDomainRuntime(
            name=domain.name,
            capability=domain.capability,
            tensor_parallel_size=domain.parallel.tensor_parallel_size,
            data_parallel_size=domain.parallel.data_parallel_size,
            expert_parallel_size=(
                domain.expert_parallel.expert_parallel_size
                if domain.capability == "af_ffn"
                else 1
            ),
            expert_placement_strategy=domain.parallel.expert_placement_strategy,
            expert_parallel_mode=domain.expert_parallel.mode,
            expert_tensor_parallel_size=(
                domain.expert_parallel.expert_tensor_parallel_size
            ),
        )
        for domain in domain_configs
    )

    preliminary: list[dict[str, Any]] = []
    kv_by_rank: dict[RankKey, AFKVHeadPartition] = {}
    component_by_id: dict[str, AFModelComponent] = {}
    for domain in domain_configs:
        tp_size = domain.parallel.tensor_parallel_size
        domain_mappings = tuple(
            sorted(
                (
                    mapping
                    for instance in domain.logical_instances
                    for mapping in instance.rank_mapping
                ),
                key=lambda item: item.domain_rank,
            )
        )
        partitions = (
            kv_head_partitions(
                num_kv_heads=kv_heads,
                tensor_parallel_size=tp_size,
                head_dim=head_dim,
                num_layers=total_layers,
                block_size=block_size,
                dtype_size_bytes=kv_bytes,
                mla_cache_width=mla.cache_width if mla else 0,
            )
            if domain.capability == "af_attention"
            else ()
        )
        expert_view = domain.expert_parallel
        experts_by_rank = _experts_by_ep(
            moe_config.num_experts if moe_config.enabled else 0,
            expert_view.expert_parallel_size,
            domain.parallel.expert_placement_strategy,
        )
        for instance in domain.logical_instances:
            mappings = tuple(
                sorted(instance.rank_mapping, key=lambda item: item.tp_rank)
            )
            tp_group = tuple(item.domain_rank for item in mappings)
            for mapping in mappings:
                physical = mapping.physical_worker.identity
                worker = worker_by_physical.get(physical)
                if worker is None:
                    raise ConfigurationError(
                        f"AF runtime domain={domain.name}, rank={mapping.domain_rank} "
                        f"references missing physical worker {physical!r}"
                    )
                if domain.capability == "af_ffn" and expert_view.mode != "expert_tp":
                    expert_ids = tuple(experts_by_rank.get(mapping.ep_rank, ()))
                    ep_group = tuple(item.domain_rank for item in domain_mappings)
                elif domain.capability == "af_ffn":
                    expert_ids = (
                        tuple(range(moe_config.num_experts))
                        if moe_config.enabled
                        else ()
                    )
                    ep_group = tp_group
                else:
                    expert_ids = ()
                    ep_group = ()
                rank_key = (domain.name, mapping.domain_rank)
                if partitions:
                    kv_by_rank[rank_key] = partitions[mapping.tp_rank]
                components = _components_for_binding(
                    capability=domain.capability,
                    router_location=cluster_config.af.router_location,
                    tp_rank=mapping.tp_rank,
                    tp_size=tp_size,
                    dp_rank=mapping.dp_rank,
                    is_leader=(mapping.tp_rank == 0),
                    expert_ids=expert_ids,
                    layers=layers,
                    hidden=hidden,
                    intermediate_source=intermediate_source,
                    vocab_size=vocab_size,
                    weight_dtype_size=weight_bytes,
                    kv_partition=kv_by_rank.get(rank_key),
                    expert_tp_size=expert_view.expert_tensor_parallel_size,
                    expert_tp_rank=(
                        mapping.tp_rank % expert_view.expert_tensor_parallel_size
                    ),
                    ffn_projection_count=model.ffn_projection_count,
                )
                for component in components:
                    previous = component_by_id.get(component.component_id)
                    if previous is not None and previous != component:
                        raise ConfigurationError(
                            f"AF component {component.component_id!r} has inconsistent "
                            f"metadata: {previous} != {component}"
                        )
                    component_by_id[component.component_id] = component
                preliminary.append(
                    {
                        "domain": domain,
                        "instance": instance.name,
                        "mapping": mapping,
                        "worker": worker,
                        "tp_group": tp_group,
                        "ep_group": ep_group,
                        "expert_ids": expert_ids,
                        "components": components,
                    }
                )

    physical_states = _compile_physical_states(
        preliminary=preliminary,
        kv_by_rank=kv_by_rank,
        worker_by_physical=worker_by_physical,
        hardware=hardware,
        usable_memory_fraction=usable_memory_fraction,
        cluster_config=cluster_config,
        moe_config=moe_config,
        hidden=hidden,
        effective_dtypes=effective_dtypes,
        model=model,
    )
    state_by_physical = {state.physical_identity: state for state in physical_states}

    bindings = tuple(
        AFRankBinding(
            domain=item["domain"].name,
            capability=item["domain"].capability,
            instance=item["instance"],
            domain_rank=item["mapping"].domain_rank,
            tp_rank=item["mapping"].tp_rank,
            pp_rank=item["mapping"].pp_rank,
            dp_rank=item["mapping"].dp_rank,
            ep_rank=item["mapping"].ep_rank,
            physical_worker=item["mapping"].physical_worker,
            hardware=item["worker"].hardware,
            tp_group=item["tp_group"],
            ep_group=item["ep_group"],
            is_scheduler_leader=(
                item["domain"].capability == "af_attention"
                and item["mapping"].tp_rank == 0
            ),
            expert_ids=item["expert_ids"],
            component_ids=tuple(
                component.component_id for component in item["components"]
            ),
            kv_partition=kv_by_rank.get(
                (item["domain"].name, item["mapping"].domain_rank)
            ),
        )
        for item in preliminary
    )
    binding_by_key = {binding.rank_key: binding for binding in bindings}

    replicas: list[AFReplicaRuntime] = []
    for domain in domain_configs:
        replica_capacity = kv_cache_capacity_tokens_per_dp_rank
        if (
            domain.capability == "af_attention"
            and kv_cache_capacity_tokens_total is not None
        ):
            replica_capacity = (
                kv_cache_capacity_tokens_total // domain.parallel.data_parallel_size
            )
            if replica_capacity < block_size:
                raise ConfigurationError(
                    "AF total capacity must allow at least one KV block per A replica"
                )
        for instance in domain.logical_instances:
            instance_bindings = tuple(
                sorted(
                    (
                        binding_by_key[(domain.name, mapping.domain_rank)]
                        for mapping in instance.rank_mapping
                    ),
                    key=lambda binding: binding.tp_rank,
                )
            )
            local_gpu: list[tuple[RankKey, int]] = []
            local_cpu: list[tuple[RankKey, int]] = []
            for binding in instance_bindings:
                if binding.kv_partition is None:
                    gpu_blocks = cpu_blocks = 0
                else:
                    state = state_by_physical[binding.physical_identity]
                    raw_gpu_blocks = (
                        state.capacity_bytes
                        - state.model_bytes
                        - state.protocol_buffer_bytes
                    ) // binding.kv_partition.bytes_per_block
                    gpu_blocks = max(
                        0, raw_gpu_blocks - int(watermark * raw_gpu_blocks)
                    )
                    if replica_capacity is not None:
                        gpu_blocks = min(
                            gpu_blocks,
                            replica_capacity // block_size,
                        )
                    cpu_blocks = int(
                        min(
                            32768,
                            32768 * _GB // binding.kv_partition.bytes_per_block,
                        )
                    )
                local_gpu.append((binding.rank_key, gpu_blocks))
                local_cpu.append((binding.rank_key, cpu_blocks))
            leader = next(
                (
                    binding.rank_key
                    for binding in instance_bindings
                    if binding.is_scheduler_leader
                ),
                None,
            )
            replicas.append(
                AFReplicaRuntime(
                    domain=domain.name,
                    capability=domain.capability,
                    instance=instance.name,
                    dp_rank=instance_bindings[0].dp_rank,
                    rank_keys=tuple(binding.rank_key for binding in instance_bindings),
                    leader_rank_key=leader,
                    local_gpu_blocks=tuple(local_gpu),
                    local_cpu_blocks=tuple(local_cpu),
                    logical_gpu_blocks=(
                        min(value for _, value in local_gpu)
                        if domain.capability == "af_attention"
                        else 0
                    ),
                    logical_cpu_blocks=(
                        min(value for _, value in local_cpu)
                        if domain.capability == "af_attention"
                        else 0
                    ),
                )
            )

    tensor_ownerships = tuple(
        AFTensorOwnership(
            domain=binding.domain,
            domain_rank=binding.domain_rank,
            dp_rank=binding.dp_rank,
            tp_rank=binding.tp_rank,
            physical_worker=binding.physical_identity,
            interval=_tensor_interval(
                hidden,
                next(
                    domain.tensor_parallel_size
                    for domain in domains
                    if domain.name == binding.domain
                ),
                binding.tp_rank,
            ),
            layout="sharded",
            canonical_producer=(binding.tp_rank == 0),
            retains_residual=(binding.capability == "af_attention"),
        )
        for binding in bindings
    )
    request_owners = _compile_request_owners(cluster_config, bindings)
    partial_destinations = _compile_partial_destinations(
        cluster_config, tensor_ownerships
    )
    collective_plans = _compile_collective_plans(cluster_config, binding_by_key)

    plan = AFRuntimePlan(
        mla=mla,
        schema_version="2.0",
        model=model_id,
        block_size=block_size,
        effective_dtypes=effective_dtypes,
        total_layers=total_layers,
        hidden_size=hidden,
        num_attention_heads=attention_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        num_experts=(moe_config.num_experts if moe_config.enabled else 0),
        domains=domains,
        rank_bindings=tuple(
            sorted(bindings, key=lambda binding: (binding.domain, binding.domain_rank))
        ),
        replicas=tuple(
            sorted(replicas, key=lambda replica: (replica.domain, replica.dp_rank))
        ),
        physical_states=tuple(physical_states),
        layers=layers,
        request_owners=request_owners,
        tensor_ownerships=tensor_ownerships,
        partial_destinations=partial_destinations,
        collective_plans=collective_plans,
        link_effective_dtypes=link_dtypes,
    )
    plan.validate()
    return plan
