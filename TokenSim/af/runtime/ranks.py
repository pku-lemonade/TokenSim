"""Compile physical rank components and per-stage KV partitions."""

from TokenSim.af.runtime.memory import _components_for_binding, _experts_by_ep
from TokenSim.af.runtime.topology import kv_head_partitions
from TokenSim.af.runtime.component_storage import stored_components
from TokenSim.config.cache_config import stage_layer_range
from TokenSim.errors import ConfigurationError


def compile_rank_components(
    *,
    domains,
    workers,
    model,
    layers,
    hidden,
    intermediate_source,
    block_size,
    weight_bytes,
    weight_dtype,
    kv_bytes,
    kv_heads,
    head_dim,
    mla,
    router_location,
):
    preliminary, kv_by_rank, component_by_id = [], {}, {}
    for domain in domains:
        tp, pp = (
            domain.parallel.tensor_parallel_size,
            domain.parallel.pipeline_parallel_size,
        )
        mappings = tuple(
            sorted(
                (
                    mapping
                    for instance in domain.logical_instances
                    for mapping in instance.rank_mapping
                ),
                key=lambda item: item.domain_rank,
            )
        )
        local_layers = {
            stage: tuple(layers[i] for i in stage_layer_range(len(layers), pp, stage))
            for stage in range(pp)
        }
        partitions = (
            {
                stage: kv_head_partitions(
                    num_kv_heads=kv_heads,
                    tensor_parallel_size=tp,
                    head_dim=head_dim,
                    num_layers=len(local_layers[stage]),
                    block_size=block_size,
                    dtype_size_bytes=kv_bytes,
                    mla_cache_width=mla.cache_width if mla else 0,
                )
                for stage in range(pp)
            }
            if domain.capability == "af_attention"
            else {}
        )
        view = domain.expert_parallel
        experts_by_rank = _experts_by_ep(
            model.moe.num_experts if model.moe.enabled else 0,
            view.expert_parallel_size,
            domain.parallel.expert_placement_strategy,
        )
        for instance in domain.logical_instances:
            ordered = tuple(
                sorted(
                    instance.rank_mapping, key=lambda item: (item.pp_rank, item.tp_rank)
                )
            )
            for mapping in ordered:
                worker = workers.get(mapping.physical_worker.identity)
                if worker is None:
                    raise ConfigurationError(
                        f"AF missing physical worker {mapping.physical_worker.identity!r}"
                    )
                tp_group = tuple(
                    item.domain_rank
                    for item in ordered
                    if item.pp_rank == mapping.pp_rank
                )
                if domain.capability == "af_ffn" and view.mode != "expert_tp":
                    expert_ids = tuple(experts_by_rank.get(mapping.ep_rank, ()))
                    ep_group = tuple(
                        item.domain_rank
                        for item in mappings
                        if item.pp_rank == mapping.pp_rank
                    )
                elif domain.capability == "af_ffn":
                    expert_ids = (
                        tuple(range(model.moe.num_experts)) if model.moe.enabled else ()
                    )
                    ep_group = tp_group
                else:
                    expert_ids, ep_group = (), ()
                if not any(
                    layer.layer_type == "moe" for layer in local_layers[mapping.pp_rank]
                ):
                    expert_ids = ()
                key = (domain.name, mapping.domain_rank)
                if partitions:
                    kv_by_rank[key] = partitions[mapping.pp_rank][mapping.tp_rank]
                role = (
                    "both"
                    if pp == 1
                    else (
                        "embedding"
                        if mapping.pp_rank == 0
                        else "lm_head" if mapping.pp_rank == pp - 1 else "none"
                    )
                )
                components = _components_for_binding(
                    capability=domain.capability,
                    router_location=router_location,
                    tp_rank=mapping.tp_rank,
                    tp_size=tp,
                    dp_rank=mapping.dp_rank,
                    is_leader=mapping.tp_rank == 0,
                    expert_ids=expert_ids,
                    layers=local_layers[mapping.pp_rank],
                    hidden=hidden,
                    intermediate_source=intermediate_source,
                    vocab_size=model.vocab_size,
                    weight_dtype_size=weight_bytes,
                    kv_partition=kv_by_rank.get(key),
                    expert_tp_size=view.expert_tensor_parallel_size,
                    expert_tp_rank=mapping.tp_rank % view.expert_tensor_parallel_size,
                    ffn_projection_count=model.ffn_projection_count,
                    embedding_role=role,
                )
                components = stored_components(
                    components,
                    operator_dtypes=model.operator_dtypes,
                    default_dtype=weight_dtype,
                    projections=model.ffn_projection_count,
                    mla=mla,
                    embedding_role=role,
                )
                for component in components:
                    previous = component_by_id.get(component.component_id)
                    if previous is not None and previous != component:
                        raise ConfigurationError(
                            f"AF component {component.component_id!r} has inconsistent metadata"
                        )
                    component_by_id[component.component_id] = component
                preliminary.append(
                    dict(
                        domain=domain,
                        instance=instance.name,
                        mapping=mapping,
                        worker=worker,
                        tp_group=tp_group,
                        ep_group=ep_group,
                        expert_ids=expert_ids,
                        components=components,
                    )
                )
    return preliminary, kv_by_rank


def validate_expert_components(plan):
    for domain in plan.domains:
        if domain.capability != "af_ffn":
            continue
        for dp_rank in range(domain.data_parallel_size):
            for layer in plan.layers:
                if layer.layer_type == "dense":
                    plan.component_execution(
                        domain.name, layer.layer_id, "dense_ffn", dp_rank
                    )
                elif layer.shared_intermediate_size:
                    plan.component_execution(
                        domain.name, layer.layer_id, "shared_expert", dp_rank
                    )
        for stage in range(domain.pipeline_parallel_size):
            moe_layers = [
                i
                for i in stage_layer_range(
                    plan.total_layers, domain.pipeline_parallel_size, stage
                )
                if plan.layer_plan(i).layer_type == "moe"
            ]
            if not moe_layers:
                continue
            dps = (
                range(domain.data_parallel_size)
                if domain.expert_parallel_mode == "expert_tp"
                else (0,)
            )
            for dp in dps:
                for expert in range(plan.num_experts):
                    owners = plan.expert_execution(
                        domain.name, expert, dp, moe_layers[0]
                    )
                    coordinates = {
                        binding.tp_rank % domain.expert_tensor_parallel_size
                        for binding in owners
                    }
                    if coordinates != set(range(domain.expert_tensor_parallel_size)):
                        raise ConfigurationError(
                            "expert execution lacks exact tensor shards"
                        )
