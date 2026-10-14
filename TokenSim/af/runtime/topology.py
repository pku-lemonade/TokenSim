"""Runtime topology."""

from __future__ import annotations

from typing import Mapping

from TokenSim.af.runtime.cache import kv_head_partitions
from TokenSim.af.runtime.records import (
    AFPartialDestination,
    AFRankBinding,
    AFRequestOwner,
    AFResolvedCollectivePlan,
    AFTensorInterval,
    AFTensorOwnership,
    RankKey,
)
from TokenSim.config.af.links import AFEffectiveDTypes
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.dtypes import canonicalize_dtype
from TokenSim.errors import ConfigurationError


def _validate_parallel_shapes(
    *,
    cluster_config: ClusterConfig,
    hidden: int,
    attention_heads: int,
    kv_heads: int,
    head_dim: int,
    mla_cache_width: int,
    intermediate: int,
    moe_intermediate: int,
    shared_intermediate: int,
    has_dense: bool,
    has_moe: bool,
) -> None:
    for domain in cluster_config.af.execution_domains:
        if domain.capability not in {"af_attention", "af_ffn"}:
            continue
        tp_size = domain.parallel.tensor_parallel_size
        if hidden % tp_size:
            raise ConfigurationError(
                f"AF runtime hidden_size={hidden} is not divisible by "
                f"domain={domain.name!r} TP={tp_size}; uneven hidden shards are "
                "not implemented"
            )
        if domain.capability == "af_attention":
            if attention_heads % tp_size:
                raise ConfigurationError(
                    f"AF runtime query heads={attention_heads} are not divisible by "
                    f"A domain={domain.name!r} TP={tp_size}"
                )
            kv_head_partitions(
                num_kv_heads=kv_heads,
                tensor_parallel_size=tp_size,
                head_dim=head_dim,
                mla_cache_width=mla_cache_width,
                num_layers=1,
                block_size=1,
                dtype_size_bytes=1,
            )
            continue
        if has_dense and intermediate % tp_size:
            raise ConfigurationError(
                f"AF runtime Dense FFN intermediate={intermediate} is not divisible "
                f"by F domain={domain.name!r} TP={tp_size}"
            )
        if shared_intermediate and shared_intermediate % tp_size:
            raise ConfigurationError(
                f"AF runtime Shared FFN intermediate={shared_intermediate} is not "
                f"divisible by F domain={domain.name!r} TP={tp_size}"
            )
        if (
            has_moe
            and domain.expert_parallel.expert_tensor_parallel_size > 1
            and moe_intermediate % domain.expert_parallel.expert_tensor_parallel_size
        ):
            raise ConfigurationError(
                f"AF runtime Routed Expert intermediate={moe_intermediate} is not "
                f"divisible by Expert TP="
                f"{domain.expert_parallel.expert_tensor_parallel_size} in "
                f"domain={domain.name!r}"
            )


def _tensor_interval(hidden: int, tp_size: int, tp_rank: int) -> AFTensorInterval:
    if hidden % tp_size:
        raise ConfigurationError(
            f"AF hidden ownership requires hidden={hidden} divisible by TP={tp_size}"
        )
    width = hidden // tp_size
    return AFTensorInterval(tp_rank * width, (tp_rank + 1) * width)


def _compile_request_owners(
    cluster_config: ClusterConfig,
    bindings: tuple[AFRankBinding, ...],
) -> tuple[AFRequestOwner, ...]:
    by_key = {binding.rank_key: binding for binding in bindings}
    owners: list[AFRequestOwner] = []
    for phase, (a_domain, _) in sorted(cluster_config.phase_to_af_domains.items()):
        for instance in sorted(
            a_domain.logical_instances,
            key=lambda item: min(mapping.dp_rank for mapping in item.rank_mapping),
        ):
            leader_mapping = min(instance.rank_mapping, key=lambda item: item.tp_rank)
            leader = by_key[(a_domain.name, leader_mapping.domain_rank)]
            owners.append(
                AFRequestOwner(
                    phase=phase,
                    domain=a_domain.name,
                    instance=instance.name,
                    dp_rank=leader.dp_rank,
                    leader_rank=leader.domain_rank,
                    physical_worker=leader.physical_identity,
                )
            )
    return tuple(owners)


def _compile_partial_destinations(
    cluster_config: ClusterConfig,
    ownerships: tuple[AFTensorOwnership, ...],
) -> tuple[AFPartialDestination, ...]:
    destinations: list[AFPartialDestination] = []
    for phase, (a_domain, f_domain) in sorted(
        cluster_config.phase_to_af_domains.items()
    ):
        a_owners = tuple(item for item in ownerships if item.domain == a_domain.name)
        f_owners = tuple(item for item in ownerships if item.domain == f_domain.name)
        for source in f_owners:
            for target in a_owners:
                interval = source.interval.intersection(target.interval)
                if interval is None:
                    continue
                destinations.append(
                    AFPartialDestination(
                        phase=phase,
                        source_domain=source.domain,
                        source_rank=source.domain_rank,
                        source_dp_rank=source.dp_rank,
                        target_domain=target.domain,
                        target_rank=target.domain_rank,
                        target_dp_rank=target.dp_rank,
                        interval=interval,
                    )
                )
    return tuple(
        sorted(
            destinations,
            key=lambda item: (
                item.phase,
                item.source_domain,
                item.source_rank,
                item.target_domain,
                item.target_rank,
                item.interval.start,
            ),
        )
    )


def _compile_collective_plans(
    cluster_config: ClusterConfig,
    binding_by_key: Mapping[RankKey, AFRankBinding],
) -> tuple[AFResolvedCollectivePlan, ...]:
    resources = cluster_config.af_communication_resources
    plans: list[AFResolvedCollectivePlan] = []
    for domain in cluster_config.af.execution_domains:
        if domain.capability not in {"af_attention", "af_ffn"}:
            continue
        for replica in domain.logical_instances:
            dp_ranks = {mapping.dp_rank for mapping in replica.rank_mapping}
            if len(dp_ranks) != 1:
                raise ConfigurationError(
                    f"AF runtime logical instance {replica.name!r} spans DP replicas"
                )
            dp_rank = next(iter(dp_ranks))
            for collective in domain.communication_plan:
                resource_group = collective.resource_group
                resource_lane = collective.resource_lane
                network = collective.network
                # Older cluster files omitted collective resource metadata.
                # Resolve it from the AF link while retaining strict checks
                # for references that name a non-existent lane.
                if resource_group is None or resource_lane is None or network is None:
                    candidates = [
                        link
                        for link in cluster_config.af.links
                        if (
                            link.attention_domain == domain.name
                            or link.ffn_domain == domain.name
                            or (
                                link.attention_domain is None
                                and link.ffn_domain is None
                                and link.attention_group == domain.worker_group
                                and (
                                    (domain.capability == "af_attention")
                                    or link.ffn_group == domain.worker_group
                                )
                            )
                        )
                        and bool(set(link.phases) & set(domain.phases))
                    ]
                    if candidates:
                        link = candidates[0]
                        resource_group = resource_group or link.resource_group
                        resource_lane = resource_lane or (
                            "shared"
                            if link.duplex == "half"
                            else (
                                "a_to_f"
                                if domain.capability == "af_attention"
                                else "f_to_a"
                            )
                        )
                        network = network or link.network
                if resource_group is None or resource_lane is None:
                    raise ConfigurationError(
                        f"AF collective {collective.operation!r} in domain "
                        f"{domain.name!r} requires resource_group and resource_lane"
                    )
                key = (resource_group, resource_lane)
                resource = resources.get(key)
                if resource is None:
                    raise ConfigurationError(
                        f"AF collective {collective.operation!r} in domain "
                        f"{domain.name!r} references missing communication lane {key!r}"
                    )
                participant_ranks = domain.resolve_collective_participants(
                    collective, dp_rank
                )
                participant_bindings = tuple(
                    binding_by_key[(domain.name, rank)] for rank in participant_ranks
                )
                plans.append(
                    AFResolvedCollectivePlan(
                        domain=domain.name,
                        dp_rank=dp_rank,
                        after_stage=collective.after_stage,
                        operation=collective.operation,
                        participant_ranks=participant_ranks,
                        physical_workers=tuple(
                            binding.physical_identity
                            for binding in participant_bindings
                        ),
                        network=network or resource.network,
                        resource_group=resource_group,
                        resource_lane=resource_lane,
                        payload_bytes=collective.payload_bytes,
                    )
                )
    return tuple(
        sorted(
            plans,
            key=lambda item: (item.domain, item.dp_rank, item.after_stage),
        )
    )


def _compile_dtypes(cluster_config, model, dtype):
    try:
        model_dtype = canonicalize_dtype(dtype, role="activation_dtype")
    except ValueError as exc:
        raise ConfigurationError(f"invalid AF runtime dtype: {exc}") from exc
    weight_dtype = canonicalize_dtype(
        cluster_config.af.weight_dtype or model.dtype, role="weight_dtype"
    )
    activation_dtype = canonicalize_dtype(
        cluster_config.af.activation_dtype or model_dtype, role="activation_dtype"
    )
    kv_dtype = canonicalize_dtype(
        cluster_config.af.kv_cache_dtype or model.kv_cache_dtype, role="kv_cache_dtype"
    )
    link_dtypes = tuple(
        (
            link.name,
            AFEffectiveDTypes(
                weight_dtype=weight_dtype,
                activation_dtype=activation_dtype,
                kv_cache_dtype=kv_dtype,
                a_to_f_dtype=canonicalize_dtype(
                    link.a_to_f_dtype or activation_dtype, role="a_to_f_dtype"
                ),
                f_to_a_dtype=canonicalize_dtype(
                    link.f_to_a_dtype or activation_dtype, role="f_to_a_dtype"
                ),
            ),
        )
        for link in cluster_config.af.links
    )
    if not link_dtypes:
        raise ConfigurationError("AF runtime plan requires at least one AF link")
    decode_link = cluster_config.phase_to_af_link.get("decode")
    primary_link_name = (
        decode_link.name if decode_link is not None else link_dtypes[0][0]
    )
    effective_dtypes = next(
        effective for name, effective in link_dtypes if name == primary_link_name
    )
    if cluster_config.af.execution_protocol == "vllm_afd" and any(
        getattr(effective, role) != "bf16"
        for _, effective in link_dtypes
        for role in (
            "weight_dtype",
            "activation_dtype",
            "kv_cache_dtype",
            "a_to_f_dtype",
            "f_to_a_dtype",
        )
    ):
        raise ConfigurationError(
            "vllm_afd requires BF16 weights, activations, KV and transfers"
        )
    return effective_dtypes, link_dtypes
