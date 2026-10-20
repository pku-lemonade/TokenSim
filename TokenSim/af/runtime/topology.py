"""Compile KV partitions, tensor ownership, communication plans and dtypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from TokenSim.af.runtime.records import (
    AFKVHeadPartition,
    AFRankBinding,
    AFReplicaRuntime,
    AFRequestOwner,
    AFResolvedCollectivePlan,
    AFTensorInterval,
    AFTensorOwnership,
    RankKey,
)
from TokenSim.config.af.links import AFEffectiveDTypes
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.cache_config import local_kv_heads, stage_layer_count
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.dtypes import canonicalize_dtype
from TokenSim.errors import ConfigurationError


if TYPE_CHECKING:
    from TokenSim.af.runtime.plan import AFRuntimePlan


@dataclass
class AFCacheConfig:
    hybrid_layout = None

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    block_size: int
    model: str
    num_gpu_blocks: int
    num_cpu_blocks: int
    size_per_token: int
    size_per_token_unsharded: int
    model_param_size: int
    model_param_size_unsharded: int
    num_layers_per_rank: int
    total_num_layers: int
    num_kv_heads: int
    local_kv_heads: int
    head_dim: int
    rank_info: ParallelRankInfo

    @classmethod
    def from_plan(
        cls, plan: AFRuntimePlan, replica: AFReplicaRuntime
    ) -> "AFCacheConfig":
        if not replica.is_attention or replica.leader_rank_key is None:
            raise ConfigurationError(
                f"AF cache config requires an A replica, got {replica.domain!r}"
            )
        bindings = [plan.binding(*rank_key) for rank_key in replica.rank_keys]
        leader = plan.binding(*replica.leader_rank_key)
        partitions = [item.kv_partition for item in bindings]
        if any(partition is None for partition in partitions):
            raise ConfigurationError(
                f"AF A replica {replica.domain}:{replica.dp_rank} lacks KV partitions"
            )
        block_bytes = sum(
            partition.bytes_per_block for partition in partitions if partition
        )
        local_partition = leader.kv_partition
        model_bytes = plan.physical_state(leader.physical_identity).model_bytes
        return cls(
            block_size=plan.block_size,
            model=plan.model,
            num_gpu_blocks=replica.logical_gpu_blocks,
            num_cpu_blocks=replica.logical_cpu_blocks,
            size_per_token=block_bytes // plan.block_size,
            size_per_token_unsharded=block_bytes // plan.block_size,
            model_param_size=model_bytes,
            model_param_size_unsharded=sum(
                state.model_bytes for state in plan.physical_states
            ),
            num_layers_per_rank=stage_layer_count(
                plan.total_layers,
                plan.domain_runtime(replica.domain).pipeline_parallel_size,
                leader.pp_rank,
            ),
            total_num_layers=plan.total_layers,
            num_kv_heads=plan.num_kv_heads,
            local_kv_heads=local_partition.local_head_count,
            head_dim=plan.head_dim,
            rank_info=ParallelRankInfo(
                global_rank=leader.domain_rank,
                rank_in_dp_group=leader.tp_rank,
                tp_rank=leader.tp_rank,
                pp_rank=leader.pp_rank,
                dp_rank=leader.dp_rank,
                kv_cache_group_id=f"{leader.domain}:dp{leader.dp_rank}",
            ),
        )


def kv_head_partitions(
    *,
    num_kv_heads: int,
    tensor_parallel_size: int,
    head_dim: int,
    num_layers: int,
    block_size: int,
    dtype_size_bytes: int,
    mla_cache_width: int = 0,
) -> tuple[AFKVHeadPartition, ...]:
    for name, value in (
        ("num_kv_heads", num_kv_heads),
        ("tensor_parallel_size", tensor_parallel_size),
        ("head_dim", head_dim),
        ("num_layers", num_layers),
        ("block_size", block_size),
        ("dtype_size_bytes", dtype_size_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"AF KV partition {name} must be positive")
    if mla_cache_width:
        if type(mla_cache_width) is not int or mla_cache_width < 1 or num_kv_heads != 1:
            raise ConfigurationError(
                "MLA cache requires positive latent+RoPE width and one shared cache"
            )
        token_bytes = mla_cache_width * dtype_size_bytes * num_layers
        return tuple(
            AFKVHeadPartition(
                tp_rank=rank,
                global_head_ids=(0,),
                replica_factor=tensor_parallel_size,
                bytes_per_token=token_bytes,
                bytes_per_block=token_bytes * block_size,
            )
            for rank in range(tensor_parallel_size)
        )
    try:
        heads_per_rank = local_kv_heads(num_kv_heads, tensor_parallel_size)
    except ConfigurationError as exc:
        message = (
            f"KV heads {num_kv_heads} cannot be evenly sharded across A-domain "
            f"TP={tensor_parallel_size}"
            if num_kv_heads >= tensor_parallel_size
            else f"A-domain TP={tensor_parallel_size} cannot form unambiguous "
            f"replication groups for {num_kv_heads} KV heads"
        )
        raise ConfigurationError(message) from exc
    replica_factor = max(1, tensor_parallel_size // num_kv_heads)
    assignments = tuple(
        tuple(
            range(
                (rank // replica_factor) * heads_per_rank,
                (rank // replica_factor + 1) * heads_per_rank,
            )
        )
        for rank in range(tensor_parallel_size)
    )
    bytes_per_head_token = head_dim * 2 * dtype_size_bytes * num_layers
    return tuple(
        AFKVHeadPartition(
            tp_rank=rank,
            global_head_ids=head_ids,
            replica_factor=replica_factor,
            bytes_per_token=len(head_ids) * bytes_per_head_token,
            bytes_per_block=len(head_ids) * bytes_per_head_token * block_size,
        )
        for rank, head_ids in enumerate(assignments)
    )


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
    a_domain, _ = cluster_config.af_domains
    for instance in sorted(
        a_domain.logical_instances,
        key=lambda item: min(mapping.dp_rank for mapping in item.rank_mapping),
    ):
        leader_mapping = min(
            instance.rank_mapping, key=lambda item: (item.pp_rank, item.tp_rank)
        )
        leader = by_key[(a_domain.name, leader_mapping.domain_rank)]
        owners.append(
            AFRequestOwner(
                domain=a_domain.name,
                instance=instance.name,
                dp_rank=leader.dp_rank,
                leader_rank=leader.domain_rank,
                physical_worker=leader.physical_identity,
            )
        )
    return tuple(owners)


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
                for pp_rank in range(domain.parallel.pipeline_parallel_size):
                    participant_ranks = domain.resolve_collective_participants(
                        collective, dp_rank, pp_rank
                    )
                    participant_bindings = tuple(
                        binding_by_key[(domain.name, rank)]
                        for rank in participant_ranks
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
                            pp_rank=pp_rank,
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
    primary_link_name = cluster_config.af_link.name
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
