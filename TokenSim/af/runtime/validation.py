"""Runtime validation."""

from __future__ import annotations

from TokenSim.af.runtime.topology import kv_head_partitions
from TokenSim.errors import ConfigurationError
from TokenSim.config.cache_config import stage_layer_count, stage_layer_range


class RuntimeValidationMixin:
    def validate(self) -> None:
        if tuple(layer.layer_id for layer in self.layers) != tuple(
            range(self.total_layers)
        ):
            raise ConfigurationError(
                "AF runtime layers must cover every model layer once"
            )
        rank_keys = [binding.rank_key for binding in self.rank_bindings]
        if len(rank_keys) != len(set(rank_keys)):
            raise ConfigurationError("AF runtime plan contains duplicate rank bindings")
        physical_ids = [state.physical_identity for state in self.physical_states]
        if len(physical_ids) != len(set(physical_ids)):
            raise ConfigurationError(
                "AF runtime plan contains duplicate physical states"
            )
        state_by_physical = {
            state.physical_identity: state for state in self.physical_states
        }
        layer_by_id = {layer.layer_id: layer for layer in self.layers}
        for binding in self.rank_bindings:
            state = state_by_physical.get(binding.physical_identity)
            if state is None or binding.rank_key not in state.bindings:
                raise ConfigurationError(
                    f"AF runtime binding domain={binding.domain}, "
                    f"rank={binding.domain_rank} has no matching physical state"
                )
            components = {item.component_id: item for item in state.components}
            if not set(binding.component_ids).issubset(components):
                raise ConfigurationError(
                    f"AF runtime domain={binding.domain}, rank={binding.domain_rank} "
                    "component ownership differs from physical ledger"
                )
            allowed = (
                {"attention", "embedding_lm_head", "router"}
                if binding.capability == "af_attention"
                else {"dense_ffn", "routed_expert", "shared_expert", "router"}
            )
            for component_id in binding.component_ids:
                component = components[component_id]
                if component.category not in allowed:
                    raise ConfigurationError(
                        f"AF runtime rank {binding.rank_key!r} has invalid component "
                        f"category {component.category!r}"
                    )
                if component.layer_id is None:
                    continue
                layer = layer_by_id[component.layer_id]
                if component.category == "router" and layer.layer_type != "moe":
                    raise ConfigurationError(
                        "Dense layer cannot own a Router component"
                    )
                if component.category == "dense_ffn" and layer.layer_type != "dense":
                    raise ConfigurationError(
                        "MoE layer cannot own a Dense FFN component"
                    )
                if component.category in {"routed_expert", "shared_expert"} and (
                    layer.layer_type != "moe"
                ):
                    raise ConfigurationError(
                        "Dense layer cannot own Routed/Shared Expert components"
                    )
        for state in self.physical_states:
            expected_bindings = tuple(
                sorted(
                    binding.rank_key
                    for binding in self.rank_bindings
                    if binding.physical_identity == state.physical_identity
                )
            )
            if tuple(sorted(state.bindings)) != expected_bindings:
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} binding "
                    "mismatch"
                )
            component_ids = [item.component_id for item in state.components]
            if len(component_ids) != len(set(component_ids)):
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} counts a "
                    "component more than once"
                )
            if state.model_bytes != sum(item.bytes for item in state.components):
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} model byte "
                    "total does not match its components"
                )
            if any(
                type(size) is not int or size < 0 for _, size in state.protocol_buffers
            ):
                raise ConfigurationError("invalid native protocol buffer reservation")
            if state.model_bytes + state.protocol_buffer_bytes >= state.capacity_bytes:
                raise ConfigurationError(
                    f"AF runtime physical state {state.physical_identity!r} exceeds "
                    f"capacity: model_bytes={state.model_bytes}, "
                    f"capacity_bytes={state.capacity_bytes}"
                )

        domain_by_name = {domain.name: domain for domain in self.domains}
        for domain in self.domains:
            domain_bindings = tuple(
                item for item in self.rank_bindings if item.domain == domain.name
            )
            stage_count = domain.tensor_parallel_size * domain.data_parallel_size
            expected_count = stage_count * domain.pipeline_parallel_size
            if len(domain_bindings) != expected_count:
                raise ConfigurationError(
                    f"AF runtime domain={domain.name} rank count mismatch: "
                    f"expected base TP*DP={expected_count}, "
                    f"actual={len(domain_bindings)}"
                )
            if domain.capability == "af_ffn" and domain.expert_parallel_mode not in {
                "expert_tp",
                "rank_reuse",
                "expert_sharded",
            }:
                raise ConfigurationError(
                    f"AF runtime domain={domain.name} has unsupported EP mode"
                )
            if domain.capability == "af_attention":
                if any(binding.ep_group for binding in domain_bindings):
                    raise ConfigurationError(
                        f"A domain={domain.name!r} must not have an Expert group"
                    )
                continue
            if domain.expert_parallel_mode == "expert_tp":
                expected_ep_size = 1
                expected_expert_tp = domain.tensor_parallel_size
            else:
                expected_ep_size = domain.expert_parallel_size
                expected_expert_tp = stage_count // expected_ep_size
                if (
                    stage_count % expected_ep_size
                    or domain.tensor_parallel_size % expected_expert_tp
                ):
                    raise ConfigurationError(
                        "expert partition must divide stage world and base TP"
                    )
            if domain.expert_parallel_size != expected_ep_size or (
                domain.expert_tensor_parallel_size != expected_expert_tp
            ):
                raise ConfigurationError(
                    f"F domain={domain.name!r} EP contract mismatch: mode="
                    f"{domain.expert_parallel_mode}, EP={domain.expert_parallel_size}, "
                    f"Expert TP={domain.expert_tensor_parallel_size}; expected "
                    f"EP={expected_ep_size}, Expert TP={expected_expert_tp}"
                )
            world_group = tuple(
                binding.domain_rank
                for binding in sorted(
                    domain_bindings, key=lambda binding: binding.domain_rank
                )
            )
            for binding in domain_bindings:
                expected_ep_group = (
                    binding.tp_group
                    if domain.expert_parallel_mode == "expert_tp"
                    else tuple(
                        rank
                        for rank in world_group
                        if self.binding(domain.name, rank).pp_rank == binding.pp_rank
                    )
                )
                if binding.ep_group != expected_ep_group:
                    raise ConfigurationError(
                        f"F rank {binding.rank_key!r} EP group mismatch: "
                        f"expected={expected_ep_group}, actual={binding.ep_group}"
                    )

        for replica in self.replicas:
            domain = domain_by_name.get(replica.domain)
            if domain is None:
                raise ConfigurationError(
                    f"AF runtime replica references missing domain {replica.domain!r}"
                )
            bindings = tuple(
                sorted(
                    (self.binding(*rank_key) for rank_key in replica.rank_keys),
                    key=lambda item: (item.pp_rank, item.tp_rank),
                )
            )
            if any(item.dp_rank != replica.dp_rank for item in bindings):
                raise ConfigurationError(
                    f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                    "contains a rank from another DP replica"
                )
            if {(item.pp_rank, item.tp_rank) for item in bindings} != {
                (stage, rank)
                for stage in range(domain.pipeline_parallel_size)
                for rank in range(domain.tensor_parallel_size)
            }:
                raise ConfigurationError(
                    f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                    "does not contain exactly one base rank per TP coordinate"
                )
            expected_tp_group = tuple(item.domain_rank for item in bindings)
            for binding in bindings:
                expected_tp_group = tuple(
                    item.domain_rank
                    for item in bindings
                    if item.pp_rank == binding.pp_rank
                )
                expected_ep_rank = (
                    (binding.dp_rank * domain.tensor_parallel_size + binding.tp_rank)
                    // domain.expert_tensor_parallel_size
                    if domain.capability == "af_ffn"
                    and domain.expert_parallel_mode != "expert_tp"
                    else 0
                )
                if binding.ep_rank != expected_ep_rank:
                    raise ConfigurationError(
                        f"AF runtime rank {binding.rank_key!r} EP identity mismatch: "
                        f"expected={expected_ep_rank}, actual={binding.ep_rank}"
                    )
                if binding.tp_group != expected_tp_group:
                    raise ConfigurationError(
                        f"AF runtime rank {binding.rank_key!r} TP group mismatch"
                    )
            if replica.is_attention:
                leaders = [item for item in bindings if item.is_scheduler_leader]
                if len(leaders) != 1 or leaders[0].rank_key != replica.leader_rank_key:
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "must have exactly one scheduler leader"
                    )
                for binding in bindings:
                    expected_partitions = kv_head_partitions(
                        num_kv_heads=self.num_kv_heads,
                        tensor_parallel_size=domain.tensor_parallel_size,
                        head_dim=self.head_dim,
                        num_layers=stage_layer_count(
                            self.total_layers,
                            domain.pipeline_parallel_size,
                            binding.pp_rank,
                        ),
                        block_size=self.block_size,
                        dtype_size_bytes=self.kv_dtype_size_bytes,
                        mla_cache_width=self.mla.cache_width if self.mla else 0,
                    )
                    if binding.kv_partition != expected_partitions[binding.tp_rank]:
                        raise ConfigurationError(
                            f"AF runtime rank {binding.rank_key!r} KV partition mismatch"
                        )
                gpu_capacities = replica.gpu_capacity_by_rank()
                cpu_capacities = replica.cpu_capacity_by_rank()
                if replica.logical_gpu_blocks != min(gpu_capacities.values()):
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "logical GPU capacity is not the smallest rank capacity"
                    )
                if replica.logical_cpu_blocks != min(cpu_capacities.values()):
                    raise ConfigurationError(
                        f"AF runtime domain={replica.domain}, replica={replica.dp_rank} "
                        "logical CPU capacity is not the smallest rank capacity"
                    )
            else:
                if replica.leader_rank_key is not None:
                    raise ConfigurationError(
                        "F replica must not own a scheduler leader"
                    )
                if any(binding.kv_partition is not None for binding in bindings):
                    raise ConfigurationError("F ranks must not own KV state")
        from TokenSim.af.runtime.ranks import validate_expert_components

        validate_expert_components(self)

        for domain in self.domains:
            for dp_rank in range(domain.data_parallel_size):
                for pp_rank in range(domain.pipeline_parallel_size):
                    ownerships = self.ownerships_for_replica(
                        domain.name, dp_rank, pp_rank
                    )
                    cursor = 0
                    for ownership in ownerships:
                        binding = self.binding(ownership.domain, ownership.domain_rank)
                        if ownership.physical_worker != binding.physical_identity:
                            raise ConfigurationError(
                                f"AF tensor ownership for rank {binding.rank_key!r} "
                                "references a different physical worker"
                            )
                        expected_residual = binding.capability == "af_attention"
                        if ownership.retains_residual != expected_residual:
                            raise ConfigurationError(
                                f"AF tensor ownership for rank {binding.rank_key!r} "
                                f"retains_residual={ownership.retains_residual}; "
                                f"expected={expected_residual}"
                            )
                        if ownership.interval.start != cursor:
                            raise ConfigurationError(
                                f"AF tensor ownership for domain={domain.name}, "
                                f"dp_rank={dp_rank} has a gap or overlap"
                            )
                        cursor = ownership.interval.end
                    if cursor != self.hidden_size:
                        raise ConfigurationError(
                            f"AF tensor ownership for domain={domain.name}, "
                            f"dp_rank={dp_rank} does not cover hidden_size={self.hidden_size}"
                        )

        ownership_by_rank = {
            (item.domain, item.domain_rank): item for item in self.tensor_ownerships
        }
        destination_keys = [
            (
                item.phase,
                item.source_domain,
                item.source_rank,
                item.target_domain,
                item.target_rank,
                item.interval.start,
                item.interval.end,
                item.source_layout,
                item.target_layout,
            )
            for item in self.partial_destinations
        ]
        if len(destination_keys) != len(set(destination_keys)):
            raise ConfigurationError("AF runtime has duplicate partial destinations")
        for destination in self.partial_destinations:
            source = ownership_by_rank.get(
                (destination.source_domain, destination.source_rank)
            )
            target = ownership_by_rank.get(
                (destination.target_domain, destination.target_rank)
            )
            if source is None or target is None:
                raise ConfigurationError(
                    "AF partial destination references a missing tensor owner"
                )
            if source.dp_rank != destination.source_dp_rank or (
                target.dp_rank != destination.target_dp_rank
            ):
                raise ConfigurationError(
                    "AF partial destination DP owner does not match its rank"
                )
            expected_interval = source.interval.intersection(target.interval)
            if expected_interval != destination.interval:
                raise ConfigurationError(
                    f"AF partial destination {destination.source_domain}:"
                    f"{destination.source_rank} -> {destination.target_domain}:"
                    f"{destination.target_rank} has interval "
                    f"{destination.interval}; expected {expected_interval}"
                )

        collective_keys = [
            (item.domain, item.dp_rank, item.pp_rank, item.after_stage)
            for item in self.collective_plans
        ]
        if len(collective_keys) != len(set(collective_keys)):
            raise ConfigurationError("AF runtime has ambiguous collective plans")
        for collective in self.collective_plans:
            if not collective.participant_ranks:
                raise ConfigurationError("AF collective must have participants")
            for rank, physical in zip(
                collective.participant_ranks, collective.physical_workers
            ):
                binding = self.binding(collective.domain, rank)
                if (
                    binding.dp_rank != collective.dp_rank
                    or binding.pp_rank != collective.pp_rank
                    or (binding.physical_identity != physical)
                ):
                    raise ConfigurationError(
                        f"AF collective {collective.operation!r} endpoint mismatch"
                    )
