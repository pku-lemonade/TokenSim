"""Runtime geometry."""

from __future__ import annotations

from typing import Literal

from TokenSim.af.runtime.records import (
    AFRankBinding,
    AFResolvedCollectivePlan,
    AFTensorOwnership,
)
from TokenSim.errors import ConfigurationError
from TokenSim.config.cache_config import stage_layer_range


class RuntimeGeometryMixin:
    def ownerships_for_replica(
        self, domain: str, dp_rank: int, pp_rank: int = 0
    ) -> tuple[AFTensorOwnership, ...]:
        domain_plan = self.domain_runtime(domain)
        candidates = self._ownership_index.get((domain, dp_rank, pp_rank), ())
        if domain_plan.capability == "af_ffn" and domain_plan.expert_parallel_size > 1:
            # Residual activations have one base TP owner.  EP ranks in a
            # rank-reuse F pool replicate that owner and must not create
            # overlapping tensor intervals.
            by_tp: dict[int, AFTensorOwnership] = {}
            for item in sorted(candidates, key=lambda value: value.domain_rank):
                by_tp.setdefault(item.tp_rank, item)
            candidates = tuple(by_tp.values())
        ownerships = tuple(
            sorted(
                candidates,
                key=lambda item: item.tp_rank,
            )
        )
        if not ownerships:
            raise ConfigurationError(
                f"AF runtime plan has no tensor ownership for domain={domain!r}, "
                f"dp_rank={dp_rank}"
            )
        return ownerships

    def collective_plan(
        self, domain: str, dp_rank: int, after_stage: str, pp_rank: int = 0
    ) -> AFResolvedCollectivePlan | None:
        matches = tuple(
            item
            for item in self.collective_plans
            if item.domain == domain
            and item.dp_rank == dp_rank
            and item.after_stage == after_stage
            and item.pp_rank == pp_rank
        )
        if len(matches) > 1:
            raise ConfigurationError(
                f"AF runtime plan has ambiguous collective for domain={domain!r}, "
                f"dp_rank={dp_rank}, after_stage={after_stage!r}"
            )
        return matches[0] if matches else None

    def pipeline_stage(self, domain: str, layer_id: int) -> int:
        self.layer_plan(layer_id)
        for stage in range(self.domain_runtime(domain).pipeline_parallel_size):
            if layer_id in stage_layer_range(
                self.total_layers,
                self.domain_runtime(domain).pipeline_parallel_size,
                stage,
                self.domain_runtime(domain).pipeline_layer_counts,
            ):
                return stage
        raise ConfigurationError("layer has no pipeline owner")

    def bindings_for_layer(self, domain: str, instance: str, layer_id: int):
        stage = self.pipeline_stage(domain, layer_id)
        return tuple(
            binding
            for binding in self.bindings_for_instance(domain, instance)
            if binding.pp_rank == stage
        )

    def local_shape(
        self,
        domain: str,
        domain_rank: int,
        layer_id: int,
        operation: str,
    ) -> dict[str, int]:
        binding = self.binding(domain, domain_rank)
        domain_plan = self.domain_runtime(domain)
        layer = self.layer_plan(layer_id)
        shape = {
            "global_hidden_size": self.hidden_size,
            "hidden_size": self.hidden_size,
            "tp_size": domain_plan.tensor_parallel_size,
        }
        if operation in {"attention", "a_rank_compute"}:
            partition = (
                binding.kv_partition.for_layer(layer_id)
                if binding.kv_partition
                else None
            )
            if binding.kv_partition is None:
                raise ConfigurationError(
                    f"attention rank {binding.rank_key!r} has no KV partition"
                )
            shape.update(
                {
                    "num_heads": layer.num_attention_heads
                    // domain_plan.tensor_parallel_size,
                    "num_kv_heads": partition.local_head_count,
                    "head_dim": layer.head_dim,
                    "kv_projections": 1 if layer.k_eq_v else 2,
                    "query_width": (
                        layer.num_attention_heads
                        // domain_plan.tensor_parallel_size
                        * layer.head_dim
                    ),
                    "kv_width": (partition.local_head_count * layer.head_dim),
                }
            )
            if self.mla is not None:
                shape.update(
                    self.mla.local_shape(
                        layer.num_attention_heads, domain_plan.tensor_parallel_size
                    )
                )
        elif operation == "dense_ffn":
            if layer.layer_type != "dense":
                raise ConfigurationError(
                    f"layer={layer_id} is {layer.layer_type}, not Dense FFN"
                )
            shape["intermediate_size"] = (
                layer.dense_intermediate_size // domain_plan.tensor_parallel_size
            )
        elif operation == "shared_expert":
            if layer.layer_type != "moe" or not layer.shared_intermediate_size:
                raise ConfigurationError(
                    f"layer={layer_id} has no Shared Expert component"
                )
            shape["intermediate_size"] = (
                layer.shared_intermediate_size // domain_plan.tensor_parallel_size
            )
        elif operation in {"routed_expert", "expert_rank_compute"}:
            if layer.layer_type != "moe" or not layer.num_experts:
                raise ConfigurationError(
                    f"layer={layer_id} has no Routed Expert component"
                )
            expert_tp_size = domain_plan.expert_tensor_parallel_size
            shape.update(
                {
                    "tp_size": expert_tp_size,
                    "expert_tp_size": expert_tp_size,
                    "expert_tp_rank": binding.tp_rank % expert_tp_size,
                    "intermediate_size": (
                        layer.routed_intermediate_size // expert_tp_size
                    ),
                }
            )
        return shape

    def expert_tp_group(self, binding: AFRankBinding) -> tuple[int, ...]:
        domain = self.domain_runtime(binding.domain)
        if domain.expert_parallel_mode == "expert_tp":
            return binding.tp_group
        return self._expert_group_index[
            (binding.domain, binding.pp_rank, binding.ep_rank)
        ]

    def expert_execution(
        self, domain: str, expert_id: int, f_service_dp_rank: int, layer_id: int = 0
    ) -> tuple[AFRankBinding, ...]:
        domain_plan = self.domain_runtime(domain)
        if domain_plan.capability != "af_ffn":
            raise ConfigurationError(
                f"expert execution requires an F domain, got {domain!r}"
            )
        if not 0 <= f_service_dp_rank < domain_plan.data_parallel_size:
            raise ConfigurationError(
                f"F domain={domain!r} has no service replica "
                f"dp_rank={f_service_dp_rank}"
            )
        candidates = self._expert_index.get(
            (domain, self.pipeline_stage(domain, layer_id), expert_id), ()
        )
        if domain_plan.expert_parallel_mode == "expert_tp":
            candidates = tuple(
                binding
                for binding in candidates
                if binding.dp_rank == f_service_dp_rank
            )
        resolved = tuple(
            sorted(candidates, key=lambda item: (item.dp_rank, item.tp_rank))
        )
        if len(resolved) != domain_plan.expert_tensor_parallel_size:
            raise ConfigurationError(
                f"expert={expert_id} in F domain={domain!r} resolves to "
                f"{len(resolved)} ranks; expected Expert TP="
                f"{domain_plan.expert_tensor_parallel_size}"
            )
        if len({binding.physical_identity for binding in resolved}) != len(resolved):
            raise ConfigurationError(
                f"expert={expert_id} in F domain={domain!r} resolves multiple "
                "Expert TP ranks to the same physical worker"
            )
        return resolved

    def component_execution(
        self,
        domain: str,
        layer_id: int,
        component_category: Literal["dense_ffn", "shared_expert"],
        f_service_dp_rank: int,
    ) -> tuple[AFRankBinding, ...]:
        domain_plan = self.domain_runtime(domain)
        if domain_plan.capability != "af_ffn":
            raise ConfigurationError(
                f"F component execution requires an F domain, got {domain!r}"
            )
        if component_category not in {"dense_ffn", "shared_expert"}:
            raise ConfigurationError(
                f"unsupported F TP component {component_category!r}; expected "
                "'dense_ffn' or 'shared_expert'"
            )
        if not 0 <= f_service_dp_rank < domain_plan.data_parallel_size:
            raise ConfigurationError(
                f"F domain={domain!r} has no service replica "
                f"dp_rank={f_service_dp_rank}"
            )

        layer = self.layer_plan(layer_id)
        if component_category == "dense_ffn" and layer.layer_type != "dense":
            raise ConfigurationError(
                f"layer={layer_id} is {layer.layer_type}, not Dense FFN"
            )
        if component_category == "shared_expert" and (
            layer.layer_type != "moe" or not layer.shared_intermediate_size
        ):
            raise ConfigurationError(f"layer={layer_id} has no Shared Expert component")

        candidates = self.bindings_for_layer(
            domain, self.replica(domain, f_service_dp_rank).instance, layer_id
        )
        if domain_plan.capability == "af_ffn" and domain_plan.expert_parallel_size > 1:
            # Dense/shared weights are replicated on every EP rank in a
            # rank-reuse F pool.  Execute one representative per base TP
            # coordinate; routed experts are resolved separately by
            # ``expert_execution``.
            by_tp: dict[int, AFRankBinding] = {}
            for binding in sorted(candidates, key=lambda item: item.domain_rank):
                by_tp.setdefault(binding.tp_rank, binding)
            resolved = tuple(sorted(by_tp.values(), key=lambda item: item.tp_rank))
        else:
            resolved = tuple(sorted(candidates, key=lambda item: item.tp_rank))
        if len(resolved) != domain_plan.tensor_parallel_size:
            raise ConfigurationError(
                f"F component={component_category!r}, layer={layer_id}, "
                f"domain={domain!r}, dp_rank={f_service_dp_rank} resolves to "
                f"{len(resolved)} ranks; expected F TP="
                f"{domain_plan.tensor_parallel_size}"
            )
        expected_group = tuple(binding.domain_rank for binding in resolved)
        for binding in resolved:
            if binding.tp_group != expected_group:
                raise ConfigurationError(
                    f"F component={component_category!r}, layer={layer_id} rank "
                    f"{binding.rank_key!r} has TP group {binding.tp_group}; "
                    f"expected {expected_group}"
                )
            state = self.physical_state(binding.physical_identity)
            matches = tuple(
                component
                for component in state.components
                if component.category == component_category
                and component.layer_id == layer_id
                and component.component_id in binding.component_ids
            )
            if len(matches) != 1:
                raise ConfigurationError(
                    f"F component={component_category!r}, layer={layer_id} does "
                    f"not resolve uniquely on domain={domain!r}, "
                    f"rank={binding.domain_rank}; found {len(matches)}"
                )
        if len({binding.physical_identity for binding in resolved}) != len(resolved):
            raise ConfigurationError(
                f"F component={component_category!r}, layer={layer_id}, "
                f"domain={domain!r}, dp_rank={f_service_dp_rank} reuses a "
                "physical worker inside its TP group"
            )
        return resolved
