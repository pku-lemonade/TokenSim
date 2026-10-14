"""Planner compute."""

from __future__ import annotations

from typing import Any, Mapping

from TokenSim.af.planning.nodes import ExecutionNode, ExecutionStage, RankLocalWork
from TokenSim.af.providers.queries import AFRoutingEdge, AFTensorLayout, AFTensorSlice
from TokenSim.config.af.domains import ExecutionDomainConfig
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


class ComputeNodePlannerMixin:
    def _runtime_compute_node(
        self,
        *,
        node_id: str,
        operation: str,
        stage: ExecutionStage,
        predecessors: tuple[str, ...],
        binding: Any,
        layer_type: str,
        layer_id: int,
        token_count: int,
        original_token_count: int | None = None,
        assignment_count: int | None = None,
        input_edge_count: int | None = None,
        output_token_count: int | None = None,
        expert_ids: tuple[int, ...] = (),
        expert_workload: tuple[tuple[int, int], ...] = (),
        tensor_slices: tuple[AFTensorSlice, ...] = (),
        source_layout: str | None = None,
        target_layout: str | None = None,
        routing_edges: tuple[AFRoutingEdge, ...] = (),
        source_dtype: str | None = None,
        target_dtype: str | None = None,
        shape: Mapping[str, int] | None = None,
    ) -> ExecutionNode:
        local_shape = dict(shape or {})
        if local_shape and self.runtime_plan is not None:
            base_shape = self.runtime_plan.local_shape(
                binding.domain, binding.domain_rank, layer_id, operation
            )
            for name in ("global_hidden_size", "tp_size"):
                local_shape.setdefault(name, base_shape.get(name, self.hidden_size))
        if not local_shape and operation in {
            "attention",
            "router",
            "dense_ffn",
            "routed_expert",
            "shared_expert",
        }:
            local_shape = self.runtime_plan.local_shape(
                binding.domain,
                binding.domain_rank,
                layer_id,
                operation,
            )
            local_shape["tokens"] = token_count
        if not local_shape and operation in {
            "weighted_combine",
            "residual",
            "layout",
        }:
            owner = self._ownership(binding)
            local_shape = {
                "tokens": token_count,
                "global_hidden_size": self.hidden_size,
                "hidden_size": owner.interval.size,
                "elements": token_count * owner.interval.size,
            }
        if not local_shape:
            local_shape = {
                "tokens": token_count,
                "hidden_size": self.runtime_plan.local_shape(
                    binding.domain,
                    binding.domain_rank,
                    layer_id,
                    (
                        "attention"
                        if binding.capability == "af_attention"
                        else "dense_ffn"
                    ),
                )["hidden_size"],
            }
        local_shape.setdefault("global_hidden_size", self.hidden_size)
        local_shape.setdefault("tp_size", max(1, len(binding.tp_group)))
        work = RankLocalWork(
            token_count=token_count,
            tensor_shape=tuple(sorted(local_shape.items())),
            expert_ids=expert_ids,
            expert_workload=expert_workload,
            original_token_count=original_token_count,
            assignment_count=assignment_count,
            input_edge_count=input_edge_count,
            output_token_count=output_token_count,
            tensor_slices=tensor_slices,
            owner_endpoint=self._endpoint(binding),
            source_layout=source_layout,
            target_layout=target_layout,
            routing_edges=routing_edges,
            source_dtype=source_dtype,
            target_dtype=target_dtype,
        )
        return ExecutionNode(
            node_id=node_id,
            operation=operation,
            kind="compute",
            predecessor_ids=tuple(dict.fromkeys(predecessors)),
            stage=stage,
            direction=None,
            source_domain=binding.domain,
            target_domain=binding.domain,
            logical_participants=(binding.instance,),
            domain_ranks=(binding.domain_rank,),
            physical_workers=(binding.physical_identity,),
            dp_rank=binding.dp_rank,
            tp_rank=binding.tp_rank,
            ep_rank=binding.ep_rank,
            tp_group=(
                (binding.domain_rank,)
                if stage == "routed_expert"
                and self.runtime_plan.domain_runtime(
                    binding.domain
                ).expert_parallel_mode
                == "rank_reuse"
                else binding.tp_group
            ),
            ep_group=binding.ep_group,
            local_work=work,
            layer_type=layer_type,
            source_endpoint=self._endpoint(binding),
            target_endpoint=self._endpoint(binding),
            tensor_slices=tensor_slices,
            source_layout=source_layout,
            target_layout=target_layout,
            routing_edges=routing_edges,
        )

    def _append_runtime_collective(
        self,
        nodes: list[ExecutionNode],
        *,
        node_id: str,
        domain: ExecutionDomainConfig,
        dp_rank: int,
        bindings: tuple[Any, ...],
        producer_ids: Mapping[int, str],
        after_stage: str,
        token_count: int,
        layer_type: str,
        required: bool,
        producer_layout: AFTensorLayout,
        required_target_layout: AFTensorLayout | None = None,
        routing_edges: tuple[AFRoutingEdge, ...] = (),
    ) -> ExecutionNode | None:
        if len(bindings) <= 1:
            return None
        plan = self.runtime_plan.collective_plan(domain.name, dp_rank, after_stage)
        if plan is None:
            if required:
                raise ConfigurationError(
                    f"AF layer requires collective after_stage={after_stage!r} for "
                    f"domain={domain.name!r}, dp_rank={dp_rank}, but no executable "
                    "collective plan is configured"
                )
            return None
        by_rank = {binding.domain_rank: binding for binding in bindings}
        try:
            participants = tuple(by_rank[rank] for rank in plan.participant_ranks)
        except KeyError as exc:
            raise ConfigurationError(
                f"collective {plan.operation!r} for domain={domain.name!r}, "
                f"dp_rank={dp_rank} references rank outside the execution branch"
            ) from exc
        if required and set(plan.participant_ranks) != set(by_rank):
            raise ConfigurationError(
                f"collective {plan.operation!r} after_stage={after_stage!r} for "
                f"domain={domain.name!r}, dp_rank={dp_rank} performs a required "
                "layout transition and must cover the complete execution branch: "
                f"expected_ranks={sorted(by_rank)}, "
                f"participant_ranks={list(plan.participant_ranks)}"
            )
        activation_bytes = dtype_size_bytes(
            self.runtime_plan.effective_dtypes.activation_dtype,
            role="activation_dtype",
        )
        local_width = self.hidden_size // max(1, len(bindings))
        full_bytes = token_count * self.hidden_size * activation_bytes
        local_bytes = token_count * local_width * activation_bytes
        transitions = {
            "all_gather": ("sharded", "replicated"),
            "reduce_scatter": ("partial", "sharded"),
            "all_reduce": ("partial", "replicated"),
        }
        source_layout, target_layout = transitions[plan.operation]
        if source_layout != producer_layout:
            raise ConfigurationError(
                f"collective {plan.operation!r} after_stage={after_stage!r} for "
                f"domain={domain.name!r}, dp_rank={dp_rank} expects "
                f"source_layout={source_layout!r}, but producer emits "
                f"{producer_layout!r}"
            )
        if (
            required_target_layout is not None
            and target_layout != required_target_layout
        ):
            raise ConfigurationError(
                f"collective {plan.operation!r} after_stage={after_stage!r} for "
                f"domain={domain.name!r}, dp_rank={dp_rank} produces "
                f"target_layout={target_layout!r}, but downstream requires "
                f"{required_target_layout!r}"
            )
        if plan.operation == "all_gather":
            input_bytes, output_bytes = local_bytes, full_bytes
        elif plan.operation == "reduce_scatter":
            input_bytes, output_bytes = full_bytes, local_bytes
        else:
            input_bytes = output_bytes = full_bytes
        node = ExecutionNode(
            node_id=node_id,
            operation=plan.operation,
            kind="communication",
            predecessor_ids=tuple(
                producer_ids[item.domain_rank] for item in participants
            ),
            input_producer_ids=tuple(
                (item.physical_identity, producer_ids[item.domain_rank])
                for item in participants
            ),
            stage=None,
            direction=plan.operation,
            source_domain=domain.name,
            target_domain=domain.name,
            logical_participants=tuple(
                dict.fromkeys(item.instance for item in participants)
            ),
            domain_ranks=plan.participant_ranks,
            physical_workers=plan.physical_workers,
            dp_rank=dp_rank,
            tp_group=plan.participant_ranks,
            network=plan.network,
            resource_group=plan.resource_group,
            resource_lane=plan.resource_lane,
            payload_bytes=(
                plan.payload_bytes if plan.payload_bytes is not None else input_bytes
            ),
            logical_input_bytes_per_rank=tuple(
                (item.domain_rank, input_bytes) for item in participants
            ),
            logical_output_bytes_per_rank=tuple(
                (item.domain_rank, output_bytes) for item in participants
            ),
            layer_type=layer_type,
            logical_stage_id=node_id,
            participant_endpoints=tuple(self._endpoint(item) for item in participants),
            source_layout=source_layout,
            target_layout=target_layout,
            metadata={
                "after_stage": after_stage,
                "collective_input_binding_required": True,
                "original_token_count": (
                    len({(edge.request_id, edge.token_id) for edge in routing_edges})
                    if routing_edges
                    else token_count
                ),
                "assignment_count": (len(routing_edges) if routing_edges else None),
            },
            routing_edges=routing_edges,
        )
        nodes.append(node)
        return node
