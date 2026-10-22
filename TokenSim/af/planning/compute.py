"""Rank-local compute nodes, expert grouping and router-logit transfers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from TokenSim.af.planning.coalesce import rewrite_dependencies
from TokenSim.af.planning.nodes import (
    ExecutionNode,
    ExecutionStage,
    RankLocalWork,
    _TransferCopy,
)
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
                self.runtime_plan.expert_tp_group(binding)
                if stage == "routed_expert"
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
        plan = self.runtime_plan.collective_plan(
            domain.name, dp_rank, after_stage, bindings[0].pp_rank
        )
        if plan is None:
            if required:
                raise ConfigurationError(
                    f"AF layer requires collective after_stage={after_stage!r} for "
                    f"domain={domain.name!r}, dp_rank={dp_rank}, but no executable "
                    "collective plan is configured"
                )
            return None
        if (
            after_stage == "expert_compute"
            and domain.expert_parallel.mode == "expert_sharded"
        ):
            ranks = tuple(binding.domain_rank for binding in bindings)
            if not set(ranks).issubset(plan.participant_ranks):
                raise ConfigurationError(
                    "expert collective template does not cover the expert TP group"
                )
            plan = replace(
                plan,
                participant_ranks=ranks,
                physical_workers=tuple(
                    binding.physical_identity for binding in bindings
                ),
            )
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
                **(
                    {"collective_algorithm": plan.algorithm}
                    if plan.algorithm != "ring"
                    else {}
                ),
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

    def _append_router_logits(
        self,
        *,
        nodes,
        tokens,
        a_leader,
        f_service_bindings,
        router_id,
        layer,
        link,
        prefix,
        layer_id,
        arrivals,
    ):
        dtype = self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype
        target = f_service_bindings[0]
        logits_copies = tuple(
            _TransferCopy(
                source_binding=a_leader,
                target_binding=target,
                tensor_slice=AFTensorSlice(
                    tensor_item_id=f"request:{request}:token:{token}:router_logits",
                    interval=(0, layer.num_experts),
                    layout="replicated",
                    owner_domain=a_leader.domain,
                    owner_domain_rank=a_leader.domain_rank,
                    owner_dp_rank=a_leader.dp_rank,
                    owner_tp_rank=a_leader.tp_rank,
                    owner_physical_worker=a_leader.physical_identity,
                    canonical_producer=router_id,
                    element_count=layer.num_experts,
                    payload_bytes=layer.num_experts
                    * dtype_size_bytes(dtype, role="a_to_f_dtype"),
                ),
                predecessor_id=router_id,
                consumer_id="expert_logits",
                source_layout="replicated",
                target_layout="replicated",
            )
            for request, token in tokens
        )
        logits_arrivals = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:router_logits",
            operation="af_router_logits",
            direction="a_to_f",
            copies=logits_copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        arrivals = {
            name: tuple(dict.fromkeys((*preds, *logits_arrivals["expert_logits"])))
            for name, preds in arrivals.items()
        }
        return arrivals


def group_routed_expert_nodes(
    nodes: tuple[ExecutionNode, ...]
) -> tuple[ExecutionNode, ...]:
    """Group the independent expert branches of one block/layer runtime DAG.

    Keep every routing edge and output transfer/collective. Only the compute
    execution boundary changes; the group waits for all of its inputs.
    """
    groups: dict[tuple, list[ExecutionNode]] = {}
    for node in nodes:
        if node.stage != "routed_expert" or node.kind != "compute":
            continue
        work = node.local_work
        if work is None:
            raise ConfigurationError("routed expert missing local work")
        # The prefix encodes iteration/block/layer. Domain/rank and layouts are
        # included even if two logical workers share a physical GPU.
        key = (
            node.node_id.split(":routed:")[0],
            node.physical_workers,
            node.source_domain,
            node.target_domain,
            node.domain_ranks,
            node.dp_rank,
            node.tp_rank,
            node.ep_rank,
            node.tp_group,
            node.ep_group,
            node.layer_type,
            node.logical_participants,
            work.source_dtype,
            work.target_dtype,
            work.source_layout,
            work.target_layout,
            tuple((k, v) for k, v in work.tensor_shape if k != "tokens"),
        )
        groups.setdefault(key, []).append(node)
    aliases = {}
    replacements = {}
    by_id = {node.node_id: node for node in nodes}
    for group in groups.values():
        if len(group) < 2:
            continue
        member_ids = {node.node_id for node in group}
        pending = [pred for node in group for pred in node.predecessor_ids]
        visited = set()
        while pending:
            pred = pending.pop()
            if pred in member_ids:
                raise ConfigurationError("cannot group dependent routed expert nodes")
            if pred not in visited:
                visited.add(pred)
                pending.extend(by_id[pred].predecessor_ids)
        first = group[0]
        edges = tuple(edge for node in group for edge in node.local_work.routing_edges)
        workload = tuple(
            sorted(pair for node in group for pair in node.local_work.expert_workload)
        )
        if len({expert for expert, _ in workload}) != len(workload):
            raise ConfigurationError("expert group contains duplicate expert workloads")
        count = sum(value for _, value in workload)
        shape = dict(first.local_work.tensor_shape)
        shape["tokens"] = count
        work = replace(
            first.local_work,
            token_count=count,
            tensor_shape=tuple(sorted(shape.items())),
            expert_ids=tuple(expert for expert, _ in workload),
            expert_workload=workload,
            assignment_count=count,
            original_token_count=len(
                {(edge.request_id, edge.token_id) for edge in edges}
            ),
            output_token_count=count,
            routing_edges=edges,
            tensor_slices=tuple(
                item for node in group for item in node.local_work.tensor_slices
            ),
        )
        group_id = (
            first.node_id.split(":routed:")[0]
            + ":routed:group:"
            + first.node_id.split(":routed:")[1]
        )
        aliases.update({node.node_id: group_id for node in group})
        replacements[first.node_id] = replace(
            first,
            node_id=group_id,
            local_work=work,
            routing_edges=edges,
            tensor_slices=tuple(item for node in group for item in node.tensor_slices),
            predecessor_ids=tuple(
                dict.fromkeys(pred for node in group for pred in node.predecessor_ids)
            ),
            metadata={
                **first.metadata,
                "execution_unit": "expert_group",
                "member_node_ids": tuple(node.node_id for node in group),
            },
        )
    result = []
    for node in nodes:
        if node.node_id in aliases and node.node_id not in replacements:
            continue
        node = replacements.get(node.node_id, node)
        result.append(node)
    return rewrite_dependencies(result, aliases)
