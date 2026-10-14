"""Planner moe."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.compute import group_routed_expert_nodes
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode, _TransferCopy
from TokenSim.af.providers.queries import AFRoutingEdge
from TokenSim.config.af.domains import ExecutionDomainConfig
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError


class MoELayerPlannerMixin:
    def _build_runtime_moe_tail(
        self,
        *,
        block: AFBlock,
        layer_id: int,
        layer: Any,
        snapshot: Any,
        link: AFLinkConfig,
        prefix: str,
        nodes: list[ExecutionNode],
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
        a_bindings: tuple[Any, ...],
        f_service_bindings: tuple[Any, ...],
        attention_ids: Mapping[int, str],
        a_collective: ExecutionNode | None,
    ) -> AFExecutionDAG:
        tokens = tuple(snapshot.original_tokens)
        if self.cluster_config.af.execution_protocol == "vllm_afd":
            return self._build_native_moe_tail(
                layer_id=layer_id,
                layer=layer,
                snapshot=snapshot,
                link=link,
                prefix=prefix,
                nodes=nodes,
                a_bindings=a_bindings,
                f_bindings=f_service_bindings,
                attention_ids=attention_ids,
                a_collective=a_collective,
            )
        a_leader = min(a_bindings, key=lambda item: item.tp_rank)
        # ``attention_ids`` is the post-attention norm frontier in the MLA
        # runtime.  Using the collective directly here would leave those norm
        # nodes as orphan terminals when TP>1.
        a_ready = tuple(attention_ids.values())
        routing_edges = tuple(
            self._routing_edge(assignment) for assignment in snapshot.assignments
        )
        router_id = f"{prefix}:router"
        router_on_f = self.cluster_config.af.router_location == "f"
        if router_on_f and f_domain.parallel.data_parallel_size != 1:
            raise ConfigurationError("F-side leader Router requires one F DP replica")
        if (
            not router_on_f
            and self.cluster_config.af.transport_router_logits
            and (len(a_bindings) != 1 or len(f_service_bindings) != 1)
        ):
            raise ConfigurationError(
                "Transporting A-side Router logits requires one A rank and one F service rank"
            )
        router_binding = f_service_bindings[0] if router_on_f else a_leader
        router_node = self._runtime_compute_node(
            node_id=router_id,
            operation="router_topk",
            stage="router",
            predecessors=a_ready,
            binding=router_binding,
            layer_type="moe",
            layer_id=layer_id,
            token_count=len(tokens),
            original_token_count=len(tokens),
            assignment_count=len(snapshot.assignments),
            output_token_count=len(tokens),
            expert_ids=tuple(expert for expert, _ in snapshot.expert_histogram),
            shape={
                "tokens": len(tokens),
                "hidden_size": self.hidden_size,
                "num_experts": layer.num_experts,
                "top_k": layer.top_k,
            },
            routing_edges=routing_edges,
        )
        if not router_on_f:
            nodes.append(router_node)

        assignments_by_expert: dict[int, list[Any]] = {}
        for assignment in snapshot.assignments:
            assignments_by_expert.setdefault(assignment.expert_id, []).append(
                assignment
            )
        routed_specs: list[tuple[Any, str, tuple[tuple[int, int], ...]]] = []
        routed_edges_by_consumer: dict[str, tuple[AFRoutingEdge, ...]] = {}
        routed_nodes: list[
            tuple[int, tuple[Any, ...], dict[int, str], tuple[Any, ...]]
        ] = []
        for expert_id, assignments in sorted(assignments_by_expert.items()):
            bindings = self.runtime_plan.expert_execution(
                f_domain.name,
                expert_id,
                block.placement.f_dp_rank,
            )
            ids = {
                binding.domain_rank: (
                    f"{prefix}:routed:e{expert_id}:r{binding.domain_rank}"
                )
                for binding in bindings
            }
            expert_tokens = tuple(
                (assignment.request_id, assignment.token_id)
                for assignment in assignments
            )
            for binding in bindings:
                routed_specs.append((binding, ids[binding.domain_rank], expert_tokens))
                routed_edges_by_consumer[ids[binding.domain_rank]] = tuple(
                    self._routing_edge(assignment) for assignment in assignments
                )
            routed_nodes.append((expert_id, bindings, ids, tuple(assignments)))

        shared_ids: dict[int, str] = {}
        shared_specs: list[tuple[Any, str, tuple[tuple[int, int], ...]]] = []
        if layer.num_shared_experts > 0:
            shared_bindings = self.runtime_plan.component_execution(
                f_domain.name,
                layer_id,
                "shared_expert",
                block.placement.f_dp_rank,
            )
            shared_ids = {
                binding.domain_rank: f"{prefix}:shared:r{binding.domain_rank}"
                for binding in shared_bindings
            }
            shared_specs.extend(
                (binding, shared_ids[binding.domain_rank], tokens)
                for binding in shared_bindings
            )
        else:
            shared_bindings = ()

        # With an F-side leader Router, every F rank receives the full block
        # independently of the sampled routing. Selecting remote destinations
        # before the Router finishes would grant the scheduler free lookahead.
        multi_f_router = router_on_f and (
            len(f_service_bindings) > 1 or len(a_bindings) > 1
        )
        activation_specs = (
            tuple(
                (binding, f"{prefix}:f_input:r{binding.domain_rank}", tokens)
                for binding in f_service_bindings
            )
            if multi_f_router
            else tuple(routed_specs + shared_specs)
        )
        a_to_f_copies = self._activation_copies(
            tokens=tokens,
            a_bindings=a_bindings,
            target_specs=activation_specs,
            producer_ids=attention_ids,
            collective=a_collective,
            barrier_predecessor=None if router_on_f else router_id,
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype,
            routing_edges_by_consumer=(
                None if multi_f_router else routed_edges_by_consumer
            ),
        )
        arrivals = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:a_to_f",
            operation="af_a_to_f",
            direction="a_to_f",
            copies=a_to_f_copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )

        if not router_on_f and self.cluster_config.af.transport_router_logits:
            arrivals = self._append_router_logits(
                nodes=nodes,
                tokens=tokens,
                a_leader=a_leader,
                f_service_bindings=f_service_bindings,
                router_id=router_id,
                layer=layer,
                link=link,
                prefix=prefix,
                layer_id=layer_id,
                arrivals=arrivals,
            )
        combine_routing_ready = {}
        if router_on_f:
            leader_arrivals = (
                arrivals[f"{prefix}:f_input:r{router_binding.domain_rank}"]
                if multi_f_router
                else tuple(
                    dict.fromkeys(
                        pred
                        for predecessors in arrivals.values()
                        for pred in predecessors
                    )
                )
            )
            router_node = replace(router_node, predecessor_ids=leader_arrivals)
            nodes.append(router_node)
            routing_ready = {router_binding.domain_rank: router_id}
            # A-side weighted Combine also needs the selected IDs/weights.
            for binding in (*f_service_bindings, *a_bindings):
                if (
                    binding.domain == f_domain.name
                    and binding.domain_rank == router_binding.domain_rank
                ):
                    continue
                node_id = (
                    f"{prefix}:routing_metadata:{binding.domain}:r{binding.domain_rank}"
                )
                source = self._endpoint(router_binding)
                target = self._endpoint(binding)
                nodes.append(
                    ExecutionNode(
                        node_id=node_id,
                        operation="af_router_metadata",
                        kind="communication",
                        predecessor_ids=(router_id,),
                        stage=None,
                        direction=(
                            "f_to_f" if binding.domain == f_domain.name else "f_to_a"
                        ),
                        source_domain=f_domain.name,
                        target_domain=binding.domain,
                        logical_participants=(
                            router_binding.instance,
                            binding.instance,
                        ),
                        domain_ranks=(router_binding.domain_rank, binding.domain_rank),
                        physical_workers=(
                            router_binding.physical_identity,
                            binding.physical_identity,
                        ),
                        dp_rank=binding.dp_rank,
                        network=link.network,
                        resource_group=link.resource_group,
                        resource_lane="shared" if link.duplex == "half" else "f_to_a",
                        # Explicit protocol: int32 expert IDs + float32 weights.
                        payload_bytes=len(tokens) * layer.top_k * 8,
                        layer_type="moe",
                        logical_stage_id=f"{prefix}:routing_metadata",
                        leg_id=node_id,
                        source_endpoint=source,
                        target_endpoint=target,
                        participant_endpoints=(source, target),
                        metadata={
                            "original_token_count": len(tokens),
                            "assignment_count": len(snapshot.assignments),
                        },
                    )
                )
                if binding.domain == f_domain.name:
                    routing_ready[binding.domain_rank] = node_id
                else:
                    combine_routing_ready[binding.domain_rank] = node_id
            if multi_f_router:
                arrivals = {
                    consumer: tuple(
                        dict.fromkeys(
                            (
                                *arrivals[f"{prefix}:f_input:r{binding.domain_rank}"],
                                routing_ready[binding.domain_rank],
                            )
                        )
                    )
                    for binding, consumer, _ in routed_specs + shared_specs
                }
            else:
                arrivals = {
                    name: tuple(dict.fromkeys((*preds, router_id)))
                    for name, preds in arrivals.items()
                }

        routed_outputs: list[_TransferCopy] = []
        combine_ids = {
            binding.domain_rank: f"{prefix}:combine:r{binding.domain_rank}"
            for binding in a_bindings
        }
        f_to_a_dtype = self.runtime_plan.dtypes_for_link(link.name).f_to_a_dtype
        for expert_id, bindings, ids, assignments in routed_nodes:
            producer_ids: dict[int, str] = {}
            for binding in bindings:
                node_id = ids[binding.domain_rank]
                producer_ids[binding.domain_rank] = node_id
                nodes.append(
                    self._runtime_compute_node(
                        node_id=node_id,
                        operation="routed_expert",
                        stage="routed_expert",
                        predecessors=arrivals[node_id],
                        binding=binding,
                        layer_type="moe",
                        layer_id=layer_id,
                        token_count=len(assignments),
                        original_token_count=len(
                            {(item.request_id, item.token_id) for item in assignments}
                        ),
                        assignment_count=len(assignments),
                        output_token_count=len(assignments),
                        expert_ids=(expert_id,),
                        expert_workload=((expert_id, len(assignments)),),
                        routing_edges=tuple(
                            self._routing_edge(assignment) for assignment in assignments
                        ),
                    )
                )
            collective = self._append_runtime_collective(
                nodes,
                node_id=f"{prefix}:routed:e{expert_id}:collective",
                domain=f_domain,
                dp_rank=bindings[0].dp_rank,
                bindings=bindings,
                producer_ids=producer_ids,
                after_stage="expert_compute",
                token_count=len(assignments),
                layer_type="moe",
                required=len(bindings) > 1,
                producer_layout="partial",
                routing_edges=tuple(
                    self._routing_edge(assignment) for assignment in assignments
                ),
            )
            items = tuple(
                (
                    (assignment.request_id, assignment.token_id),
                    f"request:{assignment.request_id}:token:{assignment.token_id}:"
                    f"routed:k{assignment.top_k_index}:expert:{expert_id}",
                    f"expert:{expert_id}:k{assignment.top_k_index}:"
                    f"weight:{assignment.routing_weight:.17g}",
                    self._routing_edge(assignment),
                )
                for assignment in assignments
            )
            routed_outputs.extend(
                self._output_copies(
                    items=items,
                    source_bindings=bindings,
                    producer_ids=producer_ids,
                    collective=collective,
                    target_bindings=a_bindings,
                    consumer_ids=combine_ids,
                    tensor_layout="partial",
                    transfer_dtype=f_to_a_dtype,
                )
            )

        shared_outputs: list[_TransferCopy] = []
        if shared_bindings:
            shared_compute_ids: dict[int, str] = {}
            for binding in shared_bindings:
                node_id = shared_ids[binding.domain_rank]
                shared_compute_ids[binding.domain_rank] = node_id
                nodes.append(
                    self._runtime_compute_node(
                        node_id=node_id,
                        operation="shared_expert",
                        stage="shared_expert",
                        predecessors=arrivals[node_id],
                        binding=binding,
                        layer_type="moe",
                        layer_id=layer_id,
                        token_count=len(tokens),
                        original_token_count=len(tokens),
                        output_token_count=len(tokens),
                    )
                )
            shared_collective = self._append_runtime_collective(
                nodes,
                node_id=f"{prefix}:shared:collective",
                domain=f_domain,
                dp_rank=block.placement.f_dp_rank,
                bindings=shared_bindings,
                producer_ids=shared_compute_ids,
                after_stage="expert_compute",
                token_count=len(tokens),
                layer_type="moe",
                required=len(shared_bindings) > 1,
                producer_layout="partial",
            )
            shared_outputs.extend(
                self._output_copies(
                    items=tuple(
                        (
                            token,
                            f"request:{token[0]}:token:{token[1]}:shared",
                            "shared",
                            None,
                        )
                        for token in tokens
                    ),
                    source_bindings=shared_bindings,
                    producer_ids=shared_compute_ids,
                    collective=shared_collective,
                    target_bindings=a_bindings,
                    consumer_ids=combine_ids,
                    tensor_layout="partial",
                    transfer_dtype=f_to_a_dtype,
                )
            )

        f_to_a_copies = tuple(routed_outputs + shared_outputs)
        returned = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:f_to_a",
            operation="af_f_to_a",
            direction="f_to_a",
            copies=f_to_a_copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        residual_ids: dict[int, str] = {}
        for binding in a_bindings:
            combine_id = combine_ids[binding.domain_rank]
            owner_slices = tuple(
                copy.tensor_slice
                for copy in f_to_a_copies
                if copy.target_binding.domain_rank == binding.domain_rank
            )
            nodes.append(
                self._runtime_compute_node(
                    node_id=combine_id,
                    operation="weighted_combine",
                    stage="weighted_combine",
                    predecessors=returned[combine_id]
                    + (
                        (combine_routing_ready[binding.domain_rank],)
                        if binding.domain_rank in combine_routing_ready
                        else ()
                    ),
                    binding=binding,
                    layer_type="moe",
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    assignment_count=len(snapshot.assignments),
                    input_edge_count=(
                        len(snapshot.assignments)
                        + (len(tokens) if shared_bindings else 0)
                    ),
                    output_token_count=len(tokens),
                    tensor_slices=owner_slices,
                    source_layout="partial",
                    target_layout="sharded",
                    routing_edges=routing_edges,
                )
            )
            residual_id = f"{prefix}:residual:r{binding.domain_rank}"
            residual_ids[binding.domain_rank] = residual_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=residual_id,
                    operation="residual",
                    stage="residual",
                    predecessors=(combine_id,),
                    binding=binding,
                    layer_type="moe",
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    output_token_count=len(tokens),
                    source_layout="sharded",
                    target_layout="sharded",
                )
            )
        self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:residual:layout",
            domain=a_domain,
            dp_rank=block.placement.a_dp_rank,
            bindings=a_bindings,
            producer_ids=residual_ids,
            after_stage="residual",
            token_count=len(tokens),
            layer_type="moe",
            required=len(a_bindings) > 1,
            producer_layout="sharded",
            required_target_layout="replicated",
        )
        if self.cluster_config.af.routed_expert_execution == "grouped":
            nodes = list(group_routed_expert_nodes(tuple(nodes)))
        return AFExecutionDAG(tuple(nodes))
