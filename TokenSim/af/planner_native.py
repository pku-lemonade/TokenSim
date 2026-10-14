"""Planner native."""

from __future__ import annotations

from TokenSim.af.planning_dag import AFExecutionDAG
from TokenSim.af.snapshot import snapshot
from TokenSim.errors import ConfigurationError


class NativeLayerPlannerMixin:
    def _build_native_moe_tail(
        self,
        *,
        layer_id,
        layer,
        snapshot,
        link,
        prefix,
        nodes,
        a_bindings,
        f_bindings,
        attention_ids,
        a_collective,
    ) -> AFExecutionDAG:
        """Single-rank Qwen AFD: send hidden, route/combine on F, return hidden.

        The planner retains mathematical combine as a separate device service.
        A fused measured estimate must explicitly cover it before that service
        can be elided. No communication time is inferred from expert count.
        """
        if len(a_bindings) != 1 or len(f_bindings) != 1 or layer.num_shared_experts:
            raise ConfigurationError(
                "native AFD MoE tail requires 1A1F without shared experts"
            )
        a, f = a_bindings[0], f_bindings[0]
        tokens = tuple(snapshot.original_tokens)
        edges = tuple(self._routing_edge(item) for item in snapshot.assignments)
        router_id, expert_id, combine_id, residual_id = (
            f"{prefix}:{name}"
            for name in ("router", "grouped_experts", "f_combine", "residual")
        )
        copies = self._activation_copies(
            tokens=tokens,
            a_bindings=a_bindings,
            target_specs=((f, router_id, tokens),),
            producer_ids=attention_ids,
            collective=a_collective,
            barrier_predecessor=None,
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype,
        )
        arrivals = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:a_to_f",
            operation="af_a_to_f",
            direction="a_to_f",
            copies=copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=router_id,
                operation="router_topk",
                stage="router",
                predecessors=arrivals[router_id],
                binding=f,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(tokens),
                original_token_count=len(tokens),
                assignment_count=len(edges),
                output_token_count=len(tokens),
                routing_edges=edges,
                expert_ids=tuple(e for e, _ in snapshot.expert_histogram),
                shape={
                    "tokens": len(tokens),
                    "hidden_size": self.hidden_size,
                    "num_experts": layer.num_experts,
                    "top_k": layer.top_k,
                },
            )
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=expert_id,
                operation="routed_expert",
                stage="routed_expert",
                predecessors=(router_id,),
                binding=f,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(edges),
                original_token_count=len(tokens),
                assignment_count=len(edges),
                output_token_count=len(edges),
                routing_edges=edges,
                expert_ids=tuple(e for e, _ in snapshot.expert_histogram),
                expert_workload=tuple(snapshot.expert_histogram),
            )
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=combine_id,
                operation="f_weighted_combine",
                stage="weighted_combine",
                predecessors=(expert_id,),
                binding=f,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(tokens),
                original_token_count=len(tokens),
                assignment_count=len(edges),
                input_edge_count=len(edges),
                output_token_count=len(tokens),
                routing_edges=edges,
                source_layout="partial",
                target_layout="replicated",
                shape={
                    "tokens": len(tokens),
                    "hidden_size": self.hidden_size,
                    "elements": len(tokens) * self.hidden_size,
                },
            )
        )
        copies = self._output_copies(
            items=tuple(
                (token, f"request:{token[0]}:token:{token[1]}:combined", None, None)
                for token in tokens
            ),
            source_bindings=f_bindings,
            producer_ids={f.domain_rank: combine_id},
            collective=None,
            target_bindings=a_bindings,
            consumer_ids={a.domain_rank: residual_id},
            tensor_layout="replicated",
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).f_to_a_dtype,
        )
        returned = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:f_to_a",
            operation="af_f_to_a",
            direction="f_to_a",
            copies=copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        nodes.append(
            self._runtime_compute_node(
                node_id=residual_id,
                operation="residual",
                stage="residual",
                predecessors=returned[residual_id],
                binding=a,
                layer_type="moe",
                layer_id=layer_id,
                token_count=len(tokens),
                original_token_count=len(tokens),
                output_token_count=len(tokens),
                source_layout="replicated",
                target_layout="replicated",
            )
        )
        return AFExecutionDAG(tuple(nodes))
