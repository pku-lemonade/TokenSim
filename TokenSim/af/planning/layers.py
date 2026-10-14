"""Planner layer."""

from __future__ import annotations

from typing import Any, Mapping

from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.config.af.domains import ExecutionDomainConfig
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError


class RuntimeLayerPlannerMixin:
    def _build_runtime_layer(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
    ) -> AFExecutionDAG:
        if block.placement is None:
            raise ConfigurationError(
                f"AF block {block.block_id} has no preselected DP placement"
            )
        layer = self.runtime_plan.layer_plan(layer_id)
        snapshot = block.routing_snapshot_for_layer(layer_id)
        if layer.layer_type == "dense" and snapshot is not None:
            raise ConfigurationError(
                f"Dense AF layer {layer_id} must not contain routing state"
            )
        if layer.layer_type == "moe" and snapshot is None:
            raise ConfigurationError(
                f"MoE AF layer {layer_id} requires a frozen routing snapshot"
            )

        a_domain = self.cluster_config.execution_domains[block.placement.a_domain]
        f_domain = self.cluster_config.execution_domains[block.placement.f_domain]
        a_instance = self._instance(a_domain, block.placement.a_instance)
        f_instance = self._instance(f_domain, block.placement.f_instance)
        a_bindings = self.runtime_plan.bindings_for_instance(
            a_domain.name, a_instance.name
        )
        f_service_bindings = self.runtime_plan.bindings_for_instance(
            f_domain.name, f_instance.name
        )
        if (
            f_domain.capability == "af_ffn"
            and f_domain.parallel.expert_parallel_size > 1
        ):
            # Dense FFN and shared-expert work is replicated across the EP
            # ranks.  Keep one representative per base TP coordinate for
            # those stages; routed experts below still resolve their owning EP
            # rank through ``expert_execution``.
            f_by_tp: dict[int, Any] = {}
            for binding in sorted(
                f_service_bindings, key=lambda item: item.domain_rank
            ):
                f_by_tp.setdefault(binding.tp_rank, binding)
            f_base_bindings = tuple(binding for _, binding in sorted(f_by_tp.items()))
        else:
            f_base_bindings = f_service_bindings
        prefix = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}"
        nodes: list[ExecutionNode] = []

        attention_ids: dict[int, str] = {}
        for binding in a_bindings:
            node_id = f"{prefix}:attention:r{binding.domain_rank}"
            attention_ids[binding.domain_rank] = node_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="attention",
                    stage="attention",
                    predecessors=(),
                    binding=binding,
                    layer_type=layer.layer_type,
                    layer_id=layer_id,
                    token_count=block.token_count,
                    original_token_count=block.token_count,
                    output_token_count=block.token_count,
                )
            )
        a_collective = self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:attention:collective",
            domain=a_domain,
            dp_rank=block.placement.a_dp_rank,
            bindings=a_bindings,
            producer_ids=attention_ids,
            after_stage="a_compute",
            token_count=block.token_count,
            layer_type=layer.layer_type,
            required=len(a_bindings) > 1,
            producer_layout="partial",
            required_target_layout=(
                "replicated" if layer.layer_type == "moe" else None
            ),
        )

        # DeepSeek-V3 applies one RMSNorm after attention and before the AF
        # transfer/router.  Keep it as an explicit A-side node so its service
        # time contends for the same worker and appears in the trace.
        post_attention_ids: dict[int, str] = {}
        post_attention_predecessors = (
            (a_collective.node_id,)
            if a_collective is not None
            else tuple(attention_ids.values())
        )
        for binding in a_bindings:
            node_id = f"{prefix}:post_attention_norm:r{binding.domain_rank}"
            post_attention_ids[binding.domain_rank] = node_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="post_attention_norm",
                    stage="layout",
                    predecessors=post_attention_predecessors,
                    binding=binding,
                    layer_type=layer.layer_type,
                    layer_id=layer_id,
                    token_count=block.token_count,
                    original_token_count=block.token_count,
                    output_token_count=block.token_count,
                    shape={
                        "tokens": block.token_count,
                        "hidden_size": self.hidden_size,
                    },
                )
            )

        if layer.layer_type == "dense":
            dag = self._build_runtime_dense_tail(
                block=block,
                layer_id=layer_id,
                layer_type=layer.layer_type,
                link=link,
                prefix=prefix,
                nodes=nodes,
                a_domain=a_domain,
                f_domain=f_domain,
                a_bindings=a_bindings,
                f_bindings=f_base_bindings,
                attention_ids=post_attention_ids,
                a_collective=a_collective,
            )
        else:
            dag = self._build_runtime_moe_tail(
                block=block,
                layer_id=layer_id,
                layer=layer,
                snapshot=snapshot,
                link=link,
                prefix=prefix,
                nodes=nodes,
                a_domain=a_domain,
                f_domain=f_domain,
                a_bindings=a_bindings,
                f_service_bindings=f_service_bindings,
                attention_ids=post_attention_ids,
                a_collective=a_collective,
            )
        for node in dag.nodes:
            self.runtime_plan.validate_execution_node(node)
        return dag

    def _build_runtime_dense_tail(
        self,
        *,
        block: AFBlock,
        layer_id: int,
        layer_type: str,
        link: AFLinkConfig,
        prefix: str,
        nodes: list[ExecutionNode],
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
        a_bindings: tuple[Any, ...],
        f_bindings: tuple[Any, ...],
        attention_ids: Mapping[int, str],
        a_collective: ExecutionNode | None,
    ) -> AFExecutionDAG:
        tokens = self._block_tokens(block)
        dense_ids = {
            binding.domain_rank: f"{prefix}:dense:r{binding.domain_rank}"
            for binding in f_bindings
        }
        a_to_f_copies = self._activation_copies(
            tokens=tokens,
            a_bindings=a_bindings,
            target_specs=tuple(
                (binding, dense_ids[binding.domain_rank], tokens)
                for binding in f_bindings
            ),
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
            copies=a_to_f_copies,
            link=link,
            layer_type=layer_type,
            layer_id=layer_id,
        )

        dense_compute_ids: dict[int, str] = {}
        for binding in f_bindings:
            node_id = dense_ids[binding.domain_rank]
            dense_compute_ids[binding.domain_rank] = node_id
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="dense_ffn",
                    stage="dense_ffn",
                    predecessors=arrivals[node_id],
                    binding=binding,
                    layer_type=layer_type,
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    output_token_count=len(tokens),
                )
            )
        f_collective = self._append_runtime_collective(
            nodes,
            node_id=f"{prefix}:dense:collective",
            domain=f_domain,
            dp_rank=block.placement.f_dp_rank,
            bindings=f_bindings,
            producer_ids=dense_compute_ids,
            after_stage="expert_compute",
            token_count=len(tokens),
            layer_type=layer_type,
            required=len(f_bindings) > 1,
            producer_layout="partial",
        )

        residual_ids = {
            binding.domain_rank: f"{prefix}:residual:r{binding.domain_rank}"
            for binding in a_bindings
        }
        f_to_a_copies = self._output_copies(
            items=tuple(
                (token, f"request:{token[0]}:token:{token[1]}:dense", None, None)
                for token in tokens
            ),
            source_bindings=f_bindings,
            producer_ids=dense_compute_ids,
            collective=f_collective,
            target_bindings=a_bindings,
            consumer_ids=residual_ids,
            tensor_layout="replicated",
            transfer_dtype=self.runtime_plan.dtypes_for_link(link.name).f_to_a_dtype,
        )
        returned = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:f_to_a",
            operation="af_f_to_a",
            direction="f_to_a",
            copies=f_to_a_copies,
            link=link,
            layer_type=layer_type,
            layer_id=layer_id,
        )
        residual_nodes: dict[int, str] = {}
        for binding in a_bindings:
            node_id = residual_ids[binding.domain_rank]
            residual_nodes[binding.domain_rank] = node_id
            slices = tuple(
                copy.tensor_slice
                for copy in f_to_a_copies
                if copy.target_binding.domain_rank == binding.domain_rank
            )
            nodes.append(
                self._runtime_compute_node(
                    node_id=node_id,
                    operation="residual",
                    stage="residual",
                    predecessors=returned[node_id],
                    binding=binding,
                    layer_type=layer_type,
                    layer_id=layer_id,
                    token_count=len(tokens),
                    original_token_count=len(tokens),
                    output_token_count=len(tokens),
                    tensor_slices=slices,
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
            producer_ids=residual_nodes,
            after_stage="residual",
            token_count=len(tokens),
            layer_type=layer_type,
            required=len(a_bindings) > 1,
            producer_layout="sharded",
            required_target_layout="replicated",
        )
        return AFExecutionDAG(tuple(nodes))
