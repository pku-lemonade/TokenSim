"""Pipeline-stage activation handoffs on the production AF resources."""

from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.af.providers.queries import AFEndpoint, AFTensorSlice
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


class PipelineExecutionMixin:
    def _pipeline_handoff(self, block, layer_id):
        if layer_id == 0:
            return None
        domain = block.placement.a_domain
        previous = self.runtime_plan.pipeline_stage(domain, layer_id - 1)
        stage = self.runtime_plan.pipeline_stage(domain, layer_id)
        if stage == previous:
            return None
        sources = self.runtime_plan.bindings_for_layer(
            domain, block.placement.a_instance, layer_id - 1
        )
        targets = self.runtime_plan.bindings_for_layer(
            domain, block.placement.a_instance, layer_id
        )
        plan = self.runtime_plan.collective_plan(
            domain, block.placement.a_dp_rank, "residual", previous
        )
        if plan is None:
            plan = self.runtime_plan.collective_plan(
                domain, block.placement.a_dp_rank, "a_compute", previous
            )
        if plan is None:
            raise ConfigurationError(
                "pipeline handoff requires a declared A communication fabric"
            )
        width = self.hidden_size
        size = dtype_size_bytes(self.dtype, role="activation_dtype")
        nodes = []
        for source, target in zip(sources, targets, strict=True):
            source_endpoint = AFEndpoint(
                source.domain,
                source.domain_rank,
                source.physical_identity,
                source.hardware,
                source.dp_rank,
                source.tp_rank,
                source.ep_rank,
            )
            target_endpoint = AFEndpoint(
                target.domain,
                target.domain_rank,
                target.physical_identity,
                target.hardware,
                target.dp_rank,
                target.tp_rank,
                target.ep_rank,
            )
            node_id = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}:pipeline:r{source.tp_rank}"
            slices = tuple(
                AFTensorSlice(
                    tensor_item_id=f"pipeline:request:{request}:token:{offset}",
                    interval=(0, width),
                    layout="replicated",
                    owner_domain=source.domain,
                    owner_domain_rank=source.domain_rank,
                    owner_dp_rank=source.dp_rank,
                    owner_tp_rank=source.tp_rank,
                    owner_physical_worker=source.physical_identity,
                    canonical_producer=node_id,
                    element_count=width,
                    payload_bytes=width * size,
                )
                for request, query, start in zip(
                    block.request_ids,
                    block.query_lengths or (1,) * len(block.request_ids),
                    block.query_offsets or (0,) * len(block.request_ids),
                )
                for offset in range(start, start + query)
            )
            nodes.append(
                ExecutionNode(
                    node_id=node_id,
                    operation="af_pipeline_handoff",
                    kind="communication",
                    predecessor_ids=(),
                    stage=None,
                    direction="a_to_a",
                    source_domain=domain,
                    target_domain=domain,
                    logical_participants=(source.instance, target.instance),
                    domain_ranks=(source.domain_rank, target.domain_rank),
                    physical_workers=(
                        source.physical_identity,
                        target.physical_identity,
                    ),
                    dp_rank=block.placement.a_dp_rank,
                    network=plan.network,
                    resource_group=plan.resource_group,
                    resource_lane=plan.resource_lane,
                    payload_bytes=block.token_count * width * size,
                    source_endpoint=source_endpoint,
                    target_endpoint=target_endpoint,
                    participant_endpoints=(source_endpoint, target_endpoint),
                    source_layout="replicated",
                    target_layout="replicated",
                    tensor_slices=slices,
                    logical_stage_id=f"i{block.iteration_id}:b{block.block_id}:l{layer_id}:pipeline",
                    metadata={"source_pp_rank": previous, "target_pp_rank": stage},
                )
            )
        return AFExecutionDAG(tuple(nodes))
