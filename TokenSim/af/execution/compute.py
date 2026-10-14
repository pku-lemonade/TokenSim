"""Executor compute."""

from __future__ import annotations

from dataclasses import replace

from TokenSim.af.caching.estimates import prepare_query
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.af.providers.queries import AFComputeQuery, AFServiceEstimate
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.errors import SimulationStateError


class ExecutorComputeMixin:
    def _run_planned_compute(
        self,
        block: AFBlock,
        layer_id: int,
        step: ExecutionNode,
    ):
        preparation_ids = yield from self._wait_step_preparation(
            block, step.physical_workers
        )
        control = self._native_controls.get(block.iteration_id)
        if control is not None and not step.predecessor_ids:
            preparation_ids = (*preparation_ids, control["a_node"])
        timing = self.cluster_config.af.communication_timing
        compute_cost_field = self._native_call_cost_field(step, "compute")
        compute_submit_us = self._host_cost_us(compute_cost_field, block)
        if (
            step.operation in ("routed_expert", "f_weighted_combine")
            and self.overhead_profile is not None
            and "compute_f_moe_block_us" in self.overhead_profile.document["costs"]
        ):
            compute_submit_us = (
                0  # One native FusedMoE call owns router, experts and combine dispatch.
            )
        if (
            step.operation == "f_weighted_combine"
            and self.overhead_profile is not None
            and "compute_f_expert_group_us" in self.overhead_profile.document["costs"]
        ):
            compute_submit_us = (
                0  # Native grouped invocation includes combine dispatch.
            )
        combine_key = (block.block_id, layer_id, step.physical_workers)
        fused = (
            self._fused_combines.get(block.iteration_id, {}).get(combine_key)
            if step.stage == "weighted_combine"
            else None
        )
        if fused is not None:
            compute_submit_us = 0
        if timing.enabled and compute_submit_us:
            submit_id = f"{step.node_id}:compute_submit"
            yield from self._run_communication_host_stage(
                block,
                layer_id,
                step,
                submit_id,
                tuple(dict.fromkeys((*step.predecessor_ids, *preparation_ids))),
                step.physical_workers,
                compute_submit_us / 1e6,
                "host_submit",
                details={
                    "cost_field": compute_cost_field,
                    "cost_scope": (
                        "per_native_call_group"
                        if compute_cost_field != "compute_submit_us"
                        else "per_compute_dag_node"
                    ),
                    "calibrated": False,
                },
            )
            step = replace(step, predecessor_ids=(submit_id,))
        elif preparation_ids and not step.predecessor_ids:
            step = replace(step, predecessor_ids=preparation_ids)
        if (
            control is not None
            and "a_device_ready" in control
            and control["a_node"] in preparation_ids
        ):
            yield control["a_device_ready"]
            step = replace(
                step,
                predecessor_ids=tuple(
                    dict.fromkeys(
                        (
                            *step.predecessor_ids,
                            f"i{block.iteration_id}:control:payload:transport",
                        )
                    )
                ),
            )
        if self.runtime_plan is not None:
            self.runtime_plan.validate_execution_node(step)
        resources = self._compute_resource_bindings.get(step.physical_workers)
        if resources is None:
            resources = self.resources.compute_resources_for(step.physical_workers)
            self._compute_resource_bindings[step.physical_workers] = resources
        worker_id = resources[0][0]
        worker = self.resources.worker(worker_id)
        local_work = step.local_work
        if local_work is None:
            raise SimulationStateError(f"compute node {step.node_id} has no local work")
        effective_dtypes = self._effective_dtypes(block.link_name)
        query = prepare_query(
            self.compute_provider,
            AFComputeQuery,
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.forward_phase or block.phase,
            query_lengths=block.query_lengths,
            stage=step.stage,
            token_count=local_work.token_count,
            context_lengths=block.context_lengths,
            model_id=self.model_id,
            dtype=effective_dtypes["activation_dtype"],
            device_type=worker.hardware,
            worker_id=worker_id,
            operation=step.operation,
            source_domain=step.source_domain,
            dp_rank=step.dp_rank,
            tp_rank=step.tp_rank or 0,
            ep_rank=step.ep_rank or 0,
            logical_participant=step.logical_participants[0],
            physical_worker=step.physical_workers[0],
            local_shape=self._query_mapping(local_work.tensor_shape),
            local_token_count=local_work.token_count,
            expert_ids=local_work.expert_ids,
            expert_workload=local_work.expert_workload,
            tp_group=step.tp_group,
            ep_group=step.ep_group,
            layer_type=step.layer_type,
            original_token_count=local_work.original_token_count,
            input_edge_count=local_work.input_edge_count,
            output_token_count=local_work.output_token_count,
            weight_dtype=(
                self.runtime_plan.weight_dtype_for(step.stage, step.operation)
                if hasattr(self.runtime_plan, "weight_dtype_for")
                else effective_dtypes["weight_dtype"]
            ),
            activation_dtype=effective_dtypes["activation_dtype"],
            kv_cache_dtype=effective_dtypes["kv_cache_dtype"],
            a_to_f_dtype=effective_dtypes["a_to_f_dtype"],
            f_to_a_dtype=effective_dtypes["f_to_a_dtype"],
            logical_stage_id=step.logical_stage_id,
            tensor_slices=step.tensor_slices,
            owner_endpoint=local_work.owner_endpoint,
            routing_edges=local_work.routing_edges,
            source_dtype=local_work.source_dtype,
            target_dtype=local_work.target_dtype,
            assignment_count=local_work.assignment_count,
        )
        stage_name = step.stage
        ready_at, start_at, end_at, estimate = yield from self._run_resources_stage(
            block,
            layer_id,
            step.node_id,
            tuple(resource for _, resource in resources),
            lambda: (
                AFServiceEstimate(
                    0,
                    fused.source_id,
                    dataset_version=fused.dataset_version,
                    match_type=fused.match_type,
                    details={
                        "timing_scope": "device_service",
                        "covered_by": "routed_experts",
                        "semantic_completion_only": True,
                    },
                )
                if fused is not None
                else self._estimate(
                    self.compute_provider, query, block, layer_id, stage_name
                )
            ),
        )
        if (
            self.cluster_config.af.execution_protocol == "vllm_afd"
            and step.stage == "routed_expert"
            and "weighted_combine" in estimate.details.get("covered_operations", ())
        ):
            self._fused_combines.setdefault(block.iteration_id, {})[
                combine_key
            ] = estimate
        if self.trace.enabled:
            resource_ids = tuple(f"worker:{item[0]}" for item in resources)
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type="compute",
                    stage=step.stage,
                    operation=step.operation,
                    resource_id=resource_ids[0],
                    resource_ids=resource_ids,
                    worker_id=worker_id,
                    ready_at=ready_at,
                    start_at=start_at,
                    end_at=end_at,
                    queue_duration=start_at - ready_at,
                    service_duration=end_at - start_at,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    match_type=estimate.match_type,
                    details={
                        **estimate.details,
                        "execution_protocol": self.cluster_config.af.execution_protocol,
                        **(
                            {"routing_replay": self.routing_replay.describe()}
                            if self.routing_replay is not None
                            and step.stage == "router"
                            else {}
                        ),
                    },
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    logical_participants=step.logical_participants,
                    participant_ranks=step.domain_ranks,
                    physical_endpoints=step.physical_workers,
                    node_id=step.node_id,
                    predecessor_ids=step.predecessor_ids,
                    dp_rank=step.dp_rank,
                    tp_rank=step.tp_rank,
                    ep_rank=step.ep_rank,
                    tp_group=step.tp_group,
                    ep_group=step.ep_group,
                    local_work=local_work.trace_mapping(),
                    layer_type=step.layer_type,
                    original_token_count=local_work.original_token_count,
                    assignment_count=local_work.assignment_count,
                    input_edge_count=local_work.input_edge_count,
                    output_token_count=local_work.output_token_count,
                    logical_stage_id=step.logical_stage_id,
                    source_endpoint=step.source_endpoint,
                    target_endpoint=step.target_endpoint,
                    tensor_slices=step.tensor_slices,
                    source_layout=step.source_layout,
                    target_layout=step.target_layout,
                    routing_edges=step.routing_edges,
                    source_dtype=local_work.source_dtype,
                    target_dtype=local_work.target_dtype,
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)
