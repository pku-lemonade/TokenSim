"""Per-invocation readiness and communication execution."""

from __future__ import annotations

from dataclasses import dataclass

import simpy

from TokenSim.af.caching.estimates import prepare_query
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.af.providers.queries import AFTransferQuery
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError, SimulationStateError


@dataclass(frozen=True)
class TransferReadiness:
    """Events completing when each stream/input and the receive call are ready.

    Callers create these events for the current invocation. A host API return
    alone does not make a CUDA stream ready. Omitted events mean readiness is
    already covered by the DAG; this assumption is exported in the trace.
    """

    sender_stream: simpy.Event | None = None
    receiver_stream: simpy.Event | None = None
    receiver_posted: simpy.Event | None = None

    def events(self, env):
        result = []
        for event in (self.sender_stream, self.receiver_stream, self.receiver_posted):
            if event is None:
                continue
            if not isinstance(event, simpy.Event) or event.env is not env:
                raise SimulationStateError(
                    "transfer readiness must use events from the current simulation"
                )
            if event not in result:
                result.append(event)
        return tuple(result)


class ExecutorCommunicationMixin:
    def _run_planned_communication(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        step: ExecutionNode,
        transport_completed=None,
        input_readiness=None,
    ):
        is_control = step.metadata.get("message_class") == "control"
        binding_key = (
            link.name,
            step.direction,
            step.resource_group,
            step.resource_lane,
            (
                step.physical_workers
                if self.cluster_config.af.transfer_topology_mode == "physical"
                or self.resources.endpoint_resources is not None
                else ()
            ),
            step.network,
        )
        binding = self._communication_resource_bindings.get(binding_key)
        if binding is None:
            keyed_resources = self.resources.communication_resources_for_step(
                link,
                step.direction,
                step.resource_group,
                step.resource_lane,
                step.physical_workers,
                network=step.network,
            )
            resource_ids = tuple(
                self.resources.format_link_resource(key) for key, _ in keyed_resources
            )
            binding = (keyed_resources, resource_ids)
            self._communication_resource_bindings[binding_key] = binding
        keyed_resources, resource_ids = binding
        input_predecessors = (
            tuple(producer for producer, _ in input_readiness[0].values())
            if input_readiness
            else ()
        )
        if step.operation == "all_to_all" and (
            step.payload_bytes == 0 or len(step.physical_workers) == 1
        ):
            keyed_resources, resource_ids = (), ()
        endpoint_workers = self._endpoint_worker_bindings.get(step.physical_workers)
        if endpoint_workers is None:
            endpoint_workers = tuple(
                self.resources.worker(self.resources.worker_id_for_physical(identity))
                for identity in step.physical_workers
            )
            self._endpoint_worker_bindings[step.physical_workers] = endpoint_workers
        source_device = endpoint_workers[0].hardware
        target_device = endpoint_workers[-1].hardware
        effective_dtypes = self._effective_dtypes(block.link_name)
        transfer_dtype = (
            effective_dtypes["a_to_f_dtype"]
            if step.direction == "a_to_f"
            else (
                effective_dtypes["f_to_a_dtype"]
                if step.direction == "f_to_a"
                else effective_dtypes["activation_dtype"]
            )
        )
        if is_control:
            transfer_dtype = step.metadata["wire_dtype"]
        is_collective = step.operation in {
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "tp_collective",
        }
        query = prepare_query(
            self.transfer_provider,
            AFTransferQuery,
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            direction=step.direction,
            token_count=block.token_count,
            hidden_size=self.hidden_size,
            dtype=transfer_dtype,
            source_device=source_device,
            target_device=target_device,
            network=step.network or link.network,
            operation=step.operation,
            transport_backend=(
                step.metadata["transport_backend"]
                if is_control
                else (
                    "pynccl"
                    if self.cluster_config.af.execution_protocol == "vllm_afd"
                    else "unspecified"
                )
            ),
            link_group=link.resource_group,
            group_size=max(1, len(step.domain_ranks)),
            payload_bytes=step.payload_bytes,
            source_domain=step.source_domain,
            target_domain=step.target_domain,
            logical_participants=step.logical_participants,
            participant_ranks=step.domain_ranks,
            physical_endpoints=step.physical_workers,
            resource_identities=resource_ids,
            topology=self._query_mapping(
                (
                    ("participant_count", len(step.domain_ranks)),
                    ("physical_endpoint_count", len(step.physical_workers)),
                )
            ),
            node_id=step.node_id,
            communication_kind=("collective" if is_collective else "endpoint_leg"),
            layer_type=step.layer_type,
            original_token_count=step.metadata.get(
                "original_token_count", block.token_count
            ),
            assignment_count=step.metadata.get("assignment_count"),
            weight_dtype=effective_dtypes["weight_dtype"],
            activation_dtype=effective_dtypes["activation_dtype"],
            kv_cache_dtype=effective_dtypes["kv_cache_dtype"],
            a_to_f_dtype=effective_dtypes["a_to_f_dtype"],
            f_to_a_dtype=effective_dtypes["f_to_a_dtype"],
            logical_stage_id=step.logical_stage_id,
            leg_id=step.leg_id,
            source_endpoint=step.source_endpoint,
            target_endpoint=step.target_endpoint,
            participant_endpoints=step.participant_endpoints,
            tensor_slices=step.tensor_slices,
            source_layout=step.source_layout,
            target_layout=step.target_layout,
            fabric=step.resource_group,
            resource_group=step.resource_group,
            resource_lane=step.resource_lane,
            logical_input_bytes_per_rank=step.logical_input_bytes_per_rank,
            logical_output_bytes_per_rank=step.logical_output_bytes_per_rank,
            routing_edges=step.routing_edges,
        )
        timing = self.cluster_config.af.communication_timing
        estimate = self._estimate(
            self.transfer_provider, query, block, layer_id, step.operation
        )
        if (
            timing.enabled
            and estimate.details.get("timing_scope") != "transport_service"
        ):
            raise SimulationStateError(
                "Separated communication requires timing_scope='transport_service'; "
                "legacy or eager intervals may include host gaps and queueing"
            )

        communication_ready_at = self.env.now
        wire_node_id = step.node_id
        wire_predecessors = step.predecessor_ids
        if input_readiness is not None:
            input_ids = {producer for producer, _ in input_readiness[0].values()}
            wire_predecessors = tuple(
                key for key in wire_predecessors if key not in input_ids
            )
        submission_finished_at = self.env.now
        receiver_process = None
        native_state = self._native_controls.get(block.iteration_id, {})
        native_call = native_state.get("calls", {}).get(step.node_id)
        receiver_id = f"{step.node_id}:receive_submit"
        needs_receiver = (
            timing.enabled
            and step.direction in {"a_to_f", "f_to_a"}
            and (
                timing.host_receive_submit_us
                or timing.a_step_prepare_us
                or timing.f_step_prepare_us
                or timing.native_control_enabled
            )
        )
        if native_call is not None:
            receiver_process = native_call["receiver"]
        elif needs_receiver:
            receive_cost_field = self._native_call_cost_field(step, "receive")
            receiver_process = self.env.process(
                self._run_receiver_submit(
                    block,
                    layer_id,
                    step,
                    receiver_id,
                    self._host_cost_us(receive_cost_field, block) / 1e6,
                    receive_cost_field,
                )
            )
        if timing.enabled:
            submit_node_id = f"{step.node_id}:host_submit"
            is_collective = step.operation in {
                "tp_collective",
                "all_reduce",
                "all_gather",
                "reduce_scatter",
            }
            submitters = (
                step.physical_workers if is_collective else (step.physical_workers[0],)
            )
            try:
                send_cost_field = self._native_call_cost_field(step, "send")
                yield from self._run_communication_host_stage(
                    block,
                    layer_id,
                    step,
                    submit_node_id,
                    tuple(
                        key
                        for key in step.predecessor_ids
                        if input_readiness is None
                        or key
                        not in {producer for producer, _ in input_readiness[0].values()}
                    ),
                    submitters,
                    self._host_cost_us(send_cost_field, block) / 1e6,
                    "host_submit",
                    details={"cost_field": send_cost_field},
                )
            except BaseException:
                if receiver_process is not None and receiver_process.is_alive:
                    receiver_process.interrupt("AF sender submission cancelled")
                    yield receiver_process
                raise
            submission_finished_at = self.env.now
            if native_call is not None:
                native_call["submitted"].succeed()
                if step.operation == "af_control_payload":
                    native_state["a_ready"].succeed()
            wire_node_id = f"{step.node_id}:transport"
            wire_predecessors = (submit_node_id,)

        readiness_mode = "dag_dependencies_only"
        peer_wait_started_at = self.env.now
        if receiver_process is not None:
            try:
                receiver_result = yield receiver_process
                if isinstance(receiver_result, Exception):
                    raise receiver_result
            except BaseException:
                if receiver_process.is_alive:
                    receiver_process.interrupt("AF receive submission cancelled")
                    yield receiver_process
                raise
            wire_predecessors = (*wire_predecessors, receiver_id)
            readiness_mode = (
                "native_preposted_call_order"
                if native_call is not None
                else "fixed_host_preparation"
            )
        if self.transfer_readiness is not None:
            readiness = self.transfer_readiness(block, layer_id, step)
            if not isinstance(readiness, TransferReadiness):
                raise SimulationStateError(
                    "transfer_readiness must return TransferReadiness"
                )
            readiness_mode = "invocation_events"
            readiness_events = readiness.events(self.env)
            if readiness_events:
                yield self.env.all_of(readiness_events)
        peer_ready_at = self.env.now

        multiple_phases = False
        if self.cluster_config.af.transfer_topology_mode == "physical":
            if hasattr(self.transfer_provider, "communication_plan"):
                plan = self.transfer_provider.communication_plan(query)
                if self.resources.node_by_worker != dict(
                    self.transfer_provider.node_by_worker
                ):
                    raise ConfigurationError(
                        "provider and executor use different communication domain placement"
                    )
                multiple_phases = len(plan.phases) > 1
                ready_at, start_at, end_at = yield from self._run_domain_plan(
                    block,
                    layer_id,
                    step,
                    wire_node_id,
                    wire_predecessors,
                    plan,
                    input_readiness,
                )
                resource_ids = tuple(
                    sorted({key for phase in plan.phases for key in phase.resource_ids})
                )
            else:
                ready_at, start_at, end_at = (
                    yield from self.resources.domain_scheduler.run(
                        resource_ids,
                        estimate.service_duration,
                        (block.iteration_id, block.block_id, layer_id, wire_node_id),
                    )
                )
        else:
            ready_at, start_at, end_at, estimate = yield from self._run_flat_transport(
                block, layer_id, wire_node_id, keyed_resources, resource_ids, estimate
            )
        if (
            estimate.payload_bytes is not None
            and step.payload_bytes is not None
            and estimate.payload_bytes != step.payload_bytes
        ):
            raise SimulationStateError(
                f"AF provider payload mismatch for node={step.node_id}: "
                f"planned={step.payload_bytes}, returned={estimate.payload_bytes}"
            )
        if self.trace.enabled:
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type=(
                        "synchronize"
                        if multiple_phases
                        else (
                            "control_transfer"
                            if is_control
                            else (
                                "collective"
                                if step.operation
                                in {
                                    "tp_collective",
                                    "all_reduce",
                                    "all_gather",
                                    "reduce_scatter",
                                }
                                else "transfer"
                            )
                        )
                    ),
                    operation=step.operation,
                    direction=step.direction,
                    resource_id=(
                        f"completion:{wire_node_id}"
                        if multiple_phases or not resource_ids
                        else resource_ids[0]
                    ),
                    resource_ids=() if multiple_phases else resource_ids,
                    ready_at=end_at if multiple_phases else ready_at,
                    start_at=end_at if multiple_phases else start_at,
                    end_at=end_at,
                    queue_duration=0.0 if multiple_phases else start_at - ready_at,
                    service_duration=0.0 if multiple_phases else end_at - start_at,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    match_type=estimate.match_type,
                    payload_bytes=step.payload_bytes,
                    details={
                        **estimate.details,
                        "communication_operation": step.operation,
                        "phase_completion_only": multiple_phases,
                        "message_class": "control" if is_control else "data",
                        "transport_backend": query.transport_backend,
                        "wire_dtype": transfer_dtype,
                        **(
                            {
                                "control_schema": step.metadata["control_schema"],
                                "payload_utf8": step.metadata.get("payload_utf8"),
                            }
                            if is_control
                            else {}
                        ),
                        "execution_protocol": self.cluster_config.af.execution_protocol,
                        "readiness_mode": readiness_mode,
                        "readiness_wait_started_at": peer_wait_started_at,
                        "peer_ready_at": peer_ready_at,
                        "readiness_wait_duration": peer_ready_at - peer_wait_started_at,
                        **(
                            {
                                "communication_ready_at": communication_ready_at,
                                "host_submit_finished_at": submission_finished_at,
                                "communication_timing_mode": "separated",
                            }
                            if timing.enabled
                            else {}
                        ),
                    },
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    logical_participants=step.logical_participants,
                    participant_ranks=step.domain_ranks,
                    physical_endpoints=step.physical_workers,
                    node_id=wire_node_id,
                    predecessor_ids=(
                        tuple(
                            f"{wire_node_id}:phase:{phase.phase_id}"
                            for phase in plan.phases
                        )
                        if multiple_phases
                        else tuple(
                            dict.fromkeys((*wire_predecessors, *input_predecessors))
                        )
                    ),
                    dp_rank=step.dp_rank,
                    tp_rank=step.tp_rank,
                    ep_rank=step.ep_rank,
                    tp_group=step.tp_group,
                    ep_group=step.ep_group,
                    logical_stage_id=step.logical_stage_id,
                    layer_type=step.layer_type,
                    tensor_slices=step.tensor_slices,
                    source_endpoint=step.source_endpoint,
                    target_endpoint=step.target_endpoint,
                    source_layout=step.source_layout,
                    target_layout=step.target_layout,
                    routing_edges=step.routing_edges,
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, wire_node_id), None)
        if transport_completed is not None and not transport_completed.triggered:
            transport_completed.succeed()
        if timing.enabled:
            is_collective = step.operation in {
                "tp_collective",
                "all_reduce",
                "all_gather",
                "reduce_scatter",
            }
            completers = (
                step.physical_workers if is_collective else (step.physical_workers[-1],)
            )
            yield from self._run_communication_host_stage(
                block,
                layer_id,
                step,
                step.node_id,
                (wire_node_id,),
                completers,
                (
                    0.0
                    if is_control
                    else self._host_cost_us("host_completion_us", block) / 1e6
                ),
                "synchronize",
                details=(
                    {
                        "cost_field": None if is_control else "host_completion_us",
                        "communication_ready_at": communication_ready_at,
                        "submission_finished_at": submission_finished_at,
                        "transfer_finished_at": end_at,
                        "dependency_wait_duration": end_at - submission_finished_at,
                        "wait_is_dependency_not_additional_service": True,
                    }
                    if self.trace.enabled
                    else None
                ),
            )
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)
