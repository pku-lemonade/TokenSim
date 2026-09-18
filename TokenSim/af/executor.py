from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import simpy

from TokenSim.af.builder import AFBlockBuilder, phase_for_requests
from TokenSim.af.latency import (
    AFComputeLatencyProvider,
    AFComputeQuery,
    AFServiceEstimate,
    AFTransferLatencyProvider,
    AFTransferQuery,
)
from TokenSim.af.models import AFBlock, AFIteration, AFPhase
from TokenSim.af.planning import (
    AFExecutionDAG,
    ExecutionNode,
    IndependentRoundRobinAFReplicaSelector,
    ParallelExecutionPlanner,
)
from TokenSim.af.resources import AFResourceRegistry
from TokenSim.af.runtime import AFRuntimePlan
from TokenSim.af.trace import AFTraceEvent, AFTraceRecorder
from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
from TokenSim.dtypes import canonicalize_dtype, dtype_size_bytes
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.moe.routing import ExpertRouting


class AFExecutor:
    def __init__(
        self,
        env: simpy.Environment,
        cluster_config: ClusterConfig,
        worker_configs: list[WorkerConfig],
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        model_id: str,
        dtype: str,
        total_layers: int,
        hidden_size: int,
        resources: AFResourceRegistry | None = None,
        trace_recorder: AFTraceRecorder | None = None,
        moe_config: Any | None = None,
        random_seed: int = 0,
        runtime_plan: AFRuntimePlan | None = None,
    ) -> None:
        self._validate_runtime(
            cluster_config,
            compute_provider,
            transfer_provider,
            model_id,
            dtype,
            total_layers,
            hidden_size,
        )
        self.env = env
        self.cluster_config = cluster_config
        self.compute_provider = compute_provider
        self.transfer_provider = transfer_provider
        self.model_id = model_id
        try:
            legacy_dtype = canonicalize_dtype(dtype, role="activation_dtype")
        except ValueError as exc:
            raise ConfigurationError(f"invalid AF executor dtype: {exc}") from exc
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.runtime_plan = runtime_plan
        self.dtype = (
            runtime_plan.effective_dtypes.activation_dtype
            if runtime_plan is not None
            else legacy_dtype
        )
        self._validate_provider_dtype_contract()
        self.resources = resources or AFResourceRegistry(
            env, cluster_config, worker_configs
        )
        self.trace = trace_recorder or AFTraceRecorder(
            cluster_config.af.trace_enabled,
            self.resources.resource_capacities(),
        )
        self.builder = AFBlockBuilder()
        self.replica_selector = IndependentRoundRobinAFReplicaSelector()
        self.expert_routing = (
            ExpertRouting(moe_config, seed=random_seed)
            if moe_config is not None and moe_config.enabled
            else None
        )
        self.plan_builder = ParallelExecutionPlanner(
            cluster_config,
            hidden_size=hidden_size,
            dtype_size_bytes=dtype_size_bytes(
                self.dtype,
                role="activation_dtype",
            ),
            num_experts=(moe_config.num_experts if moe_config is not None else 0),
            top_k=(moe_config.num_experts_per_tok if moe_config is not None else 1),
            runtime_plan=runtime_plan,
        )
        self._links = {link.name: link for link in cluster_config.af.links}
        self._next_iteration_id = 0
        self.active_iterations: dict[int, AFIteration] = {}
        self.active_stages: dict[tuple[int, int, str], dict[str, object]] = {}

    @staticmethod
    def _validate_runtime(
        cluster_config: ClusterConfig,
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        model_id: str,
        dtype: str,
        total_layers: int,
        hidden_size: int,
    ) -> None:
        if not cluster_config.af.enabled:
            raise ConfigurationError("AFExecutor requires af.enabled=true")
        missing = []
        if compute_provider is None:
            missing.append("compute latency provider")
        if transfer_provider is None:
            missing.append("transfer latency provider")
        if not isinstance(model_id, str) or not model_id:
            missing.append("model_id")
        if not isinstance(dtype, str) or not dtype:
            missing.append("dtype")
        if type(total_layers) is not int or total_layers < 1:
            missing.append("positive total_layers")
        if type(hidden_size) is not int or hidden_size < 1:
            missing.append("positive hidden_size")
        if missing:
            raise ConfigurationError(
                "AF runtime is missing required dependencies: " + ", ".join(missing)
            )

    def submit(
        self,
        source_a_worker_id: int,
        requests: Sequence[Any],
        phase: AFPhase | None = None,
    ) -> AFIteration:
        resolved_phase = phase or phase_for_requests(requests)
        if resolved_phase not in self.cluster_config.af.phases:
            raise ConfigurationError(
                f"AF phase {resolved_phase!r} is not selected by af.phases"
            )
        link = self.cluster_config.phase_to_af_link[resolved_phase]
        source_group = self.resources.group_for_worker(source_a_worker_id)
        if source_group != link.attention_group:
            raise ConfigurationError(
                f"AF phase {resolved_phase!r} requires attention group "
                f"{link.attention_group!r}, but worker {source_a_worker_id} belongs to "
                f"{source_group!r}"
            )
        source_worker = self.resources.worker(source_a_worker_id)
        if source_worker.physical_worker_ref is None:
            raise ConfigurationError(
                f"AF source worker {source_a_worker_id} has no physical identity"
            )
        source_identity = source_worker.physical_worker_ref.identity
        a_domain, f_domain = self.cluster_config.phase_to_af_domains[resolved_phase]
        a_instance = self.plan_builder.instance_for_physical(a_domain, source_identity)
        owner_key = a_domain.name
        for request in requests:
            owners = getattr(request, "af_a_owners", None)
            if owners is None:
                owners = {}
                request.af_a_owners = owners
            previous = owners.get(owner_key)
            if previous is not None and previous != a_instance.name:
                raise ConfigurationError(
                    f"request {request.id} AF {resolved_phase} owner is "
                    f"{previous!r}, cannot move to {a_instance.name!r} without KV transfer"
                )
            owners[owner_key] = a_instance.name
        iteration_id = self._next_iteration_id
        self._next_iteration_id += 1
        runtime_layer_types = self._runtime_layer_types()
        formal_routing = (
            self.runtime_plan is not None and resolved_phase == "decode"
        )
        layer_types = runtime_layer_types if formal_routing else None

        def placement_factory(_block_id, _block_requests):
            selection = self.replica_selector.select_for_preselected_a(
                resolved_phase,
                a_domain,
                f_domain,
                a_instance,
            )
            placement = self.plan_builder.placement(
                resolved_phase, source_identity, selection.f_service
            )
            if placement.a_instance != selection.a_owner.name:
                raise ConfigurationError(
                    "AF scheduler owner changed while selecting the F service "
                    f"replica: expected={selection.a_owner.name!r}, "
                    f"actual={placement.a_instance!r}"
                )
            return placement

        def histogram_factory(block_requests):
            if self.expert_routing is None:
                return {}
            return self.expert_routing.histogram_for_step(
                list(block_requests),
                phase=resolved_phase,
                decode_step=self._routing_step(resolved_phase, block_requests),
            )

        def routing_snapshot_factory(
            snapshot_iteration_id,
            snapshot_phase,
            layer_id,
            block_requests,
            placement,
        ):
            if self.expert_routing is None:
                raise ConfigurationError(
                    f"AF runtime layer {layer_id} is MoE but no ExpertRouting "
                    "configuration is available"
                )
            if placement is None:
                raise ConfigurationError(
                    f"AF runtime layer {layer_id} cannot freeze routing without "
                    "block placement"
                )
            return self.expert_routing.snapshot_for_layer(
                block_requests,
                iteration_id=snapshot_iteration_id,
                phase=snapshot_phase,
                decode_step=self._routing_step(snapshot_phase, block_requests),
                layer_id=layer_id,
                a_dp_rank=placement.a_dp_rank,
                a_tp_rank=self._canonical_a_tp_rank(placement),
            )

        return self.builder.build(
            iteration_id=iteration_id,
            requests=requests,
            af_block_size=self.cluster_config.af.af_block_size,
            source_a_worker_id=source_a_worker_id,
            link=link,
            phase=resolved_phase,
            placement_factory=placement_factory,
            # Compatibility for isolated tests without a compiled runtime plan.
            # This branch is not the formal Decode execution contract.
            histogram_factory=None if formal_routing else histogram_factory,
            layer_types=layer_types,
            routing_snapshot_factory=(
                routing_snapshot_factory
                if formal_routing and "moe" in layer_types.values()
                else None
            ),
        )

    def _runtime_layer_types(self) -> dict[int, str] | None:
        if self.runtime_plan is None:
            return None
        layer_types = {
            layer.layer_id: layer.layer_type for layer in self.runtime_plan.layers
        }
        expected_ids = set(range(self.total_layers))
        if set(layer_types) != expected_ids:
            raise ConfigurationError(
                "AF runtime layer metadata must cover every executor layer once: "
                f"expected={sorted(expected_ids)}, actual={sorted(layer_types)}"
            )
        return layer_types

    @staticmethod
    def _routing_step(phase: AFPhase, requests: Sequence[Any]) -> int:
        if phase == "prefill":
            return 0
        steps = set()
        for request in requests:
            value = getattr(request, "generation_idx", None)
            if type(value) is not int or value < 0:
                raise ConfigurationError(
                    "Decode routing requires a non-negative generation_idx for "
                    f"request {getattr(request, 'id', '<unknown>')!r}"
                )
            steps.add(value)
        if len(steps) != 1:
            raise ConfigurationError(
                "one AF block cannot freeze a routing snapshot across different "
                f"Decode steps; got generation_idx values {sorted(steps)}"
            )
        return next(iter(steps))

    def _canonical_a_tp_rank(self, placement) -> int:
        if self.runtime_plan is None:  # pragma: no cover - formal routing has a plan
            return 0
        ownerships = self.runtime_plan.ownerships_for_replica(
            placement.a_domain,
            placement.a_dp_rank,
        )
        canonical = tuple(item for item in ownerships if item.canonical_producer)
        if len(canonical) != 1:
            raise ConfigurationError(
                "AF runtime A replica must have exactly one canonical routing "
                f"producer: domain={placement.a_domain!r}, "
                f"dp_rank={placement.a_dp_rank}, found={len(canonical)}"
            )
        return canonical[0].tp_rank

    def run_iteration(self, iteration: AFIteration):
        if iteration.iteration_id in self.active_iterations:
            raise SimulationStateError(
                f"AF iteration {iteration.iteration_id} is already active"
            )
        iteration.start_at = self.env.now
        self.active_iterations[iteration.iteration_id] = iteration
        failure_event = self.env.event()
        failures: list[BaseException] = []
        done_events = [self.env.event() for _ in iteration.blocks]
        processes = [
            self.env.process(
                self._run_block_guarded(block, failure_event, failures, done_event)
            )
            for block, done_event in zip(iteration.blocks, done_events)
        ]
        try:
            if processes:
                completion = self.env.all_of(done_events)
                yield self.env.any_of((completion, failure_event))
                if failures:
                    for process in processes:
                        if process.is_alive:
                            process.interrupt(
                                f"AF iteration {iteration.iteration_id} failed"
                            )
                    yield self.env.all_of(done_events)
                    exc = failures[0]
                    if isinstance(exc, SimulationStateError):
                        raise exc
                    raise SimulationStateError(
                        f"AF iteration {iteration.iteration_id} failed: {exc}"
                    ) from exc
        except simpy.Interrupt:
            for process in processes:
                if process.is_alive:
                    process.interrupt(
                        f"AF iteration {iteration.iteration_id} cancelled"
                    )
            if processes:
                yield self.env.all_of(done_events)
            raise
        finally:
            self.active_iterations.pop(iteration.iteration_id, None)
            for key in tuple(self.active_stages):
                if key[0] == iteration.iteration_id:
                    self.active_stages.pop(key, None)

        iteration.end_at = self.env.now
        iteration.barrier_wait = {
            block_id: iteration.end_at - completed_at
            for block_id, completed_at in iteration.block_completed_at.items()
        }
        if self.trace.enabled:
            self.trace.validate_invariants(iteration.iteration_id)
        return iteration

    def _run_block_guarded(
        self,
        block: AFBlock,
        failure_event: simpy.Event,
        failures: list[BaseException],
        done_event: simpy.Event,
    ):
        try:
            yield from self._run_block(block)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            failures.append(exc)
            if not failure_event.triggered:
                failure_event.succeed()
            return
        finally:
            if not done_event.triggered:
                done_event.succeed()

    def _run_block(self, block: AFBlock):
        link = self._links[block.link_name]
        for layer_id in range(self.total_layers):
            dag = self.plan_builder.build_layer(block, layer_id, link)
            yield from self._run_dag(block, layer_id, link, dag)
        iteration = self.active_iterations[block.iteration_id]
        iteration.block_completed_at[block.block_id] = self.env.now

    def _run_dag(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        dag: AFExecutionDAG,
    ):
        dag.validate_runtime_semantics()
        processes: dict[str, simpy.Process] = {}
        done_events: dict[str, simpy.Event] = {}
        failure_event = self.env.event()
        failures: list[BaseException] = []
        completed_nodes: set[str] = set()
        for node in dag.nodes:
            done_event = self.env.event()
            done_events[node.node_id] = done_event
            predecessors = tuple(done_events[item] for item in node.predecessor_ids)
            processes[node.node_id] = self.env.process(
                self._run_dag_node_guarded(
                    block,
                    layer_id,
                    link,
                    node,
                    predecessors,
                    failure_event,
                    failures,
                    done_event,
                    completed_nodes,
                )
            )
        terminal_done = self.env.all_of(
            tuple(done_events[node_id] for node_id in dag.terminal_node_ids)
        )
        yield self.env.any_of((terminal_done, failure_event))
        if failures:
            for process in processes.values():
                if process.is_alive:
                    process.interrupt(
                        f"AF DAG failed at iteration={block.iteration_id}, "
                        f"block={block.block_id}, layer={layer_id}"
                    )
            yield self.env.all_of(tuple(done_events.values()))
            raise failures[0]
        expected_nodes = {node.node_id for node in dag.nodes}
        if completed_nodes != expected_nodes:
            raise SimulationStateError(
                "AF DAG terminals completed without executing every planned node: "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, missing={sorted(expected_nodes - completed_nodes)}"
            )
        dag.validate_runtime_semantics()

    def _run_dag_node_guarded(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        node: ExecutionNode,
        predecessors: tuple[simpy.Event, ...],
        failure_event: simpy.Event,
        failures: list[BaseException],
        done_event: simpy.Event,
        completed_nodes: set[str],
    ):
        try:
            if predecessors:
                predecessor_done = self.env.all_of(predecessors)
                yield self.env.any_of((predecessor_done, failure_event))
                if failure_event.triggered:
                    return
            if failure_event.triggered:
                return
            yield from self._run_planned_step(block, layer_id, link, node)
            completed_nodes.add(node.node_id)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            error = SimulationStateError(
                f"AF DAG node failed for iteration={block.iteration_id}, "
                f"block={block.block_id}, layer={layer_id}, node={node.node_id}: {exc}"
            )
            error.__cause__ = exc
            failures.append(error)
            if not failure_event.triggered:
                failure_event.succeed()
        finally:
            if not done_event.triggered:
                done_event.succeed()

    def _run_dag_node(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        node: ExecutionNode,
        predecessors: tuple[simpy.Event, ...],
    ):
        if predecessors:
            yield self.env.all_of(predecessors)
        yield from self._run_planned_step(block, layer_id, link, node)

    def _run_planned_step(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        step: ExecutionNode,
    ):
        if step.kind == "compute":
            yield from self._run_planned_compute(block, layer_id, step)
        else:
            yield from self._run_planned_communication(block, layer_id, link, step)

    def _run_planned_compute(
        self,
        block: AFBlock,
        layer_id: int,
        step: ExecutionNode,
    ):
        if self.runtime_plan is not None:
            self.runtime_plan.validate_execution_node(step)
        resources = self.resources.compute_resources_for(step.physical_workers)
        worker_id = resources[0][0]
        worker = self.resources.worker(worker_id)
        local_work = step.local_work
        if local_work is None:
            raise SimulationStateError(f"compute node {step.node_id} has no local work")
        effective_dtypes = self._effective_dtypes(block.link_name)
        query = AFComputeQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
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
            local_shape=dict(local_work.tensor_shape),
            local_token_count=local_work.token_count,
            expert_ids=local_work.expert_ids,
            expert_workload=local_work.expert_workload,
            tp_group=step.tp_group,
            ep_group=step.ep_group,
            layer_type=step.layer_type,
            original_token_count=local_work.original_token_count,
            assignment_count=local_work.assignment_count,
            input_edge_count=local_work.input_edge_count,
            output_token_count=local_work.output_token_count,
            weight_dtype=effective_dtypes["weight_dtype"],
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
        )
        stage_name = step.stage
        ready_at, start_at, end_at, estimate = yield from self._run_resources_stage(
            block,
            layer_id,
            step.node_id,
            tuple(resource for _, resource in resources),
            lambda: self._estimate(
                self.compute_provider, query, block, layer_id, stage_name
            ),
        )
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
                token_count=local_work.token_count,
                source_id=estimate.source_id,
                dataset_version=estimate.dataset_version,
                match_type=estimate.match_type,
                details=estimate.details,
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
                local_work=local_work.as_dict(),
                layer_type=step.layer_type,
                original_token_count=local_work.original_token_count,
                assignment_count=local_work.assignment_count,
                input_edge_count=local_work.input_edge_count,
                output_token_count=local_work.output_token_count,
                logical_stage_id=step.logical_stage_id,
                source_endpoint=step.source_endpoint,
                target_endpoint=step.target_endpoint,
                participant_endpoints=step.participant_endpoints,
                tensor_slices=step.tensor_slices,
                source_layout=step.source_layout,
                target_layout=step.target_layout,
                effective_dtypes=effective_dtypes,
                routing_edges=local_work.routing_edges,
                source_dtype=local_work.source_dtype,
                target_dtype=local_work.target_dtype,
            )
        )
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)

    def _run_planned_communication(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        step: ExecutionNode,
    ):
        keyed_resources = self.resources.communication_resources_for_step(
            link,
            step.direction,
            step.resource_group,
            step.resource_lane,
        )
        resource_ids = tuple(
            self.resources.format_link_resource(key) for key, _ in keyed_resources
        )
        endpoint_workers = [
            self.resources.worker(self.resources.worker_id_for_physical(identity))
            for identity in step.physical_workers
        ]
        source_device = endpoint_workers[0].hardware
        target_device = endpoint_workers[-1].hardware
        effective_dtypes = self._effective_dtypes(block.link_name)
        is_collective = step.operation in {
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "tp_collective",
        }
        transfer_dtype = (
            effective_dtypes["a_to_f_dtype"]
            if step.direction == "a_to_f"
            else effective_dtypes["f_to_a_dtype"]
            if step.direction == "f_to_a"
            else effective_dtypes["activation_dtype"]
        )
        original_token_count = step.metadata.get("original_token_count")
        if original_token_count is None:
            original_token_count = block.token_count
        assignment_count = step.metadata.get("assignment_count")
        query = AFTransferQuery(
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
            link_group=self.cluster_config.af.operator_link_group,
            group_size=max(1, len(step.domain_ranks)),
            payload_bytes=step.payload_bytes,
            source_domain=step.source_domain,
            target_domain=step.target_domain,
            logical_participants=step.logical_participants,
            participant_ranks=step.domain_ranks,
            physical_endpoints=step.physical_workers,
            resource_identities=resource_ids,
            topology={
                "participant_count": len(step.domain_ranks),
                "physical_endpoint_count": len(step.physical_workers),
            },
            node_id=step.node_id,
            communication_kind="collective" if is_collective else "endpoint_leg",
            layer_type=step.layer_type,
            original_token_count=original_token_count,
            assignment_count=assignment_count,
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
        if timing.enabled and estimate.details.get("timing_scope") != "transport_service":
            raise SimulationStateError(
                "Separated communication requires timing_scope='transport_service'; "
                "legacy or eager intervals may include host gaps and queueing"
            )
        communication_ready_at = self.env.now
        wire_node_id = step.node_id
        wire_predecessors = step.predecessor_ids
        submission_finished_at = self.env.now
        if timing.enabled:
            submit_node_id = f"{step.node_id}:host_submit"
            participants = (step.physical_workers if is_collective
                            else (step.physical_workers[0],))
            yield from self._run_communication_host_stage(
                block, layer_id, step, submit_node_id, step.predecessor_ids,
                participants, timing.host_submit_us / 1e6, "host_submit",
            )
            submission_finished_at = self.env.now
            wire_node_id = f"{step.node_id}:transport"
            wire_predecessors = (submit_node_id,)
        ready_at, start_at, end_at, estimate = yield from self._run_resources_stage(
            block,
            layer_id,
            wire_node_id,
            tuple(resource for _, resource in keyed_resources),
            lambda: estimate,
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
        self.trace.record(
            AFTraceEvent(
                iteration_id=block.iteration_id,
                block_id=block.block_id,
                layer_id=layer_id,
                phase=block.phase,
                event_type=(
                    "collective" if is_collective else "transfer"
                ),
                operation=step.operation,
                direction=step.direction,
                resource_id=resource_ids[0],
                resource_ids=resource_ids,
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
                payload_bytes=step.payload_bytes,
                details={**estimate.details, **(dict(
                    communication_ready_at=communication_ready_at,
                    host_submit_finished_at=submission_finished_at,
                    communication_timing_mode="separated",
                ) if timing.enabled else {}), **(
                    {"payload_encoding": "top-k int32 expert IDs + float32 weights"}
                    if step.operation == "af_router_metadata" else {})},
                source_domain=step.source_domain,
                target_domain=step.target_domain,
                logical_participants=step.logical_participants,
                participant_ranks=step.domain_ranks,
                physical_endpoints=step.physical_workers,
                node_id=wire_node_id,
                predecessor_ids=wire_predecessors,
                dp_rank=step.dp_rank,
                tp_rank=step.tp_rank,
                ep_rank=step.ep_rank,
                tp_group=step.tp_group,
                ep_group=step.ep_group,
                layer_type=step.layer_type,
                original_token_count=original_token_count,
                assignment_count=assignment_count,
                logical_stage_id=step.logical_stage_id,
                leg_id=step.leg_id,
                source_endpoint=step.source_endpoint,
                target_endpoint=step.target_endpoint,
                participant_endpoints=step.participant_endpoints,
                tensor_slices=step.tensor_slices,
                source_layout=step.source_layout,
                target_layout=step.target_layout,
                effective_dtypes=effective_dtypes,
                dtype_role=(
                    None if step.operation == "af_router_metadata" else
                    "activation_dtype" if is_collective else f"{step.direction}_dtype"
                ),
                dtype=None if step.operation == "af_router_metadata" else transfer_dtype,
                bytes_per_element=None if step.operation == "af_router_metadata" else dtype_size_bytes(
                    transfer_dtype,
                    role=(
                        "activation_dtype"
                        if is_collective
                        else f"{step.direction}_dtype"
                    ),
                ),
                network=step.network,
                resource_group=step.resource_group,
                resource_lane=step.resource_lane,
                payload_breakdown=tuple(
                    step.metadata.get("payload_breakdown", ())
                ),
                routing_edges=step.routing_edges,
            )
        )
        self.active_stages.pop((block.iteration_id, block.block_id, wire_node_id), None)
        if timing.enabled:
            participants = (step.physical_workers if is_collective
                            else (step.physical_workers[-1],))
            yield from self._run_communication_host_stage(
                block, layer_id, step, step.node_id, (wire_node_id,),
                participants, timing.host_completion_us / 1e6, "synchronize",
                details=dict(
                    communication_ready_at=communication_ready_at,
                    submission_finished_at=submission_finished_at,
                    transfer_finished_at=end_at,
                    dependency_wait_duration=end_at - submission_finished_at,
                    wait_is_dependency_not_additional_service=True,
                ),
            )
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)

    def _run_communication_host_stage(
        self, block, layer_id, step, node_id, predecessors, participants,
        duration, event_type, details=None,
    ):
        """Acquire only a host slot; never hold the link while submitting/waiting."""
        keyed = self.resources.host_resources_for(participants)
        estimate = AFServiceEstimate(
            duration, "assumption:communication-host-cost",
            dataset_version="communication-timing-config",
            details=dict(communication_operation=step.operation,
                         timing_scope="host_service", fixed_cost_us=duration*1e6,
                         host_resource_scope=self.cluster_config.af.communication_timing.host_resource_scope,
                         **(details or {})),
        )
        ready, start, end, _ = yield from self._run_resources_stage(
            block, layer_id, node_id, tuple(resource for _, resource in keyed), lambda: estimate,
        )
        self.trace.record(AFTraceEvent(
            iteration_id=block.iteration_id, block_id=block.block_id, layer_id=layer_id,
            phase=block.phase, event_type=event_type, operation=event_type,
            direction=step.direction, resource_id=keyed[0][0],
            resource_ids=tuple(key for key, _ in keyed),
            ready_at=ready, start_at=start, end_at=end,
            queue_duration=start-ready, service_duration=end-start,
            request_count=block.request_count, token_count=block.token_count,
            source_id=estimate.source_id, dataset_version=estimate.dataset_version,
            details=estimate.details, node_id=node_id, predecessor_ids=predecessors,
            logical_stage_id=step.logical_stage_id, leg_id=step.leg_id,
            source_domain=step.source_domain, target_domain=step.target_domain,
            layer_type=step.layer_type, physical_endpoints=participants,
        ))
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)

    def _effective_dtypes(self, link_name: str) -> dict[str, str]:
        if self.runtime_plan is None:
            return {
                "weight_dtype": self.dtype,
                "activation_dtype": self.dtype,
                "kv_cache_dtype": self.dtype,
                "a_to_f_dtype": self.dtype,
                "f_to_a_dtype": self.dtype,
            }
        effective = self.runtime_plan.dtypes_for_link(link_name)
        return {
            "weight_dtype": effective.weight_dtype,
            "activation_dtype": effective.activation_dtype,
            "kv_cache_dtype": effective.kv_cache_dtype,
            "a_to_f_dtype": effective.a_to_f_dtype,
            "f_to_a_dtype": effective.f_to_a_dtype,
        }

    def _validate_provider_dtype_contract(self) -> None:
        if self.runtime_plan is None:
            return
        validator = getattr(
            self.compute_provider, "validate_effective_dtypes", None
        )
        if not callable(validator):
            return
        contracts = [self.runtime_plan.effective_dtypes]
        contracts.extend(
            effective
            for _, effective in self.runtime_plan.link_effective_dtypes
            if effective not in contracts
        )
        try:
            for effective in contracts:
                validator(self.model_id, effective)
        except ValueError as exc:
            raise ConfigurationError(
                f"AF provider dtype contract validation failed: {exc}"
            ) from exc

    def _run_resources_stage(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        resources: tuple[simpy.Resource, ...],
        estimate_factory,
    ):
        ready_at = self.env.now
        self._set_active_stage(block, layer_id, stage, ready_at)
        estimate = estimate_factory()
        requests = (
            [resource.request() for resource in resources]
            if estimate.service_duration > 0
            else []
        )
        try:
            if requests:
                yield self.env.all_of(requests)
            start_at = self.env.now
            self._set_active_stage(block, layer_id, stage, ready_at, start_at)
            yield self.env.timeout(estimate.service_duration)
            end_at = self.env.now
        finally:
            for resource, request in zip(resources, requests):
                if request.triggered:
                    resource.release(request)
                else:
                    request.cancel()
        return ready_at, start_at, end_at, estimate

    @staticmethod
    def _estimate(provider, query, block, layer_id, stage) -> AFServiceEstimate:
        try:
            estimate = provider.estimate(query)
        except Exception as exc:
            raise SimulationStateError(
                f"AF estimate unsupported for iteration={block.iteration_id}, "
                f"block={block.block_id}, layer={layer_id}, stage={stage}: {exc}"
            ) from exc
        if not isinstance(estimate, AFServiceEstimate):
            raise SimulationStateError(
                f"AF provider returned {type(estimate).__name__} for "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}; expected AFServiceEstimate"
            )
        duration = estimate.service_duration
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            valid = False
        else:
            valid = math.isfinite(duration) and duration >= 0
        if not valid:
            raise SimulationStateError(
                f"AF provider returned invalid service_duration={duration!r} for "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}"
            )
        if not isinstance(estimate.source_id, str) or not estimate.source_id:
            raise SimulationStateError(
                f"AF provider returned an empty source_id for iteration="
                f"{block.iteration_id}, block={block.block_id}, layer={layer_id}, "
                f"stage={stage}"
            )
        if (
            not isinstance(estimate.dataset_version, str)
            or not estimate.dataset_version
        ):
            raise SimulationStateError(
                f"AF provider returned an empty dataset_version for iteration="
                f"{block.iteration_id}, block={block.block_id}, layer={layer_id}, "
                f"stage={stage}"
            )
        if estimate.match_type not in {"exact", "interpolated"}:
            raise SimulationStateError(
                f"AF provider returned invalid match_type={estimate.match_type!r} "
                f"for iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}"
            )
        return estimate

    def _set_active_stage(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        ready_at: float,
        start_at: float | None = None,
    ) -> None:
        node_id = stage if ":" in stage else f"legacy:{stage}"
        display_stage = stage
        if ":" in stage:
            if ":router" in stage:
                display_stage = "router"
            elif ":expert:" in stage:
                display_stage = "expert"
            elif ":a:" in stage:
                display_stage = "a"
        self.active_stages[(block.iteration_id, block.block_id, node_id)] = {
            "iteration_id": block.iteration_id,
            "block_id": block.block_id,
            "layer_id": layer_id,
            "stage": display_stage,
            "node_id": node_id,
            "ready_at": ready_at,
            "start_at": start_at,
        }

    def unfinished_snapshot(self) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for iteration in self.active_iterations.values():
            snapshots.append(
                {
                    "iteration_id": iteration.iteration_id,
                    "phase": iteration.phase,
                    "block_ids": [block.block_id for block in iteration.blocks],
                    "completed_block_ids": sorted(iteration.block_completed_at),
                    "active_stages": [
                        dict(stage)
                        for key, stage in self.active_stages.items()
                        if key[0] == iteration.iteration_id
                    ],
                }
            )
        return snapshots
