from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import simpy

from TokenSim.af.builder import AFBlockBuilder, phase_for_requests
from TokenSim.af.latency import (
    AF_MATCH_TYPES,
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
    GlobalRoundRobinReplicaSelector,
    ParallelExecutionPlanner,
)
from TokenSim.af.resources import AFResourceRegistry
from TokenSim.af.runtime import AFRuntimePlan
from TokenSim.hardware.device import dtype_bytes
from TokenSim.af.trace import AFTraceEvent, AFTraceRecorder
from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
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
        self.dtype = dtype
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.runtime_plan = runtime_plan
        self.resources = resources or AFResourceRegistry(
            env, cluster_config, worker_configs
        )
        self.trace = trace_recorder or AFTraceRecorder(
            cluster_config.af.trace_enabled,
            self.resources.resource_capacities(),
        )
        self.builder = AFBlockBuilder()
        self.replica_selector = GlobalRoundRobinReplicaSelector()
        self.expert_routing = (
            ExpertRouting(moe_config, seed=random_seed)
            if moe_config is not None and moe_config.enabled
            else None
        )
        self.plan_builder = ParallelExecutionPlanner(
            cluster_config,
            hidden_size=hidden_size,
            dtype_size_bytes=int(dtype_bytes(dtype)),
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

        def placement_factory(_block_id, _block_requests):
            f_instance = self.replica_selector.select(resolved_phase, f_domain)
            return self.plan_builder.placement(
                resolved_phase, source_identity, f_instance
            )

        def histogram_factory(block_requests):
            if self.expert_routing is None:
                return {}
            return self.expert_routing.histogram_for_step(list(block_requests))

        layer_types = None
        routing_snapshot_factory = None
        if self.runtime_plan is not None and self.runtime_plan.mla is not None and self.expert_routing is not None:
            moe_layers = getattr(self.runtime_plan, "moe_layers", None)
            if moe_layers is None and getattr(self.runtime_plan, "layers", None):
                moe_layers = tuple(
                    layer.layer_id
                    for layer in self.runtime_plan.layers
                    if layer.layer_type == "moe"
                )
            if moe_layers is not None:
                moe_set = set(moe_layers)
                layer_types = {
                    layer_id: ("moe" if layer_id in moe_set else "dense")
                    for layer_id in range(self.total_layers)
                }

                def routing_snapshot_factory(
                    iteration_id, snapshot_phase, layer_id, block_requests, placement
                ):
                    decode_step = max(
                        int(getattr(request, "generation_idx", 0))
                        for request in block_requests
                    )
                    return self.expert_routing.snapshot_for_layer(
                        block_requests,
                        iteration_id=iteration_id,
                        phase=snapshot_phase,
                        decode_step=decode_step,
                        layer_id=layer_id,
                        a_dp_rank=placement.a_dp_rank,
                        a_tp_rank=0,
                    )

        return self.builder.build(
            iteration_id=iteration_id,
            requests=requests,
            af_block_size=self.cluster_config.af.af_block_size,
            source_a_worker_id=source_a_worker_id,
            link=link,
            phase=resolved_phase,
            placement_factory=placement_factory,
            histogram_factory=histogram_factory,
            layer_types=layer_types,
            routing_snapshot_factory=routing_snapshot_factory,
        )

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
        processes: dict[str, simpy.Process] = {}
        done_events: dict[str, simpy.Event] = {}
        failure_event = self.env.event()
        failures: list[BaseException] = []
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
                )
            )
        # Tensor-parallel A execution may produce one terminal combine node
        # per rank.  The DAG makespan is defined by the last terminal, so wait
        # for all terminal nodes instead of requiring a single sink.
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
        query = AFComputeQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            stage=step.stage,
            token_count=local_work.token_count,
            context_lengths=block.context_lengths,
            model_id=self.model_id,
            dtype=self.dtype,
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
            input_edge_count=local_work.input_edge_count,
            output_token_count=local_work.output_token_count,
            assignment_count=local_work.assignment_count,
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
                token_count=block.token_count,
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
                tensor_slices=step.tensor_slices,
                source_layout=step.source_layout,
                target_layout=step.target_layout,
                routing_edges=step.routing_edges,
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
            link, step.direction, step.resource_group, step.resource_lane
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
        query = AFTransferQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            direction=step.direction,
            token_count=block.token_count,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            source_device=source_device,
            target_device=target_device,
            network=step.network or link.network,
            operation=step.operation,
            link_group=link.resource_group,
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
        )
        ready_at, start_at, end_at, estimate = yield from self._run_resources_stage(
            block,
            layer_id,
            step.node_id,
            tuple(resource for _, resource in keyed_resources),
            lambda: self._estimate(
                self.transfer_provider, query, block, layer_id, step.operation
            ),
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
                event_type=("collective" if step.operation in {"tp_collective", "all_reduce", "all_gather", "reduce_scatter"} else "transfer"),
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
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)

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
                f"block={block.block_id}, layer={layer_id}, stage={stage}, "
                f"model={getattr(query, 'model_id', '?')}, "
                f"domain={getattr(query, 'source_domain', '?')}, "
                f"rank={getattr(query, 'tp_rank', '?')}, "
                f"ep_rank={getattr(query, 'ep_rank', '?')}, "
                f"operation={getattr(query, 'operation', '?')}: {exc}"
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
        if estimate.match_type not in AF_MATCH_TYPES:
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
