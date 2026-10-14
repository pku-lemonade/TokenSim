"""Executor transport."""

from __future__ import annotations

import simpy

from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.errors import SimulationStateError


class ExecutorTransportMixin:
    def _run_domain_plan(
        self,
        block,
        layer_id,
        step,
        wire_node_id,
        predecessors,
        plan,
        input_readiness=None,
    ):
        """Instantiate cached phases with fresh dependency events and FIFO waits."""
        ready = self.env.now
        single_phase = len(plan.phases) == 1 and not plan.phases[0].local_work_bytes
        completed = (
            {}
            if single_phase
            else {phase.phase_id: self.env.event() for phase in plan.phases}
        )
        intervals = {}
        children = []

        def run_phase(phase):
            dependencies = [completed[key] for key in phase.predecessor_ids]
            input_ids = ()
            if input_readiness is not None and not phase.predecessor_ids:
                bindings, failure = input_readiness
                identities = tuple(
                    self.resources.worker(rank).physical_worker_ref.identity
                    for rank in phase.participants
                )
                input_ids = tuple(
                    dict.fromkeys(bindings[identity][0] for identity in identities)
                )
                dependencies.extend(bindings[identity][1] for identity in identities)
            if dependencies:
                yield self.env.all_of(dependencies)
            if input_readiness is not None and input_readiness[1].triggered:
                raise SimulationStateError("collective input producer failed")
            phase_predecessors = tuple(
                dict.fromkeys(
                    (
                        *input_ids,
                        *(
                            tuple(
                                f"{wire_node_id}:phase:{key}"
                                for key in phase.predecessor_ids
                            )
                            or predecessors
                        ),
                    )
                )
            )
            estimate = self.transfer_provider.phase_estimate(phase)
            identity = (
                block.iteration_id,
                block.block_id,
                layer_id,
                step.node_id,
                phase.phase_id,
            )
            node_id = f"{wire_node_id}:phase:{phase.phase_id}"
            self._set_active_stage(block, layer_id, node_id, self.env.now)
            try:
                local_children = ()
                if phase.local_work_bytes:
                    phase_ready = self.env.now
                    local_children = tuple(
                        f"{node_id}:worker:{rank}" for rank in phase.participants
                    )
                    processes = [
                        self.env.process(
                            self._run_communication_reorder(
                                block,
                                layer_id,
                                step,
                                phase,
                                child_id,
                                phase_predecessors,
                                rank,
                            )
                        )
                        for rank, child_id in zip(phase.participants, local_children)
                    ]
                    for process in processes:
                        process.defused = True
                    joined = self.env.all_of(processes)
                    joined.defused = True
                    try:
                        yield joined
                    except BaseException:
                        for process in processes:
                            if process.is_alive:
                                process.interrupt(
                                    "local communication reorder cancelled"
                                )
                        for process in processes:
                            if not process.triggered:
                                try:
                                    yield process
                                except BaseException:
                                    pass
                        raise
                    times = (phase_ready, phase_ready, self.env.now)
                elif phase.resource_ids:
                    times = yield from self.resources.domain_scheduler.run(
                        phase.resource_ids, estimate.service_duration, identity
                    )
                else:
                    times = (self.env.now, self.env.now, self.env.now)
                intervals[phase.phase_id] = times
                if len(plan.phases) > 1 and self.trace.enabled:
                    phase_ready, start, end = times
                    if local_children:
                        phase_ready = start = end
                    self.trace.record(
                        AFTraceEvent(
                            iteration_id=block.iteration_id,
                            block_id=block.block_id,
                            layer_id=layer_id,
                            phase=block.phase,
                            event_type="collective",
                            operation=phase.operation,
                            direction=step.direction,
                            resource_id=(
                                phase.resource_ids[0]
                                if phase.resource_ids
                                else f"noop:{node_id}"
                            ),
                            resource_ids=phase.resource_ids,
                            ready_at=phase_ready,
                            start_at=start,
                            end_at=end,
                            queue_duration=start - phase_ready,
                            service_duration=end - start,
                            request_count=block.request_count,
                            token_count=block.token_count,
                            source_id=estimate.source_id,
                            dataset_version=estimate.dataset_version,
                            match_type=estimate.match_type,
                            payload_bytes=phase.logical_input_bytes,
                            details={
                                **estimate.details,
                                "routing_provenance": plan.provenance,
                                "dependency_wait_duration": phase_ready - ready,
                            },
                            node_id=node_id,
                            predecessor_ids=local_children or phase_predecessors,
                            logical_stage_id=step.logical_stage_id,
                            source_domain=step.source_domain,
                            target_domain=step.target_domain,
                            layer_type=step.layer_type,
                            original_token_count=step.metadata.get(
                                "original_token_count", block.token_count
                            ),
                        )
                    )
                if not single_phase:
                    completed[phase.phase_id].succeed()
            finally:
                self.active_stages.pop(
                    (block.iteration_id, block.block_id, node_id), None
                )

        try:
            # One transport phase needs no child Process or ConditionValue.
            # Retain the urgent start and two normal completion turns of the
            # old Initialize -> Process -> AllOf path. Removing those turns
            # can change downstream compute/host FIFO ties at the same time.
            # The old phase-done event had no consumers and can be omitted.
            if single_phase:
                start_event = self.env.event()
                start_event._ok, start_event._value = True, None
                self.env.schedule(start_event, priority=simpy.events.URGENT)
                yield start_event
                error = None
                try:
                    yield from run_phase(plan.phases[0])
                except simpy.Interrupt:
                    raise
                except BaseException as exc:
                    error = exc
                yield self.env.timeout(0)
                yield self.env.timeout(0)
                if error is not None:
                    raise error
            else:
                children = [self.env.process(run_phase(phase)) for phase in plan.phases]
            if children:
                # Parent cancellation detaches its wait callback. Defuse the
                # abandoned join and each child, while yielding still propagates
                # failures to the parent and runs complete resource cleanup.
                for child in children:
                    child.defused = True
                joined = self.env.all_of(children)
                joined.defused = True
                yield joined
        except BaseException:
            for child in children:
                if child.is_alive:
                    child.interrupt("communication plan cancelled")
            # Defuse child interrupts during cleanup; the parent failure remains.
            for child in children:
                if not child.triggered:
                    try:
                        yield child
                    except BaseException:
                        pass
            raise
        start = min((times[1] for times in intervals.values()), default=ready)
        end = max((times[2] for times in intervals.values()), default=ready)
        phase_ready = min((times[0] for times in intervals.values()), default=ready)
        return phase_ready, start, end

    def _run_communication_reorder(
        self, block, layer_id, step, phase, node_id, predecessors, worker_id
    ):
        worker = self.resources.worker(worker_id)
        query = AFComputeQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            stage="layout",
            operation="data_reorder",
            token_count=1,
            local_token_count=1,
            context_lengths=block.context_lengths,
            model_id=self.model_id,
            dtype=phase.dtype,
            device_type=worker.hardware,
            worker_id=worker_id,
            source_domain=step.source_domain,
            physical_worker=worker.physical_worker_ref.identity,
            local_shape={
                "tokens": 1,
                "hidden_size": self.hidden_size,
                "global_hidden_size": self.hidden_size,
                "tp_size": 1,
                "reorder_bytes": phase.local_work_bytes,
            },
        )
        estimate = self._estimate(
            self.compute_provider, query, block, layer_id, node_id
        )
        ready, start, end, _ = yield from self._run_resources_stage(
            block,
            layer_id,
            node_id,
            (self.resources.compute_slots[worker_id],),
            lambda: estimate,
        )
        if self.trace.enabled:
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type="compute",
                    operation="data_reorder",
                    direction=None,
                    resource_id=f"worker:{worker_id}",
                    resource_ids=(f"worker:{worker_id}",),
                    ready_at=ready,
                    start_at=start,
                    end_at=end,
                    queue_duration=start - ready,
                    service_duration=end - start,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    match_type=estimate.match_type,
                    node_id=node_id,
                    predecessor_ids=predecessors,
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    layer_type=step.layer_type,
                    stage="layout",
                    logical_stage_id=step.logical_stage_id,
                    details={
                        **estimate.details,
                        "local_work_bytes": phase.local_work_bytes,
                        "layout_permutation": phase.layout_permutation,
                        "wire_bytes": 0,
                    },
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)
