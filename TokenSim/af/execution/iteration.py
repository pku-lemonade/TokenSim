"""Executor iteration."""

from __future__ import annotations

import simpy

from TokenSim.af.planning.batches import AFBlock, AFIteration
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError, SimulationStateError


class ExecutorIterationMixin:
    def run_iteration(self, iteration: AFIteration):
        if iteration.iteration_id in self.active_iterations:
            raise SimulationStateError(
                f"AF iteration {iteration.iteration_id} is already active"
            )
        native_control = (
            self.cluster_config.af.communication_timing.native_control_enabled
        )
        if native_control and self.active_iterations:
            raise SimulationStateError(
                "native control eager mode supports one in-flight iteration"
            )
        if native_control and len(iteration.blocks) > 1:
            raise ConfigurationError(
                "native control eager DP=1 requires one unsplit block per iteration"
            )
        iteration.start_at = self.env.now
        self.active_iterations[iteration.iteration_id] = iteration
        self._step_preparation[iteration.iteration_id] = self._start_step_preparation(
            iteration
        )
        failure_event = self.env.event()
        failures: list[BaseException] = []
        done_events = [self.env.event() for _ in iteration.blocks]
        processes = [
            self.env.process(
                self._run_block_guarded(block, failure_event, failures, done_event)
            )
            for block, done_event in zip(iteration.blocks, done_events)
        ]
        if native_control and iteration.blocks:
            state = {
                "a_ready": self.env.event(),
                "f_ready": self.env.event(),
                "a_node": f"i{iteration.iteration_id}:control:payload:transport",
                "f_node": f"i{iteration.iteration_id}:control:decode",
            }
            self._native_controls[iteration.iteration_id] = state
            if self.cluster_config.af.communication_timing.native_call_order_enabled:
                self._start_native_call_order(iteration.blocks[0], state)
            control_done = self.env.event()
            done_events.append(control_done)
            processes.append(
                self.env.process(
                    self._run_control_guarded(
                        iteration.blocks[0],
                        state,
                        failure_event,
                        failures,
                        control_done,
                    )
                )
            )
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
            state = self._native_controls.pop(iteration.iteration_id, None)
            receivers = (
                [call["receiver"] for call in state.get("calls", {}).values()]
                if state
                else []
            )
            for receiver in receivers:
                if receiver.is_alive:
                    receiver.interrupt("native iteration finished or cancelled")
            if receivers:
                yield self.env.all_of(receivers)
            self._fused_combines.pop(iteration.iteration_id, None)
            preparations = self._step_preparation.pop(iteration.iteration_id, {})
            for process, _ in preparations.values():
                if process.is_alive:
                    process.interrupt("AF iteration preparation cancelled")
            if preparations:
                yield self.env.all_of([process for process, _ in preparations.values()])
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

    def _run_control_guarded(self, block, state, failure_event, failures, done):
        try:
            yield from self._run_native_control(block, state)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            failures.append(exc)
            if not failure_event.triggered:
                failure_event.succeed()
        finally:
            if not done.triggered:
                done.succeed()

    def _run_block_guarded(
        self,
        block: AFBlock,
        failure_event: simpy.Event,
        failures: list[BaseException],
        done_event: simpy.Event,
    ):
        try:
            control = self._native_controls.get(block.iteration_id)
            if control is not None:
                yield self.env.any_of((control["a_ready"], failure_event))
                if failure_event.triggered:
                    return
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
        processes: list[simpy.Process] = []
        done_events: list[simpy.Event] = []
        failure_event = self.env.event()
        failures: list[BaseException] = []
        if hasattr(dag, "predecessor_indices"):
            predecessor_indices = dag.predecessor_indices
            terminal_indices = dag.terminal_indices
        else:
            indices = {node.node_id: i for i, node in enumerate(dag.nodes)}
            predecessor_indices = tuple(
                tuple(indices[p] for p in node.predecessor_ids) for node in dag.nodes
            )
            terminal_indices = tuple(indices[p] for p in dag.terminal_node_ids)
        for index, node in enumerate(dag.nodes):
            done_event = self.env.event()
            done_events.append(done_event)
            predecessors = tuple(
                done_events[item] for item in predecessor_indices[index]
            )
            bindings = getattr(node, "input_producer_ids", ())
            subgroup_inputs = None
            if (
                bindings
                and self.cluster_config.af.transfer_topology_mode == "physical"
                and hasattr(self.transfer_provider, "communication_plan")
            ):
                events_by_id = dict(zip(node.predecessor_ids, predecessors))
                subgroup_inputs = {
                    endpoint: (producer, events_by_id[producer])
                    for endpoint, producer in bindings
                }
                data_ids = {producer for _, producer in bindings}
                predecessors = tuple(
                    event
                    for producer, event in events_by_id.items()
                    if producer not in data_ids
                )
            processes.append(
                self.env.process(
                    self._run_dag_node_guarded(
                        block,
                        layer_id,
                        link,
                        node,
                        predecessors,
                        failure_event,
                        failures,
                        done_event,
                        subgroup_inputs,
                    )
                )
            )
        # Tensor-parallel A execution may produce one terminal combine node
        # per rank.  The DAG makespan is defined by the last terminal, so wait
        # for all terminal nodes instead of requiring a single sink.
        terminal_done = self.env.all_of(
            tuple(done_events[index] for index in terminal_indices)
        )
        try:
            yield self.env.any_of((terminal_done, failure_event))
            if failures:
                for process in processes:
                    if process.is_alive:
                        process.interrupt(
                            f"AF DAG failed at iteration={block.iteration_id}, "
                            f"block={block.block_id}, layer={layer_id}"
                        )
                yield self.env.all_of(tuple(done_events))
                raise failures[0]
        except simpy.Interrupt:
            for process in processes:
                if process.is_alive:
                    process.interrupt("AF model block cancelled")
            yield self.env.all_of(tuple(done_events))
            raise

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
        subgroup_inputs=None,
    ):
        try:
            if predecessors:
                predecessor_done = self.env.all_of(predecessors)
                yield self.env.any_of((predecessor_done, failure_event))
                if failure_event.triggered:
                    return
            if failure_event.triggered:
                return
            if subgroup_inputs is not None:
                yield from self._run_planned_communication(
                    block,
                    layer_id,
                    link,
                    node,
                    input_readiness=(subgroup_inputs, failure_event),
                )
            else:
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
