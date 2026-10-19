"""Host preparation, submission and shared-resource services."""

from __future__ import annotations

from types import SimpleNamespace

import simpy

from TokenSim.af.calibration.host import COMPUTE_CALL_FIELDS, MESSAGE_CALL_FIELDS
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.af.trace.metrics import AFTraceEvent


class ExecutorHostMixin:
    def _host_cost_us(self, name, block=None, *, tokens=None):
        fallback = getattr(self.cluster_config.af.communication_timing, name, 0.0)
        if self.overhead_profile is None:
            return fallback
        count = tokens if tokens is not None else block.token_count
        return self.overhead_profile.cost(name, count, fallback)

    def _native_call_cost_field(self, step, action):
        if action == "compute":
            if (
                step.operation == "router_topk"
                and self.overhead_profile is not None
                and "compute_f_moe_block_us" in self.overhead_profile.document["costs"]
            ):
                return "compute_f_moe_block_us"
            field = COMPUTE_CALL_FIELDS.get(step.operation)
            fallback = "compute_submit_us"
        else:
            backend = step.metadata.get("transport_backend", "pynccl")
            field = MESSAGE_CALL_FIELDS.get((backend, step.direction, action))
            fallback = (
                "host_submit_us" if action == "send" else "host_receive_submit_us"
            )
        if (
            self.overhead_profile is not None
            and field in self.overhead_profile.document["costs"]
        ):
            return field
        return fallback

    def host_cost_provenance(self, name, *, tokens):
        """Materialized per-call cost, distinct from whole-engine validation."""
        profile = self.overhead_profile
        covered = profile is not None and name in profile.document["costs"]
        explicit = (
            name in self.cluster_config.af.communication_timing._explicit_cost_fields
        )
        measured = covered and profile.document["evidence"]["kind"] == "measured"
        return dict(
            value_us=self._host_cost_us(name, tokens=tokens),
            unit="us",
            calibrated=measured,
            origin=(
                "measured_profile"
                if measured
                else (
                    "synthetic_test_profile"
                    if covered
                    else "manual_assumption" if explicit else "unmeasured_default"
                )
            ),
            profile_id=profile.document["profile_id"] if covered else None,
            profile_fingerprint=profile.fingerprint if covered else None,
            coverage=(
                list(profile.document["costs"][name]["coverage"]) if covered else []
            ),
            validation_status=(
                profile.document["evidence"]["validation_status"] if covered else None
            ),
        )

    def _start_step_preparation(self, iteration):
        """One independent host preparation per participating worker/iteration."""
        timing = self.cluster_config.af.communication_timing
        preparations = {}
        a_cost = self._host_cost_us(
            "a_step_prepare_us", tokens=len(iteration.request_ids)
        )
        f_cost = self._host_cost_us(
            "f_step_prepare_us", tokens=len(iteration.request_ids)
        )
        schedule_cost = self._host_cost_us(
            "engine_step_schedule_us", tokens=len(iteration.request_ids)
        )
        input_operations = tuple(
            name
            for name in ("a_input_prepare", "a_attention_metadata")
            if self.operation_profile is not None
            and name in self.operation_profile.document["operations"]
        )
        input_operation = bool(input_operations)
        if input_operation:
            a_cost = sum(
                self.operation_profile.document["operations"][name]["fixed_us"]
                for name in input_operations
            )
        if not timing.enabled or not (
            a_cost or f_cost or input_operation or schedule_cost
        ):
            return preparations
        for block in iteration.blocks:
            for domain, instance, duration, field in (
                (
                    block.placement.a_domain,
                    block.placement.a_instance,
                    a_cost,
                    "a_step_prepare_us",
                ),
                (
                    block.placement.f_domain,
                    block.placement.f_instance,
                    f_cost,
                    "f_step_prepare_us",
                ),
            ):
                if not duration and not (
                    (input_operation or schedule_cost) and field == "a_step_prepare_us"
                ):
                    continue
                for binding in self.runtime_plan.bindings_for_instance(
                    domain, instance
                ):
                    identity = binding.physical_identity
                    if identity in preparations:
                        continue
                    node_id = f"i{iteration.iteration_id}:prepare:{identity}"
                    if (
                        input_operation or schedule_cost
                    ) and field == "a_step_prepare_us":
                        process = self.env.process(
                            self._run_a_step_preparation(
                                block, identity, node_id, duration
                            )
                        )
                        preparations[identity] = (process, node_id)
                        continue
                    step = SimpleNamespace(
                        operation="step_prepare",
                        direction=None,
                        logical_stage_id=None,
                        leg_id=None,
                        source_domain=domain,
                        target_domain=domain,
                        layer_type=None,
                    )
                    process = self.env.process(
                        self._run_preparation_stage(
                            block,
                            -1,
                            step,
                            node_id,
                            (),
                            (identity,),
                            duration / 1e6,
                            "host_prepare",
                            details={
                                "cost_field": field,
                                "cost_scope": "per_step_per_worker",
                                "calibrated": False,
                            },
                        )
                    )
                    preparations[identity] = (process, node_id)
        return preparations

    def _run_a_step_preparation(self, block, identity, node_id, duration):
        try:
            predecessors = ()
            schedule = self._host_cost_us("engine_step_schedule_us", block)
            if schedule:
                scheduled = node_id + ":schedule"
                template = self._control_nodes(block)[0]
                yield from self._run_communication_host_stage(
                    block,
                    -1,
                    template,
                    scheduled,
                    (),
                    (identity,),
                    schedule / 1e6,
                    "host_prepare",
                    details={
                        "cost_field": "engine_step_schedule_us",
                        "cost_scope": "per_step_per_worker",
                    },
                )
                predecessors = (scheduled,)
            if self.operation_profile is not None and any(
                n in self.operation_profile.document["operations"]
                for n in ("a_input_prepare", "a_attention_metadata")
            ):
                yield from self._run_input_preparation(
                    block, identity, node_id, predecessors
                )
            else:
                yield from self._run_communication_host_stage(
                    block,
                    -1,
                    self._control_nodes(block)[0],
                    node_id,
                    predecessors,
                    (identity,),
                    duration / 1e6,
                    "host_prepare",
                    details={
                        "cost_field": "a_step_prepare_us",
                        "cost_scope": "per_step_per_worker",
                    },
                )
        except simpy.Interrupt:
            return

    def _run_input_preparation(self, block, identity, node_id, predecessors=()):
        try:
            names = [
                name
                for name in ("a_input_prepare", "a_attention_metadata")
                if name in self.operation_profile.document["operations"]
            ]
            for index, name in enumerate(names):
                completed = yield from self._control_stage(
                    block,
                    self._control_nodes(block)[0],
                    name,
                    predecessors,
                    identity,
                    0,
                    device=True,
                    node_id=(
                        node_id if index == len(names) - 1 else node_id + ":" + name
                    ),
                )
                predecessors = (completed,)
        except simpy.Interrupt:
            return

    def _run_preparation_stage(self, *args, **kwargs):
        try:
            yield from self._run_communication_host_stage(*args, **kwargs)
        except simpy.Interrupt:
            return

    def _run_receiver_submit(
        self,
        block,
        layer_id,
        step,
        node_id,
        duration,
        cost_field="host_receive_submit_us",
    ):
        try:
            prepared = yield from self._wait_step_preparation(
                block, (step.physical_workers[-1],)
            )
            control = self._native_controls.get(block.iteration_id)
            if (
                control is not None
                and step.direction == "a_to_f"
                and step.metadata.get("message_class") != "control"
            ):
                yield control["f_ready"]
                prepared = (*prepared, control["f_node"])
            yield from self._run_communication_host_stage(
                block,
                layer_id,
                step,
                node_id,
                tuple(dict.fromkeys((*step.predecessor_ids, *prepared))),
                (step.physical_workers[-1],),
                duration,
                "host_submit",
                details={
                    "cost_field": cost_field,
                    "cost_scope": "per_receive_call",
                    "calibrated": False,
                },
            )
        except simpy.Interrupt:
            return

    def _wait_step_preparation(self, block, identities):
        preparations = self._step_preparation.get(block.iteration_id, {})
        selected = [preparations[x] for x in identities if x in preparations]
        if selected:
            yield self.env.all_of([event for event, _ in selected])
        return tuple(node_id for _, node_id in selected)

    def _run_communication_host_stage(
        self,
        block,
        layer_id,
        step,
        node_id,
        predecessors,
        participants,
        duration,
        event_type,
        details=None,
    ):
        """Run host service without holding the communication fabric."""
        keyed = self.resources.host_resources_for(participants)
        if len(keyed) > 1:
            # Each participating host submits independently. Acquiring all hosts
            # before starting service would hold an idle host behind a busy peer.
            child_ids = tuple(f"{node_id}:host:{identity}" for identity in participants)
            children = [
                self.env.process(
                    self._run_communication_host_stage(
                        block,
                        layer_id,
                        step,
                        child_id,
                        predecessors,
                        (identity,),
                        duration,
                        event_type,
                        details,
                    )
                )
                for identity, child_id in zip(participants, child_ids)
            ]
            try:
                yield self.env.all_of(children)
            except simpy.Interrupt:
                for child in children:
                    if child.is_alive:
                        child.interrupt("AF collective host stage cancelled")
                raise
            # Retain the original DAG completion identity as a zero-cost barrier.
            predecessors = child_ids
            duration = 0.0
            event_type = "synchronize"
        estimate = AFServiceEstimate(
            duration,
            (
                f"profile:{self.overhead_profile.fingerprint}:{details['cost_field']}"
                if self.overhead_profile is not None
                and details
                and details.get("cost_field") in self.overhead_profile.document["costs"]
                else "assumption:communication-host-cost"
            ),
            dataset_version="communication-timing-config",
            details=(
                {
                    "communication_operation": step.operation,
                    "timing_scope": (
                        self.overhead_profile.document["timing_scope"]
                        if self.overhead_profile is not None
                        else "host_service"
                    ),
                    "fixed_cost_us": duration * 1e6,
                    "host_resource_scope": (
                        self.cluster_config.af.communication_timing.host_resource_scope
                    ),
                    **(details or {}),
                    **(
                        {"host_profile": self.overhead_profile.describe()}
                        if self.overhead_profile is not None
                        else {}
                    ),
                    **(
                        {
                            "effective_cost": self.host_cost_provenance(
                                details["cost_field"], tokens=block.token_count
                            ),
                            "calibrated": self.host_cost_provenance(
                                details["cost_field"], tokens=block.token_count
                            )["calibrated"],
                        }
                        if details and details.get("cost_field")
                        else {}
                    ),
                }
                if self.trace.enabled
                else {}
            ),
        )
        ready, start, end, _ = yield from self._run_resources_stage(
            block,
            layer_id,
            node_id,
            tuple(resource for _, resource in keyed),
            lambda: estimate,
        )
        if self.trace.enabled:
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type=event_type,
                    operation=event_type,
                    direction=step.direction,
                    resource_id=keyed[0][0],
                    resource_ids=tuple(key for key, _ in keyed),
                    ready_at=ready,
                    start_at=start,
                    end_at=end,
                    queue_duration=start - ready,
                    service_duration=end - start,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    details=estimate.details,
                    node_id=node_id,
                    predecessor_ids=predecessors,
                    logical_stage_id=step.logical_stage_id,
                    leg_id=step.leg_id,
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    layer_type=step.layer_type,
                    physical_endpoints=participants,
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)

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
            self.active_stages.pop((block.iteration_id, block.block_id, stage), None)
        return ready_at, start_at, end_at, estimate
