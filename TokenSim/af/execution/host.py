"""Executor host."""

from __future__ import annotations

from types import SimpleNamespace

import simpy

from TokenSim.af.calibration.host import COMPUTE_CALL_FIELDS, MESSAGE_CALL_FIELDS


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
