"""Executor resources."""

from __future__ import annotations

import simpy

from TokenSim.af.latency import AFServiceEstimate
from TokenSim.af.models import AFBlock
from TokenSim.af.trace import AFTraceEvent


class ExecutorResourcesMixin:
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
