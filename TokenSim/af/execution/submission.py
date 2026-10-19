"""Executor submission."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from TokenSim.af.planning.batching import AFIteration, validate_decode_requests
from TokenSim.errors import ConfigurationError


class ExecutorSubmissionMixin:
    def submit(
        self,
        source_a_worker_id: int,
        requests: Sequence[Any],
    ) -> AFIteration:
        if self.plan_builder is None:
            raise ConfigurationError("AF submission requires a compiled runtime plan")
        validate_decode_requests(requests)
        if self.operation_profile is not None:
            self.operation_profile.validate_workload(requests, "decode")
        if self.overhead_profile is not None:
            self.overhead_profile.validate_workload(requests, "decode")
        link = self.cluster_config.af_link
        source_group = self.resources.group_for_worker(source_a_worker_id)
        if source_group != link.attention_group:
            raise ConfigurationError(
                f"AF decode requires attention group "
                f"{link.attention_group!r}, but worker {source_a_worker_id} belongs to "
                f"{source_group!r}"
            )
        source_worker = self.resources.worker(source_a_worker_id)
        if source_worker.physical_worker_ref is None:
            raise ConfigurationError(
                f"AF source worker {source_a_worker_id} has no physical identity"
            )
        source_identity = source_worker.physical_worker_ref.identity
        a_domain, f_domain = self.cluster_config.af_domains
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
                    f"request {request.id} AF decode owner is "
                    f"{previous!r}, cannot move to {a_instance.name!r} without KV transfer"
                )
            owners[owner_key] = a_instance.name
        iteration_id = self._next_iteration_id
        self._next_iteration_id += 1
        runtime_layer_types = self._runtime_layer_types()
        layer_types = runtime_layer_types

        def placement_factory(_block_id, _block_requests):
            selection = self.replica_selector.select_for_preselected_a(
                a_domain,
                f_domain,
                a_instance,
            )
            placement = self.plan_builder.placement(
                source_identity, selection.f_service
            )
            if placement.a_instance != selection.a_owner.name:
                raise ConfigurationError(
                    "AF scheduler owner changed while selecting the F service "
                    f"replica: expected={selection.a_owner.name!r}, "
                    f"actual={placement.a_instance!r}"
                )
            return placement

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
            if self.routing_replay is not None:
                block_requests = self.routing_replay.requests_for(
                    block_requests,
                    phase=snapshot_phase,
                    layer_id=layer_id,
                    decode_step=self._routing_step(block_requests),
                )
            return self.expert_routing.snapshot_for_layer(
                block_requests,
                iteration_id=snapshot_iteration_id,
                phase=snapshot_phase,
                decode_step=self._routing_step(block_requests),
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
            placement_factory=placement_factory,
            layer_types=layer_types,
            routing_snapshot_factory=(
                routing_snapshot_factory if "moe" in layer_types.values() else None
            ),
        )

    def _runtime_layer_types(self) -> dict[int, str]:
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
    def _routing_step(requests: Sequence[Any]) -> int:
        steps = {int(getattr(request, "generation_idx", 0)) for request in requests}
        if not steps or min(steps) < 0:
            raise ConfigurationError(
                f"AF Decode steps must be non-negative; got {sorted(steps)}"
            )
        return min(steps)

    def _canonical_a_tp_rank(self, placement) -> int:
        ownerships = self.runtime_plan.ownerships_for_replica(
            placement.a_domain,
            placement.a_dp_rank,
        )
        canonical = tuple(item for item in ownerships if item.canonical_producer)
        if len(canonical) != 1:
            raise ConfigurationError(
                "AF runtime A replica must have exactly one canonical routing producer"
            )
        return canonical[0].tp_rank
