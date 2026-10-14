"""Trace validation."""

from __future__ import annotations

from TokenSim.af.latency import AFRoutingEdge
from TokenSim.af.trace_event import AFTraceEvent, _endpoint_identity, _is_f_domain
from TokenSim.errors import SimulationStateError


class TraceValidationMixin:
    def validate_invariants(self, iteration_id: int) -> None:
        events = self.events_for(iteration_id)
        by_node: dict[str, AFTraceEvent] = {}
        tolerance = 1e-12
        for event in events:
            if not event.node_id:
                raise SimulationStateError(
                    f"AF trace invariant failed: iteration={iteration_id} event has no node_id"
                )
            if event.node_id in by_node:
                raise SimulationStateError(
                    f"AF trace invariant failed: duplicate node_id={event.node_id!r}"
                )
            by_node[event.node_id] = event
            if not (
                event.ready_at <= event.start_at + tolerance
                and event.start_at <= event.end_at + tolerance
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: invalid timestamps for node={event.node_id}"
                )
            if (
                abs(event.queue_duration - (event.start_at - event.ready_at))
                > tolerance
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: queue duration mismatch for node={event.node_id}"
                )
            if (
                abs(event.service_duration - (event.end_at - event.start_at))
                > tolerance
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: service duration mismatch for node={event.node_id}"
                )
            returned_payload = event.details.get("payload_bytes")
            if (
                returned_payload is not None
                and event.payload_bytes is not None
                and returned_payload != event.payload_bytes
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: payload mismatch for node={event.node_id}: "
                    f"planned={event.payload_bytes}, provider={returned_payload}"
                )
            if event.event_type == "transfer" and event.tensor_slices:
                slice_payloads = [
                    item.payload_bytes
                    for item in event.tensor_slices
                    if item.payload_bytes is not None
                ]
                if (
                    len(slice_payloads) == len(event.tensor_slices)
                    and event.payload_bytes is not None
                    and sum(slice_payloads) != event.payload_bytes
                ):
                    raise SimulationStateError(
                        "AF trace invariant failed: tensor slice payload total "
                        f"does not match leg payload for node={event.node_id}"
                    )
            if event.stage == "residual" and _is_f_domain(event.source_domain):
                raise SimulationStateError(
                    f"AF trace invariant failed: residual cannot execute in F domain "
                    f"for node={event.node_id}"
                )
            if event.stage == "weighted_combine":
                if (
                    event.original_token_count is None
                    or event.output_token_count != event.original_token_count
                ):
                    raise SimulationStateError(
                        "AF trace invariant failed: weighted combine must preserve "
                        f"original token count for node={event.node_id}"
                    )

        node_ids = set(by_node)
        for event in events:
            missing = set(event.predecessor_ids) - node_ids
            if missing:
                raise SimulationStateError(
                    f"AF trace invariant failed: node={event.node_id} has missing "
                    f"predecessors={sorted(missing)}"
                )
            if event.predecessor_ids:
                latest = max(
                    by_node[node_id].end_at for node_id in event.predecessor_ids
                )
                if event.ready_at + tolerance < latest:
                    raise SimulationStateError(
                        f"AF trace invariant failed: dependency time for node={event.node_id}; "
                        f"ready_at={event.ready_at}, predecessor_end={latest}"
                    )
                if (
                    len(event.predecessor_ids) > 1
                    and event.details.get("readiness_mode") != "invocation_events"
                    and abs(event.ready_at - latest) > tolerance
                ):
                    raise SimulationStateError(
                        f"AF trace invariant failed: join ready time for node={event.node_id}; "
                        f"ready_at={event.ready_at}, latest_predecessor={latest}"
                    )

        transfer_copies: set[tuple[object, ...]] = set()
        for event in events:
            if event.event_type != "transfer":
                continue
            for tensor_slice in event.tensor_slices:
                copy_key = (
                    event.logical_stage_id,
                    event.direction,
                    _endpoint_identity(event.source_endpoint),
                    _endpoint_identity(event.target_endpoint),
                    tensor_slice.tensor_item_id,
                    tensor_slice.interval,
                    tensor_slice.partial_id,
                )
                if copy_key in transfer_copies:
                    raise SimulationStateError(
                        "AF trace invariant failed: duplicate physical tensor copy "
                        f"in logical stage {event.logical_stage_id!r}: {copy_key!r}"
                    )
                transfer_copies.add(copy_key)

        by_layer: dict[tuple[int, int], list[AFTraceEvent]] = {}
        for event in events:
            by_layer.setdefault((event.block_id, event.layer_id), []).append(event)
        for (block_id, layer_id), layer_events in by_layer.items():
            # Negative layer IDs explicitly represent whole-model input/output
            # communication. They still participate in dependency and resource
            # checks, but do not contain a transformer layer's A/F stages.
            if layer_id < 0:
                if any(event.layer_type is not None for event in layer_events):
                    raise SimulationStateError(
                        "Model boundary cannot have a transformer layer type"
                    )
                continue
            for operation in ("af_a_to_f", "af_f_to_a"):
                operation_events = [
                    event for event in layer_events if event.operation == operation
                ]
                stage_ids = {
                    event.logical_stage_id or event.node_id
                    for event in operation_events
                }
                if len(stage_ids) != 1:
                    raise SimulationStateError(
                        f"AF trace invariant failed: block={block_id}, layer={layer_id} "
                        f"requires exactly one {operation} logical stage; "
                        f"got {len(stage_ids)}"
                    )
            typed_events = [
                event for event in layer_events if event.layer_type is not None
            ]
            if typed_events:
                self._validate_runtime_layer_semantics(
                    iteration_id,
                    block_id,
                    layer_id,
                    typed_events,
                )

        intervals: dict[str, list[tuple[float, int, str]]] = {}
        for event in events:
            if event.service_duration <= 0:
                continue
            for resource_id in event.resource_ids or (event.resource_id,):
                intervals.setdefault(resource_id, []).extend(
                    (
                        (event.start_at, 1, event.node_id),
                        (event.end_at, -1, event.node_id),
                    )
                )
        for resource_id, points in intervals.items():
            active = 0
            capacity = self.resource_capacities.get(resource_id, 1)
            for time, delta, node_id in sorted(
                points, key=lambda item: (item[0], item[1])
            ):
                active += delta
                if active > capacity:
                    raise SimulationStateError(
                        f"AF trace invariant failed: resource={resource_id!r} capacity="
                        f"{capacity} exceeded at time={time} by node={node_id}"
                    )

    @staticmethod
    def _validate_runtime_layer_semantics(
        iteration_id: int,
        block_id: int,
        layer_id: int,
        events: list[AFTraceEvent],
    ) -> None:
        context = f"iteration={iteration_id}, block={block_id}, layer={layer_id}"
        layer_types = {event.layer_type for event in events}
        if len(layer_types) != 1:
            raise SimulationStateError(
                f"AF trace invariant failed: {context} mixes layer types "
                f"{sorted(layer_types)}"
            )
        layer_type = next(iter(layer_types))
        attention = [event for event in events if event.stage == "attention"]
        if not attention:
            raise SimulationStateError(
                f"AF trace invariant failed: {context} has no A Attention events"
            )
        a_domains = {event.source_domain for event in attention}
        if len(a_domains) != 1:
            raise SimulationStateError(
                f"AF trace invariant failed: {context} spans A domains {a_domains}"
            )
        a_domain = next(iter(a_domains))
        attention_counts = {event.original_token_count for event in attention}
        if len(attention_counts) != 1 or None in attention_counts:
            raise SimulationStateError(
                f"AF trace invariant failed: Attention token counts differ for {context}"
            )
        attention_token_count = next(iter(attention_counts))
        residuals = [event for event in events if event.stage == "residual"]
        for event in residuals:
            if event.source_domain != a_domain or event.target_domain != a_domain:
                raise SimulationStateError(
                    f"AF trace invariant failed: residual must execute in A domain "
                    f"for node={event.node_id}"
                )
        attention_ranks = {event.participant_ranks[0] for event in attention}
        residual_ranks = {event.participant_ranks[0] for event in residuals}
        if residual_ranks != attention_ranks or len(residuals) != len(residual_ranks):
            raise SimulationStateError(
                f"AF trace invariant failed: residual does not cover every A owner "
                f"exactly once for {context}"
            )
        for event in residuals:
            if (
                event.token_count != attention_token_count
                or event.original_token_count != attention_token_count
                or event.output_token_count != attention_token_count
            ):
                raise SimulationStateError(
                    "AF trace invariant failed: residual must preserve original "
                    f"tokens for node={event.node_id}"
                )

        referenced = {
            predecessor for event in events for predecessor in event.predecessor_ids
        }
        terminals = [event for event in events if event.node_id not in referenced]
        if not terminals:
            raise SimulationStateError(
                f"AF trace invariant failed: {context} has no terminal events"
            )
        for terminal in terminals:
            if (
                terminal.source_domain != a_domain
                or terminal.target_domain != a_domain
                or (
                    terminal.stage not in {"residual", "layout"}
                    and terminal.operation != "all_gather"
                    and not (
                        terminal.event_type == "synchronize"
                        and terminal.details.get("communication_operation")
                        == "all_gather"
                    )
                )
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: runtime terminal must be A-side "
                    f"residual/layout for node={terminal.node_id}"
                )

        if layer_type == "dense":
            forbidden = {
                "router",
                "routed_expert",
                "shared_expert",
                "weighted_combine",
            }
            present = {event.stage for event in events} & forbidden
            if present:
                raise SimulationStateError(
                    f"AF trace invariant failed: Dense layer has MoE stages "
                    f"{sorted(present)}"
                )
            return

        routers = [event for event in events if event.stage == "router"]
        if len(routers) != 1:
            raise SimulationStateError(
                f"AF trace invariant failed: {context} must have one Router event"
            )
        router = routers[0]
        expected_edges = tuple(router.routing_edges)
        expected_edge_set = set(expected_edges)
        if (
            len(expected_edges) != len(expected_edge_set)
            or router.assignment_count != len(expected_edges)
            or router.original_token_count is None
        ):
            raise SimulationStateError(
                f"AF trace invariant failed: Router workload is inconsistent for "
                f"node={router.node_id}"
            )
        original_tokens = router.original_token_count
        if original_tokens != attention_token_count:
            raise SimulationStateError(
                f"AF trace invariant failed: Router token count differs from "
                f"Attention for {context}"
            )

        routed = [event for event in events if event.stage == "routed_expert"]
        consumers: dict[AFRoutingEdge, list[AFTraceEvent]] = {}
        for event in routed:
            if event.assignment_count != len(event.routing_edges) or len(
                event.routing_edges
            ) != len(set(event.routing_edges)):
                raise SimulationStateError(
                    f"AF trace invariant failed: Routed Expert workload is "
                    f"inconsistent for node={event.node_id}"
                )
            for edge in event.routing_edges:
                consumers.setdefault(edge, []).append(event)
        if set(consumers) != expected_edge_set:
            raise SimulationStateError(
                f"AF trace invariant failed: Routed Expert assignment coverage "
                f"differs from Router for {context}"
            )
        for edge, edge_events in consumers.items():
            groups = {event.tp_group for event in edge_events}
            actual_ranks = [event.participant_ranks[0] for event in edge_events]
            if (
                len(groups) != 1
                or len(actual_ranks) != len(set(actual_ranks))
                or set(actual_ranks) != set(next(iter(groups)))
            ):
                raise SimulationStateError(
                    "AF trace invariant failed: routing assignment must execute "
                    f"once per Expert TP rank for expert={edge.expert_id}"
                )

        shared = [event for event in events if event.stage == "shared_expert"]
        if shared:
            groups = {event.tp_group for event in shared}
            ranks = [event.participant_ranks[0] for event in shared]
            if (
                len(groups) != 1
                or len(ranks) != len(set(ranks))
                or set(ranks) != set(next(iter(groups)))
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: Shared Expert TP coverage for {context}"
                )
            for event in shared:
                if (
                    event.token_count != original_tokens
                    or event.original_token_count != original_tokens
                    or event.output_token_count != original_tokens
                ):
                    raise SimulationStateError(
                        "AF trace invariant failed: Shared Expert must process every "
                        f"original token for node={event.node_id}"
                    )

        combines = [event for event in events if event.stage == "weighted_combine"]
        combine_ranks = {event.participant_ranks[0] for event in combines}
        if combine_ranks != attention_ranks or len(combines) != len(combine_ranks):
            raise SimulationStateError(
                f"AF trace invariant failed: combine does not cover every A owner "
                f"for {context}"
            )
        expected_inputs = len(expected_edges) + (original_tokens if shared else 0)
        for event in combines:
            expected_domain = (
                router.source_domain
                if event.operation == "f_weighted_combine"
                else a_domain
            )
            if event.source_domain != expected_domain or (
                event.operation == "f_weighted_combine"
                and (expected_domain == a_domain or len(combines) != 1)
            ):
                raise SimulationStateError(
                    "AF trace invariant failed: combine protocol domain mismatch"
                )
            if (
                event.original_token_count != original_tokens
                or event.assignment_count != len(expected_edges)
                or event.input_edge_count != expected_inputs
                or event.output_token_count != original_tokens
                or len(event.routing_edges) != len(expected_edges)
                or set(event.routing_edges) != expected_edge_set
            ):
                raise SimulationStateError(
                    f"AF trace invariant failed: combine workload is inconsistent "
                    f"for node={event.node_id}"
                )
