"""Indexed event storage and deployment-independent timing validation."""

import math

from TokenSim.errors import SimulationStateError
from TokenSim.trace.occupancy import validate_resource_capacity


class TraceRecorder:
    def __init__(self, enabled=True, resource_capacities=None):
        self.enabled = enabled
        self.validate_on_completion = True
        self.events = []
        self.iteration_event_indices = {}
        self.resource_capacities = dict(resource_capacities or {})

    def record(self, event):
        if self.enabled:
            index = len(self.events)
            self.events.append(event)
            self.iteration_event_indices.setdefault(event.iteration_id, []).append(
                index
            )

    def clear(self):
        self.events.clear()
        self.iteration_event_indices.clear()

    def close(self):
        self.clear()

    def complete_iteration(self, iteration_id):
        """Memory recorders retain completed events until explicitly cleared."""

    def events_for(self, iteration_id):
        return tuple(
            self.events[i] for i in self.iteration_event_indices.get(iteration_id, ())
        )

    def validate_invariants(self, iteration_id):
        events = self.events_for(iteration_id)
        by_node = {}
        tolerance = 1e-12
        for event in events:
            if not event.node_id or event.node_id in by_node:
                raise SimulationStateError(
                    "trace requires unique nonempty node identifiers"
                )
            by_node[event.node_id] = event
            times = (
                event.ready_at,
                event.start_at,
                event.end_at,
                event.service_duration,
                event.queue_duration,
            )
            if not all(math.isfinite(value) for value in times):
                raise SimulationStateError("trace timestamps must be finite")
            if (
                event.ready_at > event.start_at + tolerance
                or event.start_at > event.end_at + tolerance
            ):
                raise SimulationStateError("trace timestamps are out of order")
            if (
                abs(event.service_duration - (event.end_at - event.start_at))
                > tolerance
            ):
                raise SimulationStateError("trace service duration mismatch")
            if (
                abs(event.queue_duration - (event.start_at - event.ready_at))
                > tolerance
            ):
                raise SimulationStateError("trace queue duration mismatch")
        for event in events:
            for predecessor in event.predecessor_ids:
                if predecessor not in by_node:
                    raise SimulationStateError(
                        f"trace missing predecessor: {predecessor}"
                    )
                if by_node[predecessor].end_at > event.ready_at + tolerance:
                    raise SimulationStateError(
                        f"trace dependency is not ready: {event.node_id}"
                    )
        validate_resource_capacity(events, self.resource_capacities)
        return by_node
