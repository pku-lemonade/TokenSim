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
        self.event_validator = None

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
            if (
                not isinstance(event.node_id, str)
                or not event.node_id
                or event.node_id in by_node
            ):
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
            tolerance = max(1e-12, 4 * max(math.ulp(value) for value in times))
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
        successors = {node: [] for node in by_node}
        degrees = {node: 0 for node in by_node}
        for event in events:
            for predecessor in event.predecessor_ids:
                parent = by_node.get(predecessor)
                end = (
                    parent.end_at
                    if parent is not None
                    else getattr(self, "completed_predecessors", {}).get(predecessor)
                )
                if end is None:
                    raise SimulationStateError(
                        f"trace missing predecessor: {predecessor}"
                    )
                tolerance = max(1e-12, 4 * math.ulp(end), 4 * math.ulp(event.ready_at))
                if end > event.ready_at + tolerance:
                    raise SimulationStateError(
                        f"trace dependency is not ready: {event.node_id}"
                    )
                if predecessor in by_node:
                    successors[predecessor].append(event.node_id)
                    degrees[event.node_id] += 1
        ready = [node for node, degree in degrees.items() if degree == 0]
        visited = 0
        while ready:
            visited += 1
            for child in successors[ready.pop()]:
                degrees[child] -= 1
                if degrees[child] == 0:
                    ready.append(child)
        if visited != len(by_node):
            raise SimulationStateError("trace dependency graph contains a cycle")
        validate_resource_capacity(events, self.resource_capacities)
        if self.event_validator is not None:
            self.event_validator(events, by_node)
        return by_node
