"""Opt-in unit-capacity interval validation with the general sweep retained."""

from contextlib import contextmanager
from operator import itemgetter

from TokenSim.af.trace import occupancy, recorder
from TokenSim.errors import SimulationStateError


def validate_resource_capacity(events, capacities):
    intervals = {}
    for event in events:
        end = occupancy.occupied_until(event)
        if end <= event.start_at:
            continue
        for resource in event.resource_ids or (event.resource_id,):
            node = getattr(event, "node_id", f"iteration:{event.iteration_id}")
            intervals.setdefault(resource, []).append((event.start_at, end, node))
    for resource, spans in intervals.items():
        capacity = capacities.get(resource, 1)
        if capacity == 1:
            previous_end = float("-inf")
            for start, end, node in sorted(spans, key=itemgetter(0)):
                if start < previous_end:
                    raise SimulationStateError(
                        f"AF trace invariant failed: resource={resource!r} capacity="
                        f"{capacity} exceeded at time={start} by node={node}"
                    )
                previous_end = end
        else:
            points = [
                point
                for start, end, node in spans
                for point in ((start, 1, node), (end, -1, node))
            ]
            active = 0
            for time, delta, node in sorted(points, key=itemgetter(0, 1)):
                active += delta
                if active > capacity:
                    raise SimulationStateError(
                        f"AF trace invariant failed: resource={resource!r} capacity="
                        f"{capacity} exceeded at time={time} by node={node}"
                    )


@contextmanager
def installed(output=None):
    original = occupancy.validate_resource_capacity
    imported = recorder.validate_resource_capacity
    occupancy.validate_resource_capacity = validate_resource_capacity
    recorder.validate_resource_capacity = validate_resource_capacity
    try:
        yield
    finally:
        occupancy.validate_resource_capacity = original
        recorder.validate_resource_capacity = imported
