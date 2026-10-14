"""Separate endpoint serialization reservations from message completion."""

import math
from TokenSim.errors import SimulationStateError


def occupied_until(event):
    end = getattr(event, "resource_end_at", None)
    if end is None:
        end = getattr(event, "details", {}).get(
            "resource_occupancy_end_at", event.end_at
        )
    if not math.isfinite(end) or not event.start_at <= end <= event.end_at:
        raise SimulationStateError(
            "trace resource occupancy must lie inside event service"
        )
    return end


def validate_resource_capacity(events, capacities):
    intervals = {}
    for event in events:
        end = occupied_until(event)
        if end <= event.start_at:
            continue
        for resource in event.resource_ids or (event.resource_id,):
            intervals.setdefault(resource, []).extend(
                ((event.start_at, 1, event.node_id), (end, -1, event.node_id))
            )
    for resource, points in intervals.items():
        active = 0
        capacity = capacities.get(resource, 1)
        for time, delta, node_id in sorted(points, key=lambda item: (item[0], item[1])):
            active += delta
            if active > capacity:
                raise SimulationStateError(
                    f"AF trace invariant failed: resource={resource!r} capacity="
                    f"{capacity} exceeded at time={time} by node={node_id}"
                )
