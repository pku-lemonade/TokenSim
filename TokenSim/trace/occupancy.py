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


def resource_queue_intervals(event):
    intervals = getattr(event, "resource_queue_intervals", None)
    if intervals is None:
        return ((event.ready_at, event.start_at),)
    previous = -math.inf
    for start, end in intervals:
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and previous <= start <= end <= event.start_at
        ):
            raise SimulationStateError("invalid physical resource queue intervals")
        previous = end
    return intervals


def unique_executions(events):
    """Count one physical reservation for shared F batch member spans."""
    seen = {}
    for event in events:
        group = getattr(event, "execution_group_id", None)
        if group is not None:
            signature = (
                event.start_at,
                event.end_at,
                occupied_until(event),
                event.resource_ids or (event.resource_id,),
                resource_queue_intervals(event),
            )
            if group in seen:
                if seen[group] != signature:
                    raise SimulationStateError(
                        "shared F execution has inconsistent resource intervals"
                    )
                continue
            seen[group] = signature
        yield event


def validate_resource_capacity(events, capacities):
    intervals = {}
    for event in unique_executions(events):
        end = occupied_until(event)
        if end <= event.start_at:
            continue
        for resource in event.resource_ids or (event.resource_id,):
            node = getattr(event, "node_id", f"iteration:{event.iteration_id}")
            intervals.setdefault(resource, []).extend(
                ((event.start_at, 1, node), (end, -1, node))
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
