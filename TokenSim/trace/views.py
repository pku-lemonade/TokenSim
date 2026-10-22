"""Deployment-neutral views over recorded component and module service."""

from dataclasses import replace
from collections.abc import Mapping
import math


def expand_af_events(events):
    """Expose already priced components while retaining module containers."""
    for event in events:
        external = event.details.get("external_predecessors", {})
        if external:
            event = replace(
                event, predecessor_ids=tuple(external) + event.predecessor_ids
            )
        components = event.details.get("components", ())
        fields = {"name", "latency_us", "source_id", "match_type"}
        if (
            event.event_type != "compute"
            or not components
            or not event.service_duration
            or not isinstance(components, (list, tuple))
            or any(
                not isinstance(row, Mapping) or not fields.issubset(row)
                for row in components
            )
        ):
            yield event
            continue
        total = sum(row["latency_us"] for row in components) * 1e-6
        if not math.isclose(total, event.service_duration, abs_tol=1e-12, rel_tol=1e-9):
            # An operation service profile may replace a complete stage cost.
            yield replace(event, details=dict(event.details, granularity="module"))
            continue
        yield replace(
            event,
            node_id=event.node_id + ":scope",
            resource_id=event.resource_id + ":scope",
            resource_ids=(),
            execution_group_id=None,
            details=dict(event.details, accounting_role="container"),
        )
        time = event.start_at
        previous = event.predecessor_ids
        for index, component in enumerate(components):
            duration = component["latency_us"] * 1e-6
            node = (
                event.node_id
                if index == len(components) - 1
                else f"{event.node_id}:component:{index}"
            )
            details = dict(
                component,
                parent_id=event.node_id + ":scope",
                granularity=(
                    "module" if component.get("covered_operations") else "operator"
                ),
            )
            yield replace(
                event,
                stage=component["name"],
                operation=component["name"],
                start_at=time,
                end_at=time + duration,
                ready_at=event.ready_at if index == 0 else time,
                queue_duration=event.queue_duration if index == 0 else 0,
                service_duration=duration,
                node_id=node,
                predecessor_ids=previous,
                source_id=component["source_id"],
                match_type=component["match_type"],
                execution_group_id=(
                    f"{event.execution_group_id}:component:{index}"
                    if event.execution_group_id
                    else None
                ),
                resource_queue_intervals=None,
                details=details,
            )
            time += duration
            previous = (node,)
