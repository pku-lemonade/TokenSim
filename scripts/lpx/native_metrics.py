"""Assemble serving metrics in consistent service and transport event order."""

from TokenSim.af.serving.metrics import ResourceMetrics, steady_metrics
from TokenSim.af.serving.timing import (
    namespace_event,
    namespace_resource,
    transport_timing,
)


def resource_capacities(services):
    return {
        namespace_resource(name, key): value
        for name, service in services.items()
        for key, value in service.executor.resources.resource_capacities().items()
    }


def collect_metrics(
    services,
    measurement,
    token_events,
    handoff_events,
    control_events,
    communication,
    windows=None,
):
    capacities = resource_capacities(services)
    transport = [
        transport_timing(row, communication)
        for row in (*handoff_events, *control_events)
    ]
    if windows is not None:
        combined = ResourceMetrics(measurement)
        for window in windows.values():
            combined.merge_disjoint(window)
        for event in transport:
            combined.observe(event)
        return (
            combined.summary(token_events, capacities=capacities, include_end=True),
            [],
        )
    traces = [
        namespace_event(name, event)
        for name, service in services.items()
        for event in service.trace.events
    ]
    return (
        steady_metrics(
            token_events,
            [*traces, *transport],
            start=measurement["start"],
            end=measurement["end"],
            capacities=capacities,
            include_end=True,
        ),
        traces,
    )
