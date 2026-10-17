"""Compact timing history after full native trace invariant validation."""

from dataclasses import dataclass, replace
from TokenSim.af.trace.occupancy import occupied_until


@dataclass(frozen=True)
class CompletedTiming:
    iteration_id: int
    resource_id: str
    resource_ids: tuple
    ready_at: float
    start_at: float
    end_at: float
    service_duration: float
    resource_end_at: float | None = None

    @classmethod
    def from_event(cls, event):
        values = {
            key: getattr(event, key)
            for key in cls.__dataclass_fields__
            if key != "resource_end_at"
        }
        return cls(**values, resource_end_at=occupied_until(event))


def namespace_resource(service, resource):
    # Worker and host indices are local to each executor.
    # Endpoint budgets already contain globally unique physical identities.
    return (
        f"{service}/{resource}"
        if resource.startswith(("worker:", "host:"))
        else resource
    )


def namespace_event(service, event):
    return replace(
        event,
        resource_id=namespace_resource(service, event.resource_id),
        resource_ids=tuple(
            namespace_resource(service, key) for key in event.resource_ids
        ),
    )


def transport_timing(row, resources):
    keys = tuple(
        resources.format_key(key)
        for key, _ in resources.for_step(
            row["network"], "a_to_f", (row["source"], row["target"])
        )
    )
    return CompletedTiming(
        -1,
        keys[0],
        keys,
        row["ready_s"],
        row["start_s"],
        row["end_s"],
        row["end_s"] - row["start_s"],
        row.get("injection_end_s", row["end_s"]),
    )
