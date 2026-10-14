"""Per-store, per-instance counters with no media-specific result fields."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from TokenSim.errors import SimulationStateError


@dataclass
class BackendMetrics:
    backend: str
    profile: dict[str, Any]
    hit_count: int = 0
    read_operations: int = 0
    read_blocks: int = 0
    read_bytes: int = 0
    read_service_seconds: float = 0.0
    write_operations: int = 0
    write_blocks: int = 0
    write_bytes: int = 0
    write_service_seconds: float = 0.0

    def aggregate(self, other):
        if (self.backend, self.profile) != (other.backend, other.profile):
            raise SimulationStateError(
                "storage metric instances have incompatible profiles"
            )
        result = BackendMetrics(self.backend, deepcopy(self.profile))
        for item in fields(self):
            if item.name not in {"backend", "profile"}:
                setattr(
                    result,
                    item.name,
                    getattr(self, item.name) + getattr(other, item.name),
                )
        return result

    def record_io(self, operation, blocks, nbytes, service_seconds):
        if nbytes <= 0:
            return
        for suffix, value in (
            ("operations", 1),
            ("blocks", blocks),
            ("bytes", nbytes),
            ("service_seconds", service_seconds),
        ):
            name = f"{operation}_{suffix}"
            setattr(self, name, getattr(self, name) + value)


@dataclass
class StorageMetrics:
    stores: dict[str, dict[str, BackendMetrics]] = field(default_factory=dict)

    def instance(self, store, name, backend, profile):
        instances = self.stores.setdefault(store, {})
        if name not in instances:
            instances[name] = BackendMetrics(backend, deepcopy(profile))
        record = instances[name]
        if (record.backend, record.profile) != (backend, profile):
            raise SimulationStateError(
                "storage metric instance changed its backend or profile"
            )
        return record

    def aggregate(self, other):
        result = deepcopy(self)
        for store, instances in other.stores.items():
            target = result.stores.setdefault(store, {})
            for name, record in instances.items():
                target[name] = (
                    target[name].aggregate(record)
                    if name in target
                    else deepcopy(record)
                )
        return result

    def reset(self):
        result = StorageMetrics()
        for store, instances in self.stores.items():
            for name, record in instances.items():
                result.instance(store, name, record.backend, record.profile)
        return result

    def as_dict(self):
        return {
            store: {name: asdict(instances[name]) for name in sorted(instances)}
            for store, instances in sorted(self.stores.items())
        }
