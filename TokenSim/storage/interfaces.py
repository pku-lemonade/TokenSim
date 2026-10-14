"""Byte-based media estimates, independent of KV layouts and simulator clocks."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class IORequest:
    operation: Literal["read", "write"]
    nbytes: int
    access_pattern: Literal["auto", "random", "sequential"] = "auto"

    def __post_init__(self):
        if self.operation not in ("read", "write"):
            raise ConfigurationError("storage operation must be read or write")
        integer(self.nbytes, "storage I/O bytes")
        if self.access_pattern not in ("auto", "random", "sequential"):
            raise ConfigurationError("unsupported storage access pattern")


@dataclass(frozen=True)
class IOCost:
    service_seconds: float

    def __post_init__(self):
        number(self.service_seconds, "storage service time")


class StorageBackend(Protocol):
    """A deterministic media model constructed from backend-owned options.

    Estimates exclude network transfer, queueing and cache ownership.
    ``describe`` returns a stable JSON object including normalized defaults;
    shared stores use it to compare backend configurations.
    """

    @property
    def capacity_bytes(self) -> int: ...

    def estimate_io(self, request: IORequest) -> IOCost: ...

    def describe(self) -> dict[str, Any]: ...
