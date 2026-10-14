"""The generic storage plugin reference; media options belong to the plugin."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    config: dict[str, Any] = field(default_factory=dict)
    name: str = "offload"

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ConfigurationError("offload.backend must be a nonempty string")
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name == "memory"
        ):
            raise ConfigurationError(
                "offload.name must be nonempty and differ from memory"
            )
        if not isinstance(self.config, dict):
            raise ConfigurationError("offload.config must be an object")
        object.__setattr__(self, "config", deepcopy(self.config))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "BackendSpec") -> "BackendSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ConfigurationError("offload must be an object")
        try:
            return cls(**dict(value))
        except TypeError as error:
            raise ConfigurationError(f"invalid offload reference: {error}") from error
