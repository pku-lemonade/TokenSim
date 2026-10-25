"""Deployment-bound observations of KV bytes after runtime reservations."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from TokenSim.config.moe_execution import MoEExecutionConfig
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.experiments.fingerprints import identity


def runtime_memory_identity(model, device, parallel, **settings):
    """Bind model geometry, device, parallel layout, and explicit cache settings."""
    settings = dict(settings)
    settings["moe_execution"] = asdict(
        MoEExecutionConfig.from_value(settings.get("moe_execution"))
    )
    return {
        "model_sha256": identity(model.to_dict()),
        "device_sha256": identity(device.to_dict()),
        "parallel": parallel.to_dict(),
        "settings": settings,
    }


@dataclass(frozen=True)
class RuntimeMemoryProfile:
    deployment_identity: dict
    kv_bytes_per_rank: tuple[int, ...]
    provenance: dict
    uncertainty_bytes: int = 0
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigurationError("runtime memory profile requires schema_version 1")
        if (
            not isinstance(self.deployment_identity, dict)
            or not self.deployment_identity
        ):
            raise ConfigurationError(
                "runtime memory profile requires deployment_identity"
            )
        if (
            not isinstance(self.kv_bytes_per_rank, (list, tuple))
            or not self.kv_bytes_per_rank
        ):
            raise ConfigurationError(
                "runtime memory profile requires KV bytes for each global rank"
            )
        for value in self.kv_bytes_per_rank:
            integer(value, "runtime memory KV bytes", minimum=1)
        integer(self.uncertainty_bytes, "runtime memory uncertainty_bytes")
        if self.uncertainty_bytes >= min(self.kv_bytes_per_rank):
            raise ConfigurationError(
                "runtime memory uncertainty must be smaller than capacity"
            )
        required = (
            "source",
            "measurement_id",
            "framework",
            "framework_version",
            "graph_strategy",
        )
        if not isinstance(self.provenance, dict) or any(
            not isinstance(self.provenance.get(name), str)
            or not self.provenance[name].strip()
            for name in required
        ):
            raise ConfigurationError(f"runtime memory provenance requires {required}")
        # Copy JSON inputs so mutating a caller's mapping cannot change the profile.
        object.__setattr__(
            self,
            "deployment_identity",
            json.loads(json.dumps(self.deployment_identity, allow_nan=False)),
        )
        object.__setattr__(
            self, "provenance", json.loads(json.dumps(self.provenance, allow_nan=False))
        )
        object.__setattr__(self, "kv_bytes_per_rank", tuple(self.kv_bytes_per_rank))

    @classmethod
    def from_file(cls, path):
        try:
            value = json.loads(Path(path).read_text())
            if not isinstance(value, dict):
                raise ConfigurationError(
                    "runtime memory profile file must contain an object"
                )
            return cls.from_value(value)
        except (OSError, ValueError) as error:
            raise ConfigurationError(
                f"invalid runtime memory profile {path}: {error}"
            ) from error

    @classmethod
    def from_value(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls.from_file(value)
        if not isinstance(value, dict):
            raise ConfigurationError(
                "runtime memory profile must be a JSON object or file path"
            )
        try:
            return cls(**value)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"invalid runtime memory profile: {error}"
            ) from error

    def apply(self, budget, deployment_identity, rank, world_size):
        """Replace available KV bytes once; retain analytical deductions as diagnostics."""
        integer(rank, "runtime memory global rank")
        integer(world_size, "runtime memory world size", minimum=1)
        if identity(self.deployment_identity) != identity(deployment_identity):
            raise ConfigurationError(
                "runtime memory profile deployment identity mismatch"
            )
        if len(self.kv_bytes_per_rank) != world_size or not 0 <= rank < world_size:
            raise ConfigurationError(
                "runtime memory profile must cover every global rank"
            )
        capacity = self.kv_bytes_per_rank[rank]
        if capacity + budget["deductions"]["weights"] > budget["hardware_budget_bytes"]:
            raise ConfigurationError(
                "runtime memory KV bytes and weights exceed the device budget"
            )
        return {
            **budget,
            "analytical_available_kv_bytes": budget["available_kv_bytes"],
            "available_kv_bytes": capacity,
            "capacity_source": "runtime_observation",
            "runtime_observation": {
                "profile_sha256": identity(asdict(self)),
                "deployment_identity": self.deployment_identity,
                "provenance": self.provenance,
                "uncertainty_bytes": self.uncertainty_bytes,
                "rank": rank,
                "deductions_applied_to_observation": False,
            },
        }
