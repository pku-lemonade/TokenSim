"""Af link."""

from __future__ import annotations

from dataclasses import InitVar
from typing import Any

from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from TokenSim.config.af.common import (
    _AF_LINK_DTYPE_FIELDS,
    _NEW_DTYPE_FIELDS,
    _canonical_config_dtype,
    _capture_explicit_fields,
    _normalize_phases,
)
from TokenSim.config.af.communication import (
    AFCommunicationFabricConfig,
    AFCommunicationLaneConfig,
    AFCommunicationResourceConfig,
)
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class AFEffectiveDTypes:
    weight_dtype: str
    activation_dtype: str
    kv_cache_dtype: str
    a_to_f_dtype: str
    f_to_a_dtype: str
    sources: tuple[tuple[str, str], ...] = ()
    legacy_mapped: bool = False
    deprecated_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for role in _NEW_DTYPE_FIELDS:
            object.__setattr__(
                self, role, _canonical_config_dtype(getattr(self, role), role)
            )

    @property
    def source_by_role(self) -> dict[str, str]:
        return dict(self.sources)

    def as_dict(self) -> dict[str, object]:
        return {role: getattr(self, role) for role in sorted(_NEW_DTYPE_FIELDS)} | {
            "bytes_per_element": {
                role: dtype_size_bytes(getattr(self, role), role=role)
                for role in sorted(_NEW_DTYPE_FIELDS)
            },
            "source_by_role": self.source_by_role,
            "legacy_mapped": self.legacy_mapped,
            "deprecated_fields": self.deprecated_fields,
        }


@dataclass
class AFLinkConfig:
    name: str
    phases: tuple[str, ...]
    attention_group: str
    ffn_group: str
    resource_group: str
    network: str | None = None
    duplex: str | None = None
    a_to_f_capacity: int | None = None
    f_to_a_capacity: int | None = None
    half_duplex_capacity: int | None = None
    attention_domain: str | None = None
    ffn_domain: str | None = None
    colocated: bool = False
    a_to_f_dtype: str | None = None
    f_to_a_dtype: str | None = None
    dtype_presence_input: InitVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _capture_dtype_presence(cls, value: Any) -> Any:
        return _capture_explicit_fields(
            value,
            field_names=_AF_LINK_DTYPE_FIELDS,
            positional_names=(
                "name",
                "phases",
                "attention_group",
                "ffn_group",
                "resource_group",
                "network",
                "duplex",
                "a_to_f_capacity",
                "f_to_a_capacity",
                "half_duplex_capacity",
                "attention_domain",
                "ffn_domain",
                "a_to_f_dtype",
                "f_to_a_dtype",
            ),
            init_var_name="dtype_presence_input",
        )

    @field_validator(
        "a_to_f_capacity",
        "f_to_a_capacity",
        "half_duplex_capacity",
        mode="before",
    )
    @classmethod
    def _validate_capacity_type(cls, value: Any) -> int | None:
        if value is not None and type(value) is not int:
            raise ConfigurationError("AF link capacity must be a positive integer")
        return value

    def __post_init__(self, dtype_presence_input: tuple[str, ...]) -> None:
        if type(self.colocated) is not bool:
            raise ConfigurationError("AF colocated flag must be a boolean")
        self._explicit_dtype_fields = frozenset(dtype_presence_input)
        self.dtype_presence_input = tuple(sorted(self._explicit_dtype_fields))
        for field_name in (
            "name",
            "attention_group",
            "ffn_group",
            "resource_group",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"AF link {self.name!r} field {field_name} must be a non-empty string"
                )
        self.phases = _normalize_phases(self.phases, f"AF link {self.name!r} phases")
        for role in self._explicit_dtype_fields:
            setattr(self, role, _canonical_config_dtype(getattr(self, role), role))
        self._inline_fabric_declared = any(
            value is not None
            for value in (
                self.network,
                self.duplex,
                self.a_to_f_capacity,
                self.f_to_a_capacity,
                self.half_duplex_capacity,
            )
        )
        if not self._inline_fabric_declared:
            return
        if not isinstance(self.network, str) or not self.network.strip():
            raise ConfigurationError(
                f"AF link {self.name!r} inline fabric requires network"
            )
        if self.duplex not in {"full", "half"}:
            raise ConfigurationError(
                f"AF link {self.name!r} duplex must be 'full' or 'half'"
            )
        capacity_fields = (
            "a_to_f_capacity",
            "f_to_a_capacity",
            "half_duplex_capacity",
        )
        for field_name in capacity_fields:
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ConfigurationError(
                    f"AF link {self.name!r} {field_name} must be a positive integer"
                )
        if self.duplex == "full":
            if self.a_to_f_capacity is None or self.f_to_a_capacity is None:
                raise ConfigurationError(
                    f"AF link {self.name!r} full duplex requires "
                    "a_to_f_capacity and f_to_a_capacity"
                )
            if self.half_duplex_capacity is not None:
                raise ConfigurationError(
                    f"AF link {self.name!r} full duplex cannot set half_duplex_capacity"
                )
        else:
            if self.half_duplex_capacity is None:
                raise ConfigurationError(
                    f"AF link {self.name!r} half duplex requires half_duplex_capacity"
                )
            if self.a_to_f_capacity is not None or self.f_to_a_capacity is not None:
                raise ConfigurationError(
                    f"AF link {self.name!r} half duplex cannot set directional capacities"
                )

    @property
    def explicit_dtype_fields(self) -> frozenset[str]:
        return self._explicit_dtype_fields

    @property
    def has_inline_fabric(self) -> bool:
        return self._inline_fabric_declared

    def inline_fabric(self) -> AFCommunicationFabricConfig:
        if not self.has_inline_fabric:
            raise ConfigurationError(
                f"AF link {self.name!r} has no inline communication fabric"
            )
        if self.duplex == "half":
            lanes = (
                AFCommunicationLaneConfig(
                    name="shared", capacity=self.half_duplex_capacity
                ),
            )
        else:
            lanes = (
                AFCommunicationLaneConfig(name="a_to_f", capacity=self.a_to_f_capacity),
                AFCommunicationLaneConfig(name="f_to_a", capacity=self.f_to_a_capacity),
            )
        return AFCommunicationFabricConfig(
            resource_group=self.resource_group,
            network=self.network,
            duplex=self.duplex,
            lanes=lanes,
        )

    def bind_fabric(self, fabric: AFCommunicationFabricConfig) -> None:
        if fabric.resource_group != self.resource_group:
            raise ConfigurationError(
                f"AF link {self.name!r} expected resource_group "
                f"{self.resource_group!r}, got {fabric.resource_group!r}"
            )
        if self.has_inline_fabric:
            inline = self.inline_fabric()
            if inline.network != fabric.network or inline.duplex != fabric.duplex:
                raise ConfigurationError(
                    f"AF resource_group {self.resource_group!r} conflicts between "
                    f"link {self.name!r} and communication fabric registry"
                )
            for lane in inline.lanes:
                resolved = fabric.lane(lane.name)
                if resolved.capacity != lane.capacity:
                    raise ConfigurationError(
                        f"AF resource_group {self.resource_group!r} lane "
                        f"{lane.name!r} conflicts between link {self.name!r} and "
                        "communication fabric registry"
                    )
        required_lanes = (
            ("shared",) if fabric.duplex == "half" else ("a_to_f", "f_to_a")
        )
        for lane_name in required_lanes:
            fabric.lane(lane_name)
        self.network = fabric.network
        self.duplex = fabric.duplex
        if fabric.duplex == "half":
            self.half_duplex_capacity = fabric.lane("shared").capacity
            self.a_to_f_capacity = None
            self.f_to_a_capacity = None
        else:
            self.a_to_f_capacity = fabric.lane("a_to_f").capacity
            self.f_to_a_capacity = fabric.lane("f_to_a").capacity
            self.half_duplex_capacity = None

    @property
    def effective_capacities(self) -> tuple[int, ...]:
        if self.duplex is None:
            raise ConfigurationError(
                f"AF link {self.name!r} has not resolved communication fabric "
                f"{self.resource_group!r}"
            )
        if self.duplex == "half":
            return (self.half_duplex_capacity,)
        return (self.a_to_f_capacity, self.f_to_a_capacity)

    def communication_resources(self) -> tuple[AFCommunicationResourceConfig, ...]:
        if self.duplex == "half":
            return (
                AFCommunicationResourceConfig(
                    resource_group=self.resource_group,
                    resource_key=(self.resource_group, "shared"),
                    network=self.network,
                    duplex=self.duplex,
                    direction="shared",
                    capacity=self.half_duplex_capacity,
                ),
            )
        return (
            AFCommunicationResourceConfig(
                resource_group=self.resource_group,
                resource_key=(self.resource_group, "a_to_f"),
                network=self.network,
                duplex=self.duplex,
                direction="a_to_f",
                capacity=self.a_to_f_capacity,
            ),
            AFCommunicationResourceConfig(
                resource_group=self.resource_group,
                resource_key=(self.resource_group, "f_to_a"),
                network=self.network,
                duplex=self.duplex,
                direction="f_to_a",
                capacity=self.f_to_a_capacity,
            ),
        )
