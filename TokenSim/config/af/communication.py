"""Af fabric."""

from __future__ import annotations

from dataclasses import InitVar
import math
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from TokenSim.config.af.common import _capture_explicit_fields
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class AFCommunicationResourceConfig:
    resource_group: str
    resource_key: tuple[str, str]
    network: str
    duplex: str
    direction: str
    capacity: int
    queue_duration_field: str = "queue_duration"
    service_duration_field: str = "service_duration"


@dataclass(frozen=True)
class AFCommunicationLaneConfig:
    """Capacity counts independent lanes, each with the declared link bandwidth.

    This is not a shared-bandwidth model: increasing capacity adds physical
    service lanes, rather than dividing one link's bandwidth between flows.
    """

    name: str
    capacity: int

    @field_validator("capacity", mode="before")
    @classmethod
    def _validate_capacity_type(cls, value: Any) -> int:
        if type(value) is not int:
            raise ConfigurationError(
                "communication fabric lane capacity must be a positive integer"
            )
        return value

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError(
                "communication fabric lane name must be a non-empty string"
            )
        if self.capacity < 1:
            raise ConfigurationError(
                f"communication fabric lane {self.name!r} capacity must be a "
                "positive integer"
            )


@dataclass(frozen=True)
class AFCommunicationFabricConfig:
    resource_group: str
    network: str
    duplex: Literal["full", "half"]
    lanes: tuple[AFCommunicationLaneConfig, ...]

    def __post_init__(self) -> None:
        for field_name in ("resource_group", "network"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"communication fabric {field_name} must be a non-empty string"
                )
        if self.duplex not in {"full", "half"}:
            raise ConfigurationError(
                f"communication fabric {self.resource_group!r} duplex must be "
                "'full' or 'half'"
            )
        if not self.lanes:
            raise ConfigurationError(
                f"communication fabric {self.resource_group!r} lanes must be non-empty"
            )
        names = [lane.name for lane in self.lanes]
        if len(names) != len(set(names)):
            raise ConfigurationError(
                f"communication fabric {self.resource_group!r} has duplicate lane names"
            )

    def lane(self, name: str) -> AFCommunicationLaneConfig:
        for lane in self.lanes:
            if lane.name == name:
                return lane
        raise ConfigurationError(
            f"communication fabric {self.resource_group!r} has no resource lane "
            f"{name!r}; available lanes={sorted(lane.name for lane in self.lanes)}"
        )

    @property
    def signature(self) -> tuple[str, str, tuple[tuple[str, int], ...]]:
        return (
            self.network,
            self.duplex,
            tuple(sorted((lane.name, lane.capacity) for lane in self.lanes)),
        )

    def communication_resources(self) -> tuple[AFCommunicationResourceConfig, ...]:
        return tuple(
            AFCommunicationResourceConfig(
                resource_group=self.resource_group,
                resource_key=(self.resource_group, lane.name),
                network=self.network,
                duplex=self.duplex,
                direction=lane.name,
                capacity=lane.capacity,
            )
            for lane in self.lanes
        )


@dataclass(frozen=True)
class AFCommunicationTimingConfig:
    enabled: bool = False
    host_submit_us: float = 5.0
    host_completion_us: float = 0.0
    host_resource_scope: str = "per_worker"
    a_step_prepare_us: float = 0.0
    f_step_prepare_us: float = 0.0
    compute_submit_us: float = 0.0
    host_receive_submit_us: float = 0.0
    native_control_enabled: bool = False
    control_encode_us: float = 0.0
    control_h2d_us: float = 0.0
    control_length_readback_us: float = 0.0
    control_payload_readback_us: float = 0.0
    control_decode_us: float = 0.0
    native_call_order_enabled: bool = False
    cost_presence_input: InitVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _capture_cost_presence(cls, value):
        names = (
            "enabled",
            "host_submit_us",
            "host_completion_us",
            "host_resource_scope",
            "a_step_prepare_us",
            "f_step_prepare_us",
            "compute_submit_us",
            "host_receive_submit_us",
            "native_control_enabled",
            "control_encode_us",
            "control_h2d_us",
            "control_length_readback_us",
            "control_payload_readback_us",
            "control_decode_us",
            "native_call_order_enabled",
        )
        return _capture_explicit_fields(
            value,
            field_names=frozenset(n for n in names if n.endswith("_us")),
            positional_names=names,
            init_var_name="cost_presence_input",
        )

    def __post_init__(self, cost_presence_input):
        object.__setattr__(
            self, "_explicit_cost_fields", frozenset(cost_presence_input)
        )

    @field_validator(
        "enabled", "native_control_enabled", "native_call_order_enabled", mode="before"
    )
    @classmethod
    def _validate_enabled(cls, value, info):
        if type(value) is not bool:
            raise ConfigurationError(
                f"communication_timing.{info.field_name} must be boolean"
            )
        return value

    @field_validator(
        "host_submit_us",
        "host_completion_us",
        "a_step_prepare_us",
        "f_step_prepare_us",
        "compute_submit_us",
        "host_receive_submit_us",
        "control_encode_us",
        "control_h2d_us",
        "control_length_readback_us",
        "control_payload_readback_us",
        "control_decode_us",
        mode="before",
    )
    @classmethod
    def _validate_duration(cls, value, info):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ConfigurationError(
                f"communication_timing.{info.field_name} must be finite and non-negative"
            )
        return float(value)

    @field_validator("host_resource_scope", mode="before")
    @classmethod
    def _validate_scope(cls, value):
        if value not in {"per_worker", "shared"}:
            raise ConfigurationError(
                "communication_timing.host_resource_scope must be 'per_worker' or 'shared'"
            )
        return value
