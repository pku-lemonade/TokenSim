"""Node aggregate communication budgets, independent of physical port graphs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass as record

from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass, Field

from TokenSim.errors import ConfigurationError

MODEL_VERSION = "node-domain-fifo-v1"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CommunicationDomainConfig:
    node_size: int = 8
    node_count: int | None = None
    local_bandwidth_bytes_per_s: float = 300e9
    inter_tx_bandwidth_bytes_per_s: float = 50e9
    inter_rx_bandwidth_bytes_per_s: float = 50e9
    local_latency_us: float = 0.0
    inter_latency_us: float = 0.0
    placement_strategy: str = "manual"
    mixed_a_slots: int = 2
    # Physical worker identity -> flat GPU index. Overrides reserve slots first.
    device_indices: dict[str, int] = Field(default_factory=dict)
    source_id: str = "assumption:node-aggregate"

    @field_validator("node_size", "node_count", "mixed_a_slots", mode="before")
    @classmethod
    def integer_fields(cls, value, info):
        if value is None and info.field_name == "node_count":
            return value
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"communication_domains.{info.field_name} must be a positive integer")
        return value

    @field_validator(
        "local_bandwidth_bytes_per_s", "inter_tx_bandwidth_bytes_per_s",
        "inter_rx_bandwidth_bytes_per_s", "local_latency_us", "inter_latency_us",
        mode="before",
    )
    @classmethod
    def finite_numbers(cls, value, info):
        if "bandwidth" in info.field_name and isinstance(value, str):
            match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(Gb/s|GB/s|Mb/s|MB/s|bit/s|bytes/s)\s*", value)
            if not match:
                raise ConfigurationError("bandwidth string requires explicit bit/byte units")
            scales = {"Gb/s": 1e9/8, "GB/s": 1e9, "Mb/s": 1e6/8,
                      "MB/s": 1e6, "bit/s": 1/8, "bytes/s": 1}
            value = float(match[1]) * scales[match[2]]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigurationError(f"communication_domains.{info.field_name} must be finite")
        if value < 0 or ("bandwidth" in info.field_name and value == 0):
            raise ConfigurationError(f"invalid communication_domains.{info.field_name}")
        return value

    @field_validator("device_indices", mode="before")
    @classmethod
    def valid_indices(cls, value):
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or type(index) is not int or index < 0
            for key, index in value.items()
        ):
            raise ConfigurationError("communication_domains.device_indices requires identity -> nonnegative integer")
        if len(set(value.values())) != len(value):
            raise ConfigurationError("communication domain overrides duplicate a GPU slot")
        return value

    def __post_init__(self):
        if self.placement_strategy not in {"manual", "dedicated", "mixed"}:
            raise ConfigurationError("communication domain placement_strategy must be manual, dedicated or mixed")
        if self.placement_strategy == "mixed" and self.mixed_a_slots >= self.node_size:
            raise ConfigurationError("mixed_a_slots must leave at least one F slot")
        if self.placement_strategy != "manual" and self.node_count is None:
            raise ConfigurationError("automatic domain placement requires explicit node_count inventory")


@record(frozen=True, order=True)
class DomainResource:
    node_id: int
    kind: str

    @property
    def identity(self):
        return f"domain:node{self.node_id}:{self.kind}"


def transfer_resources(source_node: int, target_node: int) -> tuple[DomainResource, ...]:
    if source_node == target_node:
        return (DomainResource(source_node, "local"),)
    return (DomainResource(source_node, "inter_tx"), DomainResource(target_node, "inter_rx"))


def transmission_service(config: CommunicationDomainConfig, source_node: int,
                         target_node: int, wire_bytes: float) -> float:
    if wire_bytes == 0:
        return 0.0
    if not math.isfinite(wire_bytes) or wire_bytes < 0:
        raise ConfigurationError("wire bytes must be finite and nonnegative")
    if source_node == target_node:
        return config.local_latency_us * 1e-6 + wire_bytes / config.local_bandwidth_bytes_per_s
    return config.inter_latency_us * 1e-6 + wire_bytes / min(
        config.inter_tx_bandwidth_bytes_per_s, config.inter_rx_bandwidth_bytes_per_s)
