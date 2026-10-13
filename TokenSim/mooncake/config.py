from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from numbers import Integral, Real
import math

from TokenSim.errors import ConfigurationError


SUPPORTED_CONNECTORS = {
    "MooncakeConnector",
    "MooncakeStoreConnector",
    "MultiConnector",
}
SUPPORTED_MODES = {"embedded", "standalone-store"}
SUPPORTED_PROTOCOLS = {"tcp", "rdma", "nvlink", "nvmeof"}
SUPPORTED_OFFLOAD_TIERS = {"ssd", "hbf"}
SUPPORTED_ADMISSION_POLICIES = {"always", "never"}
SUPPORTED_EVICTION_POLICIES = {"lru", "none"}
SUPPORTED_SAVE_POLICIES = {"mooncake", "every_step", "once"}

_GB = 1 << 30


@dataclass
class MooncakeConfig:
    master_server_address: str = "default"
    tenant_id: str = "default"
    cache_prefix: str = ""
    cache_layout: str = "uniform"
    mode: str = "embedded"
    protocol: str = "rdma"
    global_segment_size: int = 8 * _GB
    local_buffer_size: int = 1 * _GB
    enable_offload: bool = False
    offload_tier: str = "ssd"
    ssd_capacity_bytes: int = 0
    ssd_read_bandwidth_bytes_per_s: float = 7 * _GB
    ssd_write_bandwidth_bytes_per_s: float = 3 * _GB
    ssd_read_latency_us: float = 100.0
    ssd_write_latency_us: float = 200.0
    hbf_capacity_bytes: int = 0
    hbf_seq_read_bandwidth_bytes_per_s: float = 128 * _GB
    hbf_seq_write_bandwidth_bytes_per_s: float = 128 * _GB
    hbf_random_4k_read_bandwidth_bytes_per_s: float | None = None
    hbf_random_4k_write_bandwidth_bytes_per_s: float | None = None
    hbf_read_latency_us: float = 12.0
    hbf_write_latency_us: float = 120.0
    replica_num: int = 1
    admission_policy: str = "always"
    eviction_policy: str = "lru"
    # "every_step" re-saves a request's prompt blocks on every scheduled step
    # (historical behavior); "once" saves each request a single time.
    save_policy: str = "mooncake"
    load_async: bool = False
    transfer_overlap: bool = False
    num_nics: int = 1
    parallel_paths: int = 1
    fixed_latency_us: float | None = None
    bandwidth_bytes_per_s: float | None = None
    memory_capacity_blocks: int | None = None
    ssd_capacity_blocks: int | None = None
    hbf_capacity_blocks: int | None = None
    preferred_segment: str | None = None
    group_id: int = 0
    pcp_rank: int = 0
    dcp_rank: int = 0
    # Aggregate host memory available to the store and staging pools.
    host_available_bytes: int | None = None
    transfer_utilization: float = 1.0
    protocol_bandwidth_bytes_per_s: dict[str, float] = field(default_factory=dict)
    protocol_latency_us: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("master_server_address", "tenant_id", "cache_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str) or (name != "cache_prefix" and not value):
                raise ConfigurationError(
                    f"{name} must be a valid store/namespace string"
                )
        if self.cache_layout not in {"uniform", "grouped"}:
            raise ConfigurationError("cache_layout must be uniform or grouped")
        if self.mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"unsupported Mooncake mode {self.mode!r}; expected one of "
                + f"{sorted(SUPPORTED_MODES)}"
            )
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ConfigurationError(
                f"unsupported Mooncake protocol {self.protocol!r}; expected one of "
                + f"{sorted(SUPPORTED_PROTOCOLS)}"
            )
        if self.offload_tier not in SUPPORTED_OFFLOAD_TIERS:
            raise ConfigurationError(
                f"unsupported Mooncake offload_tier {self.offload_tier!r}; "
                + f"expected one of {sorted(SUPPORTED_OFFLOAD_TIERS)}"
            )
        if self.admission_policy not in SUPPORTED_ADMISSION_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake admission_policy "
                + f"{self.admission_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_ADMISSION_POLICIES)}"
            )
        if self.eviction_policy not in SUPPORTED_EVICTION_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake eviction_policy "
                + f"{self.eviction_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_EVICTION_POLICIES)}"
            )
        if self.save_policy not in SUPPORTED_SAVE_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake save_policy "
                + f"{self.save_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_SAVE_POLICIES)}"
            )
        for name in (
            "global_segment_size",
            "local_buffer_size",
            "replica_num",
            "num_nics",
            "parallel_paths",
            "group_id",
            "pcp_rank",
            "dcp_rank",
            "ssd_capacity_bytes",
            "hbf_capacity_bytes",
        ):
            value = getattr(self, name)
            minimum = 1 if name in {"replica_num", "num_nics", "parallel_paths"} else 0
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
            setattr(self, name, int(value))
        for name in ("enable_offload", "load_async", "transfer_overlap"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"{name} must be a boolean")
        if self.transfer_overlap and not self.load_async:
            raise ConfigurationError("transfer_overlap requires load_async")
        for direction in ("read", "write"):
            seq_name = f"hbf_seq_{direction}_bandwidth_bytes_per_s"
            _finite_number(getattr(self, seq_name), seq_name, positive=True)
            name = f"hbf_random_4k_{direction}_bandwidth_bytes_per_s"
            if getattr(self, name) is None:
                setattr(
                    self,
                    name,
                    0.8 * getattr(self, seq_name),
                )
        for name in (
            "ssd_read_bandwidth_bytes_per_s",
            "ssd_write_bandwidth_bytes_per_s",
            "ssd_read_latency_us",
            "ssd_write_latency_us",
            "hbf_seq_read_bandwidth_bytes_per_s",
            "hbf_seq_write_bandwidth_bytes_per_s",
            "hbf_random_4k_read_bandwidth_bytes_per_s",
            "hbf_random_4k_write_bandwidth_bytes_per_s",
            "hbf_read_latency_us",
            "hbf_write_latency_us",
        ):
            _finite_number(getattr(self, name), name, positive="bandwidth" in name)
        for name in (
            "memory_capacity_blocks",
            "ssd_capacity_blocks",
            "hbf_capacity_blocks",
            "host_available_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < (1 if name == "host_available_bytes" else 0)
            ):
                raise ConfigurationError(
                    f"{name} must be an integer with a valid byte/block budget"
                )
        for name in ("fixed_latency_us", "bandwidth_bytes_per_s"):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name, positive="bandwidth" in name)
        for name in ("protocol_bandwidth_bytes_per_s", "protocol_latency_us"):
            for protocol, value in getattr(self, name).items():
                if protocol not in SUPPORTED_PROTOCOLS:
                    raise ConfigurationError(f"unknown protocol {protocol!r} in {name}")
                _finite_number(value, name, positive="bandwidth" in name)
        _finite_number(self.transfer_utilization, "transfer_utilization", positive=True)
        if self.transfer_utilization > 1:
            raise ConfigurationError("transfer_utilization must be in (0, 1]")
        capacity_blocks = getattr(self, f"{self.offload_tier}_capacity_blocks")
        capacity_bytes = getattr(self, f"{self.offload_tier}_capacity_bytes")
        if self.enable_offload and (
            capacity_blocks == 0 or capacity_blocks is None and capacity_bytes == 0
        ):
            raise ConfigurationError(
                f"enabled {self.offload_tier.upper()} offload needs a positive capacity"
            )


def parse_mooncake_config(value: dict[str, Any] | None = None) -> MooncakeConfig:
    raw = dict(value or {})
    _normalize_size(raw, "global_segment_size", "global_segment_size_gb")
    _normalize_size(raw, "local_buffer_size", "local_buffer_size_gb")
    try:
        return MooncakeConfig(**raw)
    except (TypeError, ValueError, OverflowError, AttributeError) as error:
        raise ConfigurationError(f"invalid Mooncake configuration: {error}") from error


def validate_mooncake_connector_config(
    connector_name: str | None,
    extra_config: dict[str, Any] | None,
) -> None:
    if connector_name not in SUPPORTED_CONNECTORS:
        return
    extra_config = dict(extra_config or {})
    if connector_name == "MultiConnector":
        if set(extra_config) - {"connectors"}:
            raise ConfigurationError(
                "MultiConnector options belong to its child connectors"
            )
        children = extra_config.get("connectors", [])
        if not isinstance(children, list):
            raise ConfigurationError("MultiConnector connectors must be a list")
        for child in children:
            if not isinstance(child, dict):
                raise ConfigurationError(
                    "MultiConnector child config must be an object"
                )
            validate_mooncake_connector_config(
                child.get("kv_connector"),
                child.get("kv_connector_extra_config", {}),
            )
        return
    parse_mooncake_config(extra_config)


def _normalize_size(raw: dict[str, Any], bytes_key: str, gb_key: str) -> None:
    if bytes_key in raw:
        if gb_key in raw:
            raise ConfigurationError(f"choose either {bytes_key} or {gb_key}, not both")
        return
    if gb_key in raw:
        value = raw.pop(gb_key)
        _finite_number(value, gb_key)
        raw[bytes_key] = int(value * _GB)


def _finite_number(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or positive
        and value == 0
    ):
        raise ConfigurationError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
