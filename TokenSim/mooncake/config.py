from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from TokenSim.errors import ConfigurationError


SUPPORTED_CONNECTORS = {
    "MooncakeConnector",
    "MooncakeStoreConnector",
    "MultiConnector",
}
SUPPORTED_MODES = {"embedded", "standalone-store"}
SUPPORTED_PROTOCOLS = {"tcp", "rdma", "nvlink", "nvmeof"}
SUPPORTED_OFFLOAD_TIERS = {"ssd"}
SUPPORTED_ADMISSION_POLICIES = {"always", "never"}
SUPPORTED_EVICTION_POLICIES = {"lru", "none"}
SUPPORTED_SAVE_POLICIES = {"mooncake", "every_step", "once"}

_GB = 1 << 30


@dataclass
class MooncakeConfig:
    mode: str = "embedded"
    protocol: str = "rdma"
    global_segment_size: int = 8 * _GB
    local_buffer_size: int = 1 * _GB
    enable_offload: bool = False
    offload_tier: str = "ssd"
    ssd_capacity_gb: float = 0.0
    ssd_read_bw_gbps: float = 7.0
    ssd_write_bw_gbps: float = 3.0
    ssd_read_latency_us: float = 100.0
    ssd_write_latency_us: float = 200.0
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
    bandwidth_gbps: float | None = None
    memory_capacity_blocks: int | None = None
    ssd_capacity_blocks: int | None = None
    staging_buffer_size: int = 256 * 1024 * 1024
    preferred_segment: str | None = None
    group_id: int = 0
    pcp_rank: int = 0
    dcp_rank: int = 0
    protocol_bandwidth_gbps: dict[str, float] = field(default_factory=dict)
    protocol_latency_us: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
        for field_name in (
            "global_segment_size",
            "local_buffer_size",
            "staging_buffer_size",
            "replica_num",
            "num_nics",
            "parallel_paths",
        ):
            value = getattr(self, field_name)
            if int(value) < 0:
                raise ConfigurationError(f"{field_name} must be non-negative")
            setattr(self, field_name, int(value))
        if self.replica_num < 1:
            raise ConfigurationError("replica_num must be at least 1")
        if self.num_nics < 1:
            raise ConfigurationError("num_nics must be at least 1")
        if self.parallel_paths < 1:
            raise ConfigurationError("parallel_paths must be at least 1")
        for field_name in (
            "ssd_read_bw_gbps",
            "ssd_write_bw_gbps",
            "ssd_read_latency_us",
            "ssd_write_latency_us",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ConfigurationError(f"{field_name} must be non-negative")
        if self.ssd_capacity_gb < 0:
            raise ConfigurationError("ssd_capacity_gb must be non-negative")
        if self.memory_capacity_blocks is not None and self.memory_capacity_blocks < 0:
            raise ConfigurationError("memory_capacity_blocks must be non-negative")
        if self.ssd_capacity_blocks is not None and self.ssd_capacity_blocks < 0:
            raise ConfigurationError("ssd_capacity_blocks must be non-negative")


def parse_mooncake_config(value: dict[str, Any] | None = None) -> MooncakeConfig:
    raw = dict(value or {})
    raw.pop("connectors", None)
    _normalize_size(raw, "global_segment_size", "global_segment_size_gb")
    _normalize_size(raw, "local_buffer_size", "local_buffer_size_gb")
    _normalize_size(raw, "staging_buffer_size", "staging_buffer_size_gb")
    return MooncakeConfig(**raw)


def validate_mooncake_connector_config(
    connector_name: str | None,
    extra_config: dict[str, Any] | None,
) -> None:
    if connector_name not in SUPPORTED_CONNECTORS:
        return
    extra_config = dict(extra_config or {})
    if connector_name == "MultiConnector":
        children = extra_config.get("connectors", [])
        if not isinstance(children, list):
            raise ConfigurationError("MultiConnector connectors must be a list")
        for child in children:
            if not isinstance(child, dict):
                raise ConfigurationError("MultiConnector child config must be an object")
            validate_mooncake_connector_config(
                child.get("kv_connector"),
                child.get("kv_connector_extra_config", {}),
            )
        return
    parse_mooncake_config(extra_config)


def _normalize_size(raw: dict[str, Any], bytes_key: str, gb_key: str) -> None:
    if bytes_key in raw:
        return
    if gb_key in raw:
        raw[bytes_key] = int(float(raw.pop(gb_key)) * _GB)
