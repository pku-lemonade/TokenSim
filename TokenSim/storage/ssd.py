"""Linear SSD media estimates and adaptation of legacy Mooncake options."""

from dataclasses import asdict, dataclass, fields

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from .interfaces import IOCost, IORequest

GIB = 1 << 30


@dataclass(frozen=True)
class SSDConfig:
    capacity_bytes: int = 0
    read_bandwidth_bytes_per_s: float = 7 * GIB
    write_bandwidth_bytes_per_s: float = 3 * GIB
    read_latency_us: float = 100.0
    write_latency_us: float = 200.0

    def __post_init__(self):
        object.__setattr__(
            self, "capacity_bytes", integer(self.capacity_bytes, "SSD capacity_bytes")
        )
        for direction in ("read", "write"):
            for suffix in ("bandwidth_bytes_per_s", "latency_us"):
                name = f"{direction}_{suffix}"
                value = number(
                    getattr(self, name),
                    f"SSD {name}",
                    positive=suffix.startswith("bandwidth"),
                )
                object.__setattr__(self, name, float(value))


class SSDBackend:
    def __init__(self, config):
        self.config = SSDConfig(**config)

    @property
    def capacity_bytes(self):
        return self.config.capacity_bytes

    def estimate_io(self, request: IORequest) -> IOCost:
        if not request.nbytes:
            return IOCost(0.0)
        direction = request.operation
        return IOCost(
            getattr(self.config, f"{direction}_latency_us") / 1e6
            + request.nbytes
            / getattr(self.config, f"{direction}_bandwidth_bytes_per_s")
        )

    def describe(self):
        return {"kind": "ssd", **asdict(self.config)}


def legacy_ssd_backend(config, block_bytes):
    capacity = (
        config.ssd_capacity_bytes
        if config.ssd_capacity_blocks is None
        else config.ssd_capacity_blocks * block_bytes
    )
    return SSDBackend(
        {
            "capacity_bytes": capacity,
            "read_bandwidth_bytes_per_s": config.ssd_read_bandwidth_bytes_per_s,
            "write_bandwidth_bytes_per_s": config.ssd_write_bandwidth_bytes_per_s,
            "read_latency_us": config.ssd_read_latency_us,
            "write_latency_us": config.ssd_write_latency_us,
        }
    )


def validate_plugin_selection(config):
    if config.ssd_capacity_blocks is not None or any(
        getattr(config, f"ssd_{item.name}") != item.default
        for item in fields(SSDConfig)
    ):
        raise ConfigurationError(
            "offload plugins and legacy SSD options cannot be combined"
        )


def legacy_ssd_profile(config):
    return {
        "tier": "ssd",
        "memory_capacity_blocks": config.memory_capacity_blocks,
        "ssd_capacity_blocks": config.ssd_capacity_blocks,
        "ssd_capacity_bytes": config.ssd_capacity_bytes,
        "ssd_read_bandwidth_bytes_per_s": config.ssd_read_bandwidth_bytes_per_s,
        "ssd_write_bandwidth_bytes_per_s": config.ssd_write_bandwidth_bytes_per_s,
        "ssd_read_latency_us": config.ssd_read_latency_us,
        "ssd_write_latency_us": config.ssd_write_latency_us,
        "write_blocking": True,
        "write_persistence": "blocking",
    }
