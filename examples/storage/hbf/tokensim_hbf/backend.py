"""HBF parameters and operation-size bandwidth curves owned by this plugin."""

from dataclasses import asdict, dataclass
import math

from TokenSim.config.validation import integer, number
from TokenSim.storage import IOCost, IORequest

GIB = 1 << 30
RANDOM_IO_BYTES = 4 * (1 << 10)
SEQUENTIAL_IO_BYTES = 10 * (1 << 20)


@dataclass(frozen=True)
class HBFConfig:
    capacity_bytes: int
    seq_read_bandwidth_bytes_per_s: float = 128 * GIB
    seq_write_bandwidth_bytes_per_s: float = 128 * GIB
    random_4k_read_bandwidth_bytes_per_s: float | None = None
    random_4k_write_bandwidth_bytes_per_s: float | None = None
    read_latency_us: float = 12.0
    write_latency_us: float = 120.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "capacity_bytes",
            integer(self.capacity_bytes, "HBF capacity_bytes", minimum=1),
        )
        for direction in ("read", "write"):
            seq_name = f"seq_{direction}_bandwidth_bytes_per_s"
            number(getattr(self, seq_name), f"HBF {seq_name}", positive=True)
            random_name = f"random_4k_{direction}_bandwidth_bytes_per_s"
            if getattr(self, random_name) is None:
                object.__setattr__(self, random_name, 0.8 * getattr(self, seq_name))
            number(getattr(self, random_name), f"HBF {random_name}", positive=True)
            number(getattr(self, f"{direction}_latency_us"), f"HBF {direction} latency")
            for name in (seq_name, random_name, f"{direction}_latency_us"):
                object.__setattr__(self, name, float(getattr(self, name)))


class HBFBackend:
    def __init__(self, config):
        self.config = HBFConfig(**config)

    @property
    def capacity_bytes(self):
        return self.config.capacity_bytes

    def estimate_io(self, request: IORequest) -> IOCost:
        if not request.nbytes:
            return IOCost(0.0)
        direction = request.operation
        return IOCost(
            getattr(self.config, f"{direction}_latency_us") / 1e6
            + request.nbytes / self._bandwidth(request)
        )

    def _bandwidth(self, request):
        random_bw = getattr(
            self.config, f"random_4k_{request.operation}_bandwidth_bytes_per_s"
        )
        sequential_bw = getattr(
            self.config, f"seq_{request.operation}_bandwidth_bytes_per_s"
        )
        if request.access_pattern == "random":
            return float(random_bw)
        if request.access_pattern == "sequential":
            return float(sequential_bw)
        if request.nbytes <= RANDOM_IO_BYTES:
            return float(random_bw)
        if request.nbytes >= SEQUENTIAL_IO_BYTES:
            return float(sequential_bw)
        position = math.log(request.nbytes / RANDOM_IO_BYTES) / math.log(
            SEQUENTIAL_IO_BYTES / RANDOM_IO_BYTES
        )
        return float(random_bw + (sequential_bw - random_bw) * position)

    def describe(self):
        return {"kind": "hbf", **asdict(self.config)}
