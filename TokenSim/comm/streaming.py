"""Conservative output-vector geometry for a scheduled direct AllReduce.

This model describes uniform output production, not a measured LP30 schedule.
Collective launch is paid once; propagation does not occupy injection bandwidth.
"""

from dataclasses import dataclass
import math

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class StreamingConfig:
    vector_bytes: int = 320
    overlap: bool = True
    early_output: bool = True
    collective_launch_us: float | None = None

    def __post_init__(self):
        integer(self.vector_bytes, "streaming vector_bytes", minimum=1)
        for name in ("overlap", "early_output"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"streaming {name} must be a boolean")
        if self.collective_launch_us is not None:
            number(self.collective_launch_us, "streaming collective_launch_us")


@dataclass(frozen=True)
class OutputVector:
    offset_bytes: int
    size_bytes: int
    production_fraction: float
    wire_bytes_per_phase: float
    wire_seconds_per_phase: float


def output_vectors(payload_bytes, element_bytes, vector_bytes, estimate, hop_us):
    """Split a whole-buffer estimate without recharging its launch or bandwidth.

    Both direct phases use (n-1)/n of the complete output per endpoint.
    All participants must produce a vector before its reduction may begin.
    The caller must serialize both phases on the same endpoint injection resources.
    """
    integer(payload_bytes, "streaming payload_bytes", minimum=1)
    integer(element_bytes, "streaming element_bytes", minimum=1)
    integer(vector_bytes, "streaming vector_bytes", minimum=1)
    number(hop_us, "streaming hop_us")
    if payload_bytes % element_bytes or vector_bytes % element_bytes:
        raise ConfigurationError("streaming buffers must align to activation elements")
    if (
        estimate.operation != "all_reduce"
        or estimate.match_type != "analytical"
        or estimate.group_size < 2
        or len(estimate.levels) != 1
        or estimate.levels[0].algorithm != "direct"
        or estimate.levels[0].steps != 2
        or estimate.levels[0].fan != estimate.group_size
        or estimate.message_bytes != payload_bytes
    ):
        raise ConfigurationError(
            "streaming requires a single-level analytical direct AllReduce"
        )
    level = estimate.levels[0]
    wire_us = level.latency_us - 2 * hop_us
    if wire_us < -1e-10 or not math.isclose(
        estimate.latency_us,
        estimate.launch_us + level.latency_us,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ConfigurationError("streaming collective timing decomposition differs")
    phase_bytes = payload_bytes * (estimate.group_size - 1) / estimate.group_size
    if not math.isclose(level.bytes_per_step, phase_bytes, rel_tol=1e-10):
        raise ConfigurationError("streaming collective wire volume differs")
    return tuple(
        OutputVector(
            offset,
            size := min(vector_bytes, payload_bytes - offset),
            size / payload_bytes,
            phase_bytes * size / payload_bytes,
            max(0.0, wire_us) * 1e-6 * size / payload_bytes / 2,
        )
        for offset in range(0, payload_bytes, vector_bytes)
    )
