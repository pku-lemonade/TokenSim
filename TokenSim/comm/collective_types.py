"""Collective queries, estimates, traffic factors and algorithm rounds."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.topology import GroupLayout


OPERATIONS = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "send_recv",
    "broadcast",
)

ALGORITHMS = ("auto", "ring", "tree", "direct")

EP_ALL2ALL_MODE_SCALE = {
    "naive": 1.5,
    "allgather_reducescatter": 1.0,
    "deepep_high_throughput": 0.7,
    "deepep_low_latency": 0.5,
}

DEFAULT_LAUNCH_US = {"nvidia_gpu": 8.0, "groq_tsp": 0.5, "generic": 10.0}


def ring_traffic_factor(operation: str, n: int) -> float:
    """Bytes moved per rank relative to the buffer size (nccl-tests busbw factors)."""
    if n <= 1:
        return 0.0
    if operation == "all_reduce":
        return 2.0 * (n - 1) / n
    if operation in {"all_gather", "reduce_scatter", "all_to_all"}:
        return (n - 1) / n
    if operation in {"send_recv", "broadcast"}:
        return 1.0
    raise ConfigurationError(f"unsupported collective operation {operation!r}")


@dataclass(frozen=True)
class CollectiveQuery:
    operation: str
    message_bytes: float
    layout: GroupLayout
    dtype: str = "fp16"
    algorithm: str = "auto"

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ConfigurationError(
                f"unsupported collective operation {self.operation!r}"
            )
        if self.algorithm not in ALGORITHMS:
            raise ConfigurationError(
                f"unsupported collective algorithm {self.algorithm!r}"
            )

    @property
    def group_size(self) -> int:
        return self.layout.size


@dataclass(frozen=True)
class EPAllToAllQuery:
    """One expert-parallel dispatch+combine pair for a MoE layer on one rank."""

    layout: GroupLayout
    num_tokens: int  # tokens this rank dispatches (before top-k expansion)
    hidden_size: int
    top_k: int
    num_experts: int
    dtype: str = "bf16"
    mode: str = "allgather_reducescatter"
    activation_bytes: float = 2.0

    @property
    def ep_size(self) -> int:
        return self.layout.size

    @property
    def payload_bytes(self) -> float:
        """Bytes one rank sends in one direction: every token goes to top_k experts."""
        return (
            float(self.num_tokens)
            * self.top_k
            * self.hidden_size
            * self.activation_bytes
        )


@dataclass(frozen=True)
class LevelEstimate:
    level: str
    fan: int
    algorithm: str
    steps: int
    bytes_per_step: float
    link: str
    latency_us: float


@dataclass(frozen=True)
class CollectiveEstimate:
    latency_us: float
    operation: str
    group_size: int
    message_bytes: float
    match_type: str  # measured | interpolated | analytical
    source_id: str
    levels: tuple[LevelEstimate, ...] = ()
    launch_us: float = 0.0
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def latency_s(self) -> float:
        return self.latency_us * 1e-6

    @property
    def bottleneck_level(self) -> str | None:
        if not self.levels:
            return None
        return max(self.levels, key=lambda item: item.latency_us).level


def _rounds(
    operation: str, algorithm: str, n: int, message_bytes: float
) -> tuple[int, float]:
    """Return (sequential rounds, total bytes injected per endpoint)."""
    if n <= 1 or message_bytes <= 0:
        return 0, 0.0
    ring_bytes = ring_traffic_factor(operation, n) * message_bytes
    log_steps = max(1, math.ceil(math.log2(n)))
    if operation == "all_reduce":
        if algorithm == "ring":
            return 2 * (n - 1), ring_bytes
        if algorithm == "tree":
            # reduce then broadcast along a binary tree: log n rounds each,
            # every rank forwards the full buffer once per phase
            return 2 * log_steps, 2.0 * message_bytes
        if algorithm == "direct":
            # one-shot reduce-scatter and all-gather over a non-blocking fabric
            return 2, ring_bytes
    elif operation in {"all_gather", "reduce_scatter"}:
        if algorithm == "ring":
            return n - 1, ring_bytes
        if algorithm == "tree":
            return log_steps, message_bytes
        if algorithm == "direct":
            return 1, ring_bytes
    elif operation == "all_to_all":
        if algorithm == "ring":
            return n - 1, ring_bytes
        return 1, ring_bytes
    elif operation == "send_recv":
        return 1, message_bytes
    elif operation == "broadcast":
        if algorithm == "tree":
            return log_steps, message_bytes
        return n - 1, message_bytes
    raise ConfigurationError(
        f"no round model for operation={operation!r} algorithm={algorithm!r}"
    )
