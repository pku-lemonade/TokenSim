"""Hierarchical alpha-beta model for collective communication.

The model answers "how long does an ``operation`` over ``group_size`` ranks
laid out as :class:`GroupLayout` take for ``message_bytes``" in O(levels)
time, independent of the number of devices in the system. When a measured
``collective`` table covers the query (same operation, dtype, group size and
node count) it is used instead of the formula and the result is tagged
``measured``.

Formula (per topology level ``l`` crossed by the group, innermost first)::

    steps_l      = algorithm-dependent count of sequential transfer rounds
    bytes_l      = total payload each endpoint injects at this level
    t_l          = steps_l * alpha_l + bytes_l / (beta_l * links_l * c_l * eff(M_l))
    t_collective = launch_us + sum_l t_l

where ``alpha_l`` is the hop latency of the level's link class, ``beta_l`` its
per-direction bandwidth, ``links_l`` the number of parallel links an endpoint
can drive at that level, ``c_l`` the fraction of peak a collective reaches at
large messages (``LinkClass.collective_efficiency``), and ``eff`` the
message-size ramp ``M / (M + half_bandwidth_bytes)`` applied to the buffer
handled at the level. The parameters are fitted from nccl-tests curves; the
formula is only used when no measured ``collective`` table covers a query.

Algorithms: ``ring`` (bandwidth optimal, 2(n-1) rounds for all-reduce),
``tree`` (latency optimal, 2*log2 n rounds), ``direct`` (one round, every
rank exchanges with every other rank - the Groq software-scheduled network
and NVSwitch multicast behave this way), ``auto`` picks the faster of ring
and tree per level, matching what NCCL's tuner does in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.topology import GroupLayout, TopologySpec

OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send_recv", "broadcast")
ALGORITHMS = ("auto", "ring", "tree", "direct")

# Software launch/synchronization cost of a collective on top of link latency.
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
            raise ConfigurationError(f"unsupported collective operation {self.operation!r}")
        if self.algorithm not in ALGORITHMS:
            raise ConfigurationError(f"unsupported collective algorithm {self.algorithm!r}")

    @property
    def group_size(self) -> int:
        return self.layout.size


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


class CollectiveModel:
    def __init__(
        self,
        topology: TopologySpec,
        links: LinkCatalog,
        *,
        launch_us: float | None = None,
        family: str = "generic",
        measured_lookup: Any | None = None,
        measured_nodes_field: str = "nodes",
    ) -> None:
        topology.validate_links(links)
        self.topology = topology
        self.links = links
        self.launch_us = DEFAULT_LAUNCH_US.get(family, DEFAULT_LAUNCH_US["generic"]) if launch_us is None else launch_us
        self.family = family
        # Optional OperatorLookup with a populated ``collective`` table.
        self.measured_lookup = measured_lookup
        self.measured_nodes_field = measured_nodes_field

    # -- public API ----------------------------------------------------------

    def estimate(self, query: CollectiveQuery) -> CollectiveEstimate:
        n = query.group_size
        if n <= 1 or query.message_bytes <= 0:
            return CollectiveEstimate(0.0, query.operation, n, query.message_bytes, "analytical", "trivial")
        measured = self._measured(query)
        if measured is not None:
            return measured
        return self.analytical(query)

    def analytical(self, query: CollectiveQuery) -> CollectiveEstimate:
        layout = query.layout
        n = query.group_size
        levels: list[LevelEstimate] = []
        total = self.launch_us
        # Level l groups ``fan[l]`` participants (devices or lower-level
        # groups). Hierarchical collectives run the operation within each
        # level in turn: reduce-scatter inward, then all-gather outward; the
        # inner level handles ``1/fan_outer`` of the data on the second pass.
        remaining_share = 1.0
        for level_index in range(layout.lowest_common_level + 1):
            fan = layout.fan[level_index]
            if fan <= 1:
                continue
            level = self.topology.levels[level_index]
            link = self.links.get(level.link)
            level_bytes = query.message_bytes * remaining_share
            algorithm = self._pick_algorithm(query, level.collective_algorithm, fan, link, level_bytes)
            steps, injected = _rounds(query.operation, algorithm, fan, level_bytes)
            bandwidth = (
                link.bandwidth_per_direction_bytes_per_s
                * level.links_per_endpoint
                * link.collective_efficiency
                / level.oversubscription
            )
            transfer_us = injected / (bandwidth * link.efficiency(level_bytes)) * 1e6 if injected > 0 else 0.0
            latency_us = steps * link.latency_us + transfer_us
            levels.append(
                LevelEstimate(
                    level=level.name,
                    fan=fan,
                    algorithm=algorithm,
                    steps=steps,
                    bytes_per_step=injected / steps if steps else 0.0,
                    link=link.link_id,
                    latency_us=latency_us,
                )
            )
            total += latency_us
            if query.operation in {"all_reduce", "reduce_scatter", "all_gather"}:
                remaining_share /= fan
        return CollectiveEstimate(
            latency_us=total,
            operation=query.operation,
            group_size=n,
            message_bytes=query.message_bytes,
            match_type="analytical",
            source_id=f"analytical:collective:{self.topology.topology_id}",
            levels=tuple(levels),
            launch_us=self.launch_us,
        )

    def point_to_point(self, message_bytes: float, layout: GroupLayout) -> CollectiveEstimate:
        """One direct transfer between two ranks across their lowest common level."""
        if message_bytes <= 0 or layout.lowest_common_level < 0:
            return CollectiveEstimate(0.0, "send_recv", layout.size, message_bytes, "analytical", "trivial")
        level = self.topology.levels[layout.lowest_common_level]
        link = self.links.get(level.link)
        latency_us = link.transfer_us(message_bytes, streams=level.links_per_endpoint) / level.oversubscription
        return CollectiveEstimate(
            latency_us=latency_us,
            operation="send_recv",
            group_size=layout.size,
            message_bytes=message_bytes,
            match_type="analytical",
            source_id=f"analytical:p2p:{link.link_id}",
            levels=(LevelEstimate(level.name, 2, "direct", 1, message_bytes, link.link_id, latency_us),),
        )

    # -- internals -------------------------------------------------------------

    def _measured(self, query: CollectiveQuery) -> CollectiveEstimate | None:
        lookup = self.measured_lookup
        if lookup is None or not lookup.has_table("collective"):
            return None
        layout = query.layout
        nodes = 1
        if layout.lowest_common_level >= 1:
            nodes = int(math.prod(layout.fan[1:]))
        key = {
            "dtype": query.dtype,
            "operation": query.operation,
            "group_size": query.group_size,
            "nodes": nodes,
            "message_bytes": int(round(query.message_bytes)),
        }
        try:
            result = lookup.lookup("collective", key)
        except Exception:
            return None
        return CollectiveEstimate(
            latency_us=result.latency_us,
            operation=query.operation,
            group_size=query.group_size,
            message_bytes=query.message_bytes,
            match_type="measured" if result.match_type == "exact" else result.match_type,
            source_id=result.source_id,
            detail={"key": dict(result.key)},
        )

    def _pick_algorithm(self, query: CollectiveQuery, level_default: str, fan: int, link: LinkClass, message_bytes: float) -> str:
        algorithm = query.algorithm if query.algorithm != "auto" else level_default
        if algorithm != "auto":
            return algorithm
        if query.operation in {"all_to_all", "send_recv", "broadcast"}:
            return "direct" if query.operation != "broadcast" else "tree"
        # Latency-optimal tree for small messages, ring for large ones.
        candidates = {}
        bandwidth = link.bandwidth_per_direction_bytes_per_s * link.collective_efficiency
        for name in ("ring", "tree"):
            steps, injected = _rounds(query.operation, name, fan, message_bytes)
            transfer = injected / (bandwidth * link.efficiency(message_bytes)) * 1e6 if injected else 0.0
            candidates[name] = steps * link.latency_us + transfer
        return min(candidates, key=candidates.get)


def _rounds(operation: str, algorithm: str, n: int, message_bytes: float) -> tuple[int, float]:
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
    raise ConfigurationError(f"no round model for operation={operation!r} algorithm={algorithm!r}")
