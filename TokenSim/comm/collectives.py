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
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.operator_data.lookup import FALLBACK_POLICIES, MissingOperatorDataError

OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send_recv", "broadcast")
ALGORITHMS = ("auto", "ring", "tree", "direct")

# Analytical scaling of the plain all-to-all estimate per ParallelConfig.all2all_backend
# (project assumption, grade D; inherited from the legacy roofline backend).
# Measured DeepEP tables replace these factors whenever they cover the query.
EP_ALL2ALL_MODE_SCALE = {
    "naive": 1.5,
    "allgather_reducescatter": 1.0,
    "deepep_high_throughput": 0.7,
    "deepep_low_latency": 0.5,
}

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
class EPAllToAllQuery:
    """One expert-parallel dispatch+combine pair for a MoE layer on one rank."""

    layout: GroupLayout
    num_tokens: int          # tokens this rank dispatches (before top-k expansion)
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
        return float(self.num_tokens) * self.top_k * self.hidden_size * self.activation_bytes


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
    """Prices collectives from measured tables first, the alpha-beta model otherwise.

    ``fallback`` follows the operator backend's policy: ``table_first`` uses
    the formula when no measured row answers a query and records the miss in
    ``missing_report``; ``table_only`` raises :class:`MissingOperatorDataError`
    instead, so a run can never silently rest on communication formulas;
    ``analytical_only`` never consults the tables.
    """

    def __init__(
        self,
        topology: TopologySpec,
        links: LinkCatalog,
        *,
        launch_us: float | None = None,
        family: str = "generic",
        measured_lookup: Any | None = None,
        measured_nodes_field: str = "nodes",
        fallback: str = "table_first",
        missing_report: MissingShapeReport | None = None,
    ) -> None:
        topology.validate_links(links)
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(f"fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}")
        self.topology = topology
        self.links = links
        self.launch_us = DEFAULT_LAUNCH_US.get(family, DEFAULT_LAUNCH_US["generic"]) if launch_us is None else launch_us
        self.family = family
        # Optional OperatorLookup with ``collective`` and/or ``ep_all2all`` tables.
        self.measured_lookup = measured_lookup if fallback != "analytical_only" else None
        self.measured_nodes_field = measured_nodes_field
        self.fallback = fallback
        self.missing_report = missing_report

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

    def ep_all2all(self, query: EPAllToAllQuery) -> CollectiveEstimate:
        """Dispatch + combine of one MoE layer, measured DeepEP tables first.

        The ``ep_all2all`` table is consulted for the requested ``mode`` (and, for
        the two DeepEP modes, the other DeepEP mode as a fallback). Without a
        measured row the pair is priced as two analytical all-to-all collectives
        scaled by :data:`EP_ALL2ALL_MODE_SCALE`.
        """
        n = query.ep_size
        if n <= 1 or query.num_tokens <= 0:
            return CollectiveEstimate(0.0, "ep_all2all", n, 0.0, "analytical", "trivial")
        measured = self._measured_ep_all2all(query)
        if measured is not None:
            return measured
        payload = query.payload_bytes
        if not query.mode.startswith("deepep"):
            # naive / allgather_reducescatter are priced by formula by design;
            # still surface it so a critical path on formulas is never silent.
            self._miss(
                MissingOperatorDataError(
                    "ep_all2all",
                    self._ep_all2all_key(query, query.mode, "dispatch"),
                    f"all2all_backend {query.mode!r} has no measured ep_all2all table; "
                    "use a deepep_* mode for measured dispatch/combine",
                    kind="no_measured_mode",
                )
            )
        base = self.analytical(CollectiveQuery("all_to_all", payload, query.layout, query.dtype))
        scale = EP_ALL2ALL_MODE_SCALE.get(query.mode, 1.0)
        return CollectiveEstimate(
            latency_us=2.0 * base.latency_us * scale,
            operation="ep_all2all",
            group_size=n,
            message_bytes=payload,
            match_type="analytical",
            source_id=f"analytical:ep_all2all:{query.mode}:{self.topology.topology_id}",
            levels=base.levels,
            launch_us=2.0 * base.launch_us,
            detail={"mode": query.mode, "mode_scale": scale, "phases": ("dispatch", "combine")},
        )

    def _nodes(self, layout: GroupLayout) -> int:
        return int(math.prod(layout.fan[1:])) if layout.lowest_common_level >= 1 else 1

    def _ep_all2all_key(self, query: EPAllToAllQuery, mode: str, phase: str) -> dict[str, Any]:
        return {
            "dtype": query.dtype,
            "phase": phase,
            "mode": mode,
            "ep_size": query.ep_size,
            "nodes": self._nodes(query.layout),
            "hidden_size": int(query.hidden_size),
            "top_k": int(query.top_k),
            "num_experts": int(query.num_experts),
            "num_tokens": int(query.num_tokens),
        }

    def _miss(self, error: MissingOperatorDataError) -> None:
        """Apply the fallback policy to a query no measured row answers."""
        if self.fallback == "table_only":
            raise error
        if self.fallback == "table_first" and self.missing_report is not None:
            self.missing_report.record_error(error)

    def _measured_ep_all2all(self, query: EPAllToAllQuery) -> CollectiveEstimate | None:
        if not query.mode.startswith("deepep"):
            return None
        lookup = self.measured_lookup
        if lookup is None or not lookup.has_table("ep_all2all"):
            self._miss(
                MissingOperatorDataError(
                    "ep_all2all",
                    self._ep_all2all_key(query, query.mode, "dispatch"),
                    "table has no rows in this package",
                    kind="table_absent",
                )
            )
            return None
        layout = query.layout
        nodes = self._nodes(layout)
        # The other DeepEP kernel family stands in when the requested one has no rows.
        modes = [query.mode] + [m for m in ("deepep_high_throughput", "deepep_low_latency") if m != query.mode]
        last_error: MissingOperatorDataError | None = None
        for mode in modes:
            total = 0.0
            sources: list[str] = []
            match_types: list[str] = []
            phases_ok = True
            for phase in ("dispatch", "combine"):
                key = self._ep_all2all_key(query, mode, phase)

                def growth_reference(_table: str, k: Mapping[str, Any]) -> float | None:
                    probe = EPAllToAllQuery(
                        layout, int(k["num_tokens"]), int(k["hidden_size"]), query.top_k,
                        query.num_experts, query.dtype, mode, query.activation_bytes,
                    )
                    base = self.analytical(CollectiveQuery("all_to_all", probe.payload_bytes, layout, query.dtype))
                    return base.latency_us

                try:
                    result = lookup.lookup("ep_all2all", key, scaler=growth_reference)
                except MissingOperatorDataError as exc:
                    if mode == query.mode:
                        last_error = exc
                    phases_ok = False
                    break
                total += result.latency_us
                sources.append(result.source_id)
                match_types.append(result.match_type)
            if not phases_ok:
                continue
            if all(m == "exact" for m in match_types):
                match_type = "measured"
            elif "extrapolated" in match_types:
                match_type = "extrapolated"
            else:
                match_type = "interpolated"
            return CollectiveEstimate(
                latency_us=total,
                operation="ep_all2all",
                group_size=query.ep_size,
                message_bytes=query.payload_bytes,
                match_type=match_type,
                source_id="+".join(dict.fromkeys(sources)),
                detail={"mode": mode, "requested_mode": query.mode, "nodes": nodes, "phases": ("dispatch", "combine")},
            )
        if last_error is not None:
            self._miss(last_error)
        return None

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
        layout = query.layout
        key = {
            "dtype": query.dtype,
            "operation": query.operation,
            "group_size": query.group_size,
            "nodes": self._nodes(layout),
            "message_bytes": int(round(query.message_bytes)),
        }
        lookup = self.measured_lookup
        if lookup is None or not lookup.has_table("collective"):
            self._miss(
                MissingOperatorDataError(
                    "collective", key, "table has no rows in this package", kind="table_absent"
                )
            )
            return None

        def growth_reference(_table: str, k: Mapping[str, Any]) -> float | None:
            # Past the measured message-size range keep the boundary point's
            # efficiency and follow the alpha-beta model's growth.
            probe = CollectiveQuery(
                query.operation, float(k["message_bytes"]), layout, query.dtype, query.algorithm
            )
            return self.analytical(probe).latency_us

        try:
            result = lookup.lookup("collective", key, scaler=growth_reference)
        except MissingOperatorDataError as exc:
            self._miss(exc)
            return None
        return CollectiveEstimate(
            latency_us=result.latency_us,
            operation=query.operation,
            group_size=query.group_size,
            message_bytes=query.message_bytes,
            match_type="measured" if result.match_type == "exact" else result.match_type,
            source_id=result.source_id,
            detail={"key": dict(result.key), "flags": list(result.detail.get("flags", []))},
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
