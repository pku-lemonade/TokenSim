from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from TokenSim.comm.collectives import CollectiveEstimate, CollectiveModel, CollectiveQuery
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.hardware.topology import TopologyPlacement


@dataclass
class ParallelSyncEvent:
    kind: str
    source_worker_id: int
    target_worker_id: int | None
    dp_rank: int
    tp_rank: int
    pp_rank: int
    bytes: int
    latency: float
    link_type: str
    group_size: int = 1
    match_type: str = "analytical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_worker_id": self.source_worker_id,
            "target_worker_id": self.target_worker_id,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "bytes": self.bytes,
            "latency": self.latency,
            "link_type": self.link_type,
            "group_size": self.group_size,
            "match_type": self.match_type,
        }


@dataclass
class ParallelStats:
    latency_total: float = 0.0
    tp_collective_latency: float = 0.0
    pp_transfer_latency: float = 0.0
    ep_all2all_latency: float = 0.0
    sync_event_count: int = 0
    tp_shard_event_count: int = 0
    # Aggregated in place of a per-event list: one entry per step used to grow
    # RSS unboundedly while the export only needs totals.
    link_type_counts: dict[str, int] = field(default_factory=dict)
    match_type_counts: dict[str, int] = field(default_factory=dict)

    def record_event(self, event: ParallelSyncEvent) -> None:
        self.sync_event_count += 1
        self.latency_total += event.latency
        self.link_type_counts[event.link_type] = self.link_type_counts.get(event.link_type, 0) + 1
        self.match_type_counts[event.match_type] = (
            self.match_type_counts.get(event.match_type, 0) + 1
        )
        if event.kind == "tp_collective":
            self.tp_collective_latency += event.latency
        elif event.kind == "pp_stage_transfer":
            self.pp_transfer_latency += event.latency
        elif event.kind == "ep_all2all":
            self.ep_all2all_latency += event.latency

    def record_tp_shard(self) -> None:
        self.tp_shard_event_count += 1

    def aggregate(self, other: "ParallelStats") -> "ParallelStats":
        link_type_counts = dict(self.link_type_counts)
        for link_type, count in other.link_type_counts.items():
            link_type_counts[link_type] = link_type_counts.get(link_type, 0) + count
        match_type_counts = dict(self.match_type_counts)
        for match_type, count in other.match_type_counts.items():
            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + count
        return ParallelStats(
            latency_total=self.latency_total + other.latency_total,
            tp_collective_latency=self.tp_collective_latency + other.tp_collective_latency,
            pp_transfer_latency=self.pp_transfer_latency + other.pp_transfer_latency,
            ep_all2all_latency=self.ep_all2all_latency + other.ep_all2all_latency,
            sync_event_count=self.sync_event_count + other.sync_event_count,
            tp_shard_event_count=self.tp_shard_event_count + other.tp_shard_event_count,
            link_type_counts=link_type_counts,
            match_type_counts=match_type_counts,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parallel_latency_total": self.latency_total,
            "parallel_tp_collective_latency": self.tp_collective_latency,
            "parallel_pp_transfer_latency": self.pp_transfer_latency,
            "parallel_ep_all2all_latency": self.ep_all2all_latency,
            "parallel_sync_event_count": self.sync_event_count,
            "parallel_tp_shard_event_count": self.tp_shard_event_count,
            "parallel_link_type_counts": dict(self.link_type_counts),
            "parallel_comm_match_type_counts": dict(self.match_type_counts),
        }


class ParallelCommunicator:
    """Estimates collective and point-to-point latency for one worker.

    Groups are derived from the rank coordinates of the engine's workers; the
    cost of a collective comes from :class:`CollectiveModel`, which consults
    measured tables first and the hierarchical alpha-beta model otherwise.
    """

    def __init__(
        self,
        *,
        placement: TopologyPlacement,
        collective_model: CollectiveModel,
        workers: Sequence[Any] | None,
        worker_id: int,
        rank_info: ParallelRankInfo,
        parallel_config: ParallelConfig,
        dtype: str = "fp16",
    ) -> None:
        self.placement = placement
        self.collective_model = collective_model
        self.workers = list(workers or [])
        self.worker_id = worker_id
        self.rank_info = rank_info
        self.parallel_config = parallel_config
        self.dtype = dtype
        self.stats = ParallelStats()
        self._group_cache: dict[str, tuple[int, ...]] = {}

    # -- group discovery ------------------------------------------------------

    def _ranks(self, kind: str) -> tuple[int, ...]:
        cached = self._group_cache.get(kind)
        if cached is not None and len(self.workers) > 0:
            return cached
        me = self.rank_info
        ids: list[int] = []
        for worker in self.workers:
            dp = getattr(worker, "dp_rank", None)
            tp = getattr(worker, "tp_rank", None)
            pp = getattr(worker, "pp_rank", None)
            if kind == "tp" and dp == me.dp_rank and pp == me.pp_rank:
                ids.append(worker.id)
            elif kind == "ep" and pp == me.pp_rank:
                ids.append(worker.id)
            elif kind == "pp_next":
                next_pp = (me.pp_rank + 1) % self.parallel_config.pipeline_parallel_size
                if dp == me.dp_rank and tp == me.tp_rank and pp == next_pp:
                    ids.append(worker.id)
        if self.worker_id not in ids and kind != "pp_next":
            ids.append(self.worker_id)
        result = tuple(sorted(set(ids)))
        self._group_cache[kind] = result
        return result

    def tp_group(self) -> tuple[int, ...]:
        return self._ranks("tp")

    def ep_group(self) -> tuple[int, ...]:
        return self._ranks("ep")

    def pp_next_worker(self) -> int | None:
        ids = self._ranks("pp_next")
        return ids[0] if ids else None

    def _link_type(self, worker_ids: Sequence[int]) -> str:
        return self.placement.level_name(self.placement.lowest_common_level(worker_ids))

    # -- estimates ------------------------------------------------------------

    def estimate_tp_collective(self, bytes_: int, operation: str = "all_reduce", count: int = 1) -> float:
        """Latency (seconds) of ``count`` identical collectives over the TP group."""
        if self.parallel_config.tensor_parallel_size <= 1 or bytes_ <= 0 or count <= 0:
            return 0.0
        group = self.tp_group()
        if len(group) <= 1:
            return 0.0
        estimate = self._collective(operation, bytes_, group)
        latency = estimate.latency_s * count * self.parallel_config.tensor_parallel_collective_latency_scale
        self._record("tp_collective", None, bytes_ * count, latency, group, estimate)
        return latency

    def estimate_pp_stage_transfer(self, bytes_: int, count: int = 1) -> float:
        if self.parallel_config.pipeline_parallel_size <= 1 or bytes_ <= 0 or count <= 0:
            return 0.0
        target = self.pp_next_worker()
        if target is None or target == self.worker_id:
            return 0.0
        estimate = self.collective_model.point_to_point(bytes_, self.placement.layout([self.worker_id, target]))
        latency = estimate.latency_s * count * self.parallel_config.pipeline_parallel_activation_latency_scale
        self._record("pp_stage_transfer", target, bytes_ * count, latency, (self.worker_id, target), estimate)
        return latency

    def estimate_ep_all2all(self, bytes_: int, count: int = 1) -> float:
        if bytes_ <= 0 or count <= 0:
            return 0.0
        group = self.ep_group()
        if len(group) <= 1:
            return 0.0
        estimate = self._collective("all_to_all", bytes_, group)
        latency = estimate.latency_s * count
        self._record("ep_all2all", None, bytes_ * count, latency, group, estimate)
        return latency

    def estimate_point_to_point(self, bytes_: int, target_worker_id: int) -> float:
        """One-off transfer (KV cache migration) to another worker, in seconds."""
        if bytes_ <= 0 or target_worker_id == self.worker_id:
            return 0.0
        estimate = self.collective_model.point_to_point(
            bytes_, self.placement.layout([self.worker_id, target_worker_id])
        )
        return estimate.latency_s

    def record_tp_shard(self) -> None:
        self.stats.record_tp_shard()

    def describe(self) -> dict[str, Any]:
        tp = self.tp_group()
        ep = self.ep_group()
        return {
            "topology": self.placement.topology.topology_id,
            "device_index": self.placement.device_index(self.worker_id),
            "tp_group": list(tp),
            "tp_group_fan": list(self.placement.layout(tp).fan) if tp else [],
            "ep_group_size": len(ep),
        }

    # -- internals ---------------------------------------------------------------

    def _collective(self, operation: str, bytes_: int, group: Sequence[int]) -> CollectiveEstimate:
        layout = self.placement.layout(group)
        return self.collective_model.estimate(
            CollectiveQuery(operation=operation, message_bytes=float(bytes_), layout=layout, dtype=self.dtype)
        )

    def _record(
        self,
        kind: str,
        target: int | None,
        bytes_: int,
        latency: float,
        group: Sequence[int],
        estimate: CollectiveEstimate,
    ) -> None:
        self.stats.record_event(
            ParallelSyncEvent(
                kind=kind,
                source_worker_id=self.worker_id,
                target_worker_id=target,
                dp_rank=self.rank_info.dp_rank,
                tp_rank=self.rank_info.tp_rank,
                pp_rank=self.rank_info.pp_rank,
                bytes=int(bytes_),
                latency=latency,
                link_type=self._link_type(group),
                group_size=len(group),
                match_type=estimate.match_type,
            )
        )
