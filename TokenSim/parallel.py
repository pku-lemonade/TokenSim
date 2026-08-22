from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from TokenSim.config.config import ParallelConfig, ParallelRankInfo, _GB


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
        }


@dataclass
class ParallelStats:
    latency_total: float = 0.0
    tp_collective_latency: float = 0.0
    pp_transfer_latency: float = 0.0
    ep_all2all_latency: float = 0.0
    sync_event_count: int = 0
    roofline_conversion_count: int = 0
    # Aggregated in place of a per-event list: one entry per step used to grow
    # RSS unboundedly while the export only needs totals.
    link_type_counts: dict[str, int] = field(default_factory=dict)

    def record_event(self, event: ParallelSyncEvent) -> None:
        self.sync_event_count += 1
        self.latency_total += event.latency
        self.link_type_counts[event.link_type] = (
            self.link_type_counts.get(event.link_type, 0) + 1
        )
        if event.kind == "tp_collective":
            self.tp_collective_latency += event.latency
        elif event.kind == "pp_stage_transfer":
            self.pp_transfer_latency += event.latency
        elif event.kind == "ep_all2all":
            self.ep_all2all_latency += event.latency

    def record_roofline_conversion(self) -> None:
        self.roofline_conversion_count += 1

    def aggregate(self, other: "ParallelStats") -> "ParallelStats":
        link_type_counts = dict(self.link_type_counts)
        for link_type, count in other.link_type_counts.items():
            link_type_counts[link_type] = link_type_counts.get(link_type, 0) + count
        return ParallelStats(
            latency_total=self.latency_total + other.latency_total,
            tp_collective_latency=(
                self.tp_collective_latency + other.tp_collective_latency
            ),
            pp_transfer_latency=self.pp_transfer_latency + other.pp_transfer_latency,
            ep_all2all_latency=self.ep_all2all_latency + other.ep_all2all_latency,
            sync_event_count=self.sync_event_count + other.sync_event_count,
            roofline_conversion_count=(
                self.roofline_conversion_count + other.roofline_conversion_count
            ),
            link_type_counts=link_type_counts,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "parallel_latency_total": self.latency_total,
            "parallel_tp_collective_latency": self.tp_collective_latency,
            "parallel_pp_transfer_latency": self.pp_transfer_latency,
            "parallel_ep_all2all_latency": self.ep_all2all_latency,
            "parallel_sync_event_count": self.sync_event_count,
            "parallel_roofline_conversion_count": self.roofline_conversion_count,
        }


class ParallelCommunicator:
    def __init__(
        self,
        *,
        roofline: Any,
        workers: list[Any] | None,
        worker_id: int,
        rank_info: ParallelRankInfo,
        parallel_config: ParallelConfig,
        hardware: str,
    ) -> None:
        self.roofline = roofline
        self.workers = workers or []
        self.worker_id = worker_id
        self.rank_info = rank_info
        self.parallel_config = parallel_config
        self.hardware = hardware
        self.stats = ParallelStats()

    def estimate_tp_collective(self, bytes_: int) -> float:
        if self.parallel_config.tensor_parallel_size <= 1 or bytes_ <= 0:
            return 0.0
        target_id, link_type, latency = self._collective_link(bytes_)
        latency *= self.parallel_config.tensor_parallel_collective_latency_scale
        self.stats.record_event(
            ParallelSyncEvent(
                kind="tp_collective",
                source_worker_id=self.worker_id,
                target_worker_id=target_id,
                dp_rank=self.rank_info.dp_rank,
                tp_rank=self.rank_info.tp_rank,
                pp_rank=self.rank_info.pp_rank,
                bytes=bytes_,
                latency=latency,
                link_type=link_type,
            )
        )
        return latency

    def estimate_pp_stage_transfer(self, bytes_: int) -> float:
        if self.parallel_config.pipeline_parallel_size <= 1 or bytes_ <= 0:
            return 0.0
        target_id, link_type, latency = self._adjacent_pp_link(bytes_)
        latency *= self.parallel_config.pipeline_parallel_activation_latency_scale
        self.stats.record_event(
            ParallelSyncEvent(
                kind="pp_stage_transfer",
                source_worker_id=self.worker_id,
                target_worker_id=target_id,
                dp_rank=self.rank_info.dp_rank,
                tp_rank=self.rank_info.tp_rank,
                pp_rank=self.rank_info.pp_rank,
                bytes=bytes_,
                latency=latency,
                link_type=link_type,
            )
        )
        return latency

    def estimate_ep_all2all(self, bytes_: int) -> float:
        if bytes_ <= 0:
            return 0.0
        target_id, link_type, latency = self._ep_link(bytes_)
        self.stats.record_event(
            ParallelSyncEvent(
                kind="ep_all2all",
                source_worker_id=self.worker_id,
                target_worker_id=target_id,
                dp_rank=self.rank_info.dp_rank,
                tp_rank=self.rank_info.tp_rank,
                pp_rank=self.rank_info.pp_rank,
                bytes=bytes_,
                latency=latency,
                link_type=link_type,
            )
        )
        return latency

    def record_roofline_conversion(self) -> None:
        self.stats.record_roofline_conversion()

    def _collective_link(self, bytes_: int) -> tuple[int | None, str, float]:
        candidates = [
            worker
            for worker in self.workers
            if getattr(worker, "dp_rank", None) == self.rank_info.dp_rank
            and getattr(worker, "pp_rank", None) == self.rank_info.pp_rank
            and getattr(worker, "tp_rank", None) != self.rank_info.tp_rank
        ]
        if not candidates:
            return None, "local", 0.0
        latencies = [self._point_to_point_latency(worker, bytes_) for worker in candidates]
        target, link_type, latency = max(latencies, key=lambda item: item[2])
        return target, link_type, latency

    def _adjacent_pp_link(self, bytes_: int) -> tuple[int | None, str, float]:
        next_pp_rank = (self.rank_info.pp_rank + 1) % self.parallel_config.pipeline_parallel_size
        candidates = [
            worker
            for worker in self.workers
            if getattr(worker, "dp_rank", None) == self.rank_info.dp_rank
            and getattr(worker, "tp_rank", None) == self.rank_info.tp_rank
            and getattr(worker, "pp_rank", None) == next_pp_rank
        ]
        if not candidates:
            return None, "local", 0.0
        target, link_type, latency = self._point_to_point_latency(candidates[0], bytes_)
        return target, link_type, latency

    def _ep_link(self, bytes_: int) -> tuple[int | None, str, float]:
        candidates = [
            worker
            for worker in self.workers
            if getattr(worker, "pp_rank", None) == self.rank_info.pp_rank
            and (
                getattr(worker, "tp_rank", None) != self.rank_info.tp_rank
                or getattr(worker, "dp_rank", None) != self.rank_info.dp_rank
            )
        ]
        if not candidates:
            return None, "local", 0.0
        latencies = [self._point_to_point_latency(worker, bytes_) for worker in candidates]
        target, link_type, latency = max(latencies, key=lambda item: item[2])
        return target, link_type, latency

    def _point_to_point_latency(self, target_worker: Any, bytes_: int) -> tuple[int, str, float]:
        source_network = getattr(self._self_worker(), "network", None)
        target_network = getattr(target_worker, "network", None)
        source_hardware = getattr(self._self_worker(), "hardware", self.hardware)
        target_hardware = getattr(target_worker, "hardware", self.hardware)
        links = getattr(self.roofline, "links", {})
        hardwares = getattr(self.roofline, "hardwares", {})

        if source_network is not None and source_network == target_network:
            source_nvlink = getattr(hardwares[source_hardware], "Nvlink", None)
            target_nvlink = getattr(hardwares[target_hardware], "Nvlink", None)
            if source_nvlink in links and target_nvlink in links:
                link_type = "nvlink"
                latency = max(links[source_nvlink].Latency, links[target_nvlink].Latency)
                bandwidth = min(links[source_nvlink].UniBW, links[target_nvlink].UniBW)
                return target_worker.id, link_type, latency + bytes_ / _GB / bandwidth

        source_nettype = getattr(self._self_worker(), "nettype", None)
        target_nettype = getattr(target_worker, "nettype", None)
        if source_nettype in links and target_nettype in links:
            link_type = "network"
            latency = max(links[source_nettype].Latency, links[target_nettype].Latency)
            bandwidth = min(links[source_nettype].UniBW, links[target_nettype].UniBW)
            return target_worker.id, link_type, latency + bytes_ / _GB / bandwidth
        return target_worker.id, "local", 0.0

    def _self_worker(self) -> Any:
        if 0 <= self.worker_id < len(self.workers):
            return self.workers[self.worker_id]
        return self
