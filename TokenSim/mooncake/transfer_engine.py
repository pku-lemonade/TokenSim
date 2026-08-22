from __future__ import annotations

from dataclasses import dataclass

from TokenSim.mooncake.config import MooncakeConfig

_GB = 1 << 30


@dataclass(frozen=True)
class Segment:
    name: str
    tier: str
    worker_id: int | None = None
    node: str | None = None
    capacity_bytes: int = 0


@dataclass
class BatchTransfer:
    source_segment: Segment
    target_segment: Segment
    blocks: int
    bytes: int
    protocol: str
    fixed_latency: float
    bandwidth_gbps: float
    parallel_paths: int = 1
    topology: str = "local"
    kind: str = "transfer"

    @property
    def latency(self) -> float:
        if self.bytes <= 0:
            return 0.0
        bandwidth = max(1e-9, self.bandwidth_gbps * max(1, self.parallel_paths))
        return self.fixed_latency + self.bytes / _GB / bandwidth


@dataclass
class TransferJob:
    transfer: BatchTransfer
    issued_time: float = 0.0
    async_: bool = False
    overlap: bool = False

    @property
    def completion_time(self) -> float:
        return self.issued_time + self.transfer.latency

    @property
    def blocking_latency(self) -> float:
        if self.async_ and self.overlap:
            return 0.0
        return self.transfer.latency


class TransferEngineSimulator:
    def __init__(self, config: MooncakeConfig):
        self.config = config
        self.jobs: list[TransferJob] = []

    def create_transfer(
        self,
        *,
        source_segment: Segment,
        target_segment: Segment,
        blocks: int,
        bytes_: int,
        protocol: str | None = None,
        kind: str = "transfer",
        topology: str = "local",
    ) -> BatchTransfer:
        protocol = protocol or self.config.protocol
        return BatchTransfer(
            source_segment=source_segment,
            target_segment=target_segment,
            blocks=blocks,
            bytes=bytes_,
            protocol=protocol,
            fixed_latency=self._fixed_latency(protocol, topology),
            bandwidth_gbps=self._bandwidth(protocol),
            parallel_paths=max(self.config.parallel_paths, self.config.num_nics),
            topology=topology,
            kind=kind,
        )

    def submit(
        self,
        transfer: BatchTransfer,
        *,
        now: float = 0.0,
        async_: bool | None = None,
        overlap: bool | None = None,
    ) -> TransferJob:
        job = TransferJob(
            transfer=transfer,
            issued_time=now,
            async_=self.config.load_async if async_ is None else async_,
            overlap=self.config.transfer_overlap if overlap is None else overlap,
        )
        self.jobs.append(job)
        return job

    def pending_jobs(self, now: float) -> int:
        return sum(1 for job in self.jobs if job.completion_time > now)

    def _fixed_latency(self, protocol: str, topology: str) -> float:
        if self.config.fixed_latency_us is not None:
            latency_us = self.config.fixed_latency_us
        else:
            latency_us = self.config.protocol_latency_us.get(
                protocol,
                {
                    "tcp": 40.0,
                    "rdma": 8.0,
                    "nvlink": 2.0,
                    "nvmeof": 80.0,
                }.get(protocol, 40.0),
            )
        if topology == "cross_node":
            latency_us *= 1.5
        return latency_us / 1e6

    def _bandwidth(self, protocol: str) -> float:
        if self.config.bandwidth_gbps is not None:
            return self.config.bandwidth_gbps
        return self.config.protocol_bandwidth_gbps.get(
            protocol,
            {
                "tcp": 12.5,
                "rdma": 50.0,
                "nvlink": 300.0,
                "nvmeof": 7.0,
            }.get(protocol, 12.5),
        )
