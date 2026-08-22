from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConnectorTransferPlan:
    request_ids: list[int] = field(default_factory=list)
    request_block_counts: dict[int, int] = field(default_factory=dict)
    source_worker_id: int | None = None
    target_worker_id: int | None = None
    blocks: int = 0
    bytes: int = 0
    latency: float = 0.0
    kind: str = "noop"
    keys: list[str] = field(default_factory=list)
    tier: str | None = None
    async_transfer: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class KVConnectorMetadata:
    connector_name: str = "NoopConnector"
    loads: list[ConnectorTransferPlan] = field(default_factory=list)
    saves: list[ConnectorTransferPlan] = field(default_factory=list)
    preempted_request_ids: list[int] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.loads or self.saves or self.preempted_request_ids)


@dataclass
class KVConnectorWorkerMetadata:
    finished_sending: set[int] = field(default_factory=set)
    finished_recving: set[int] = field(default_factory=set)
    events: list[str] = field(default_factory=list)

    def aggregate(
        self,
        other: "KVConnectorWorkerMetadata",
    ) -> "KVConnectorWorkerMetadata":
        return KVConnectorWorkerMetadata(
            finished_sending=self.finished_sending | other.finished_sending,
            finished_recving=self.finished_recving | other.finished_recving,
            events=[*self.events, *other.events],
        )


@dataclass
class ConnectorStats:
    transfer_count: int = 0
    transfer_blocks: int = 0
    transfer_bytes: int = 0
    transfer_latency: float = 0.0
    load_wait_time: float = 0.0
    save_wait_time: float = 0.0
    producer_count: int = 0
    consumer_count: int = 0
    both_count: int = 0
    placement_decision_count: int = 0

    @property
    def blocks(self) -> int:
        return self.transfer_blocks

    @property
    def bytes(self) -> int:
        return self.transfer_bytes

    @property
    def latency(self) -> float:
        return self.transfer_latency

    def record_transfer(
        self,
        *,
        blocks: int = 0,
        bytes_: int = 0,
        latency: float = 0.0,
        kind: str = "transfer",
        blocking_latency: float | None = None,
    ) -> None:
        self.transfer_count += 1
        self.transfer_blocks += blocks
        self.transfer_bytes += bytes_
        self.transfer_latency += latency
        if blocking_latency is None:
            blocking_latency = latency
        if kind == "load":
            self.load_wait_time += blocking_latency
        elif kind == "save":
            self.save_wait_time += blocking_latency

    def record_role(self, role: str | None) -> None:
        if role == "kv_producer":
            self.producer_count += 1
        elif role == "kv_consumer":
            self.consumer_count += 1
        elif role == "kv_both":
            self.both_count += 1

    def aggregate(self, other: "ConnectorStats") -> "ConnectorStats":
        return ConnectorStats(
            transfer_count=self.transfer_count + other.transfer_count,
            transfer_blocks=self.transfer_blocks + other.transfer_blocks,
            transfer_bytes=self.transfer_bytes + other.transfer_bytes,
            transfer_latency=self.transfer_latency + other.transfer_latency,
            load_wait_time=self.load_wait_time + other.load_wait_time,
            save_wait_time=self.save_wait_time + other.save_wait_time,
            producer_count=self.producer_count + other.producer_count,
            consumer_count=self.consumer_count + other.consumer_count,
            both_count=self.both_count + other.both_count,
            placement_decision_count=(
                self.placement_decision_count + other.placement_decision_count
            ),
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "connector_transfer_count": self.transfer_count,
            "connector_transfer_blocks": self.transfer_blocks,
            "connector_transfer_bytes": self.transfer_bytes,
            "connector_transfer_latency": self.transfer_latency,
            "connector_load_wait_time": self.load_wait_time,
            "connector_save_wait_time": self.save_wait_time,
            "connector_producer_count": self.producer_count,
            "connector_consumer_count": self.consumer_count,
            "connector_both_count": self.both_count,
            "connector_placement_decision_count": self.placement_decision_count,
        }
