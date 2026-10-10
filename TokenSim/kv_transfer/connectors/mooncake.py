from __future__ import annotations

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import Segment
from TokenSim.mooncake.config import parse_mooncake_config
from TokenSim.mooncake.transfer_engine import TransferEngineSimulator
from TokenSim.mooncake.metrics import MooncakeStats

from .base import BaseKVConnector
from .metadata import ConnectorTransferPlan


class MooncakeConnector(BaseKVConnector):
    name = "MooncakeConnector"

    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__(config, cache_config)
        if config.kv_connector_extra_config.get("transfer_overlap", False):
            from TokenSim.errors import ConfigurationError

            raise ConfigurationError(
                "point-to-point metadata timing does not support transfer_overlap"
            )
        self.options = parse_mooncake_config(config.kv_connector_extra_config)
        self.transfer_engine = TransferEngineSimulator(self.options)
        self.mooncake_stats = MooncakeStats()

    def build_transfer_plan(
        self,
        *,
        requests: list[Request],
        source_worker_id: int,
        target_worker_id: int,
        kind: str,
        latency: float | None = None,
        request_block_counts: dict[int, int] | None = None,
    ) -> ConnectorTransferPlan:
        if request_block_counts is None:
            request_block_counts = {
                req.id: req.num_physical_token_blocks for req in requests
            }
        blocks = sum(request_block_counts.values())
        bytes_ = self._bytes_for_blocks(blocks)
        topology = "local" if source_worker_id == target_worker_id else "cross_node"
        transfer = self.transfer_engine.create_transfer(
            source_segment=Segment(
                name=f"worker-{source_worker_id}",
                tier="vram",
                worker_id=source_worker_id,
            ),
            target_segment=Segment(
                name=f"worker-{target_worker_id}",
                tier="vram",
                worker_id=target_worker_id,
            ),
            blocks=blocks,
            bytes_=bytes_,
            kind="p2p",
            topology=topology,
        )
        if latency is None:
            latency = transfer.latency
        keys = self._keys_for_requests(requests, request_block_counts)
        return ConnectorTransferPlan(
            request_ids=[req.id for req in requests],
            request_block_counts=dict(request_block_counts),
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            blocks=blocks,
            bytes=bytes_,
            latency=latency,
            kind=kind,
            keys=keys,
            tier="p2p",
            async_transfer=self.options.load_async,
            extra={
                "protocol": transfer.protocol,
                "topology": transfer.topology,
                "connector": self.name,
            },
        )

    def start_load_kv(self) -> float:
        latency = 0.0
        for plan in self._own_plans(self._metadata.loads):
            latency += plan.latency
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=plan.latency,
                kind="load",
            )
            self.mooncake_stats.record_transfer(
                bytes_=plan.bytes,
                latency=plan.latency,
                kind="p2p",
                blocking_latency=(
                    0.0
                    if plan.async_transfer and self.options.transfer_overlap
                    else plan.latency
                ),
            )
        return latency

    def wait_for_save(self) -> float:
        latency = 0.0
        for plan in self._own_plans(self._metadata.saves):
            latency += plan.latency
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=plan.latency,
                kind="save",
            )
            self.mooncake_stats.record_transfer(
                bytes_=plan.bytes,
                latency=plan.latency,
                kind="p2p",
                blocking_latency=(
                    0.0
                    if plan.async_transfer and self.options.transfer_overlap
                    else plan.latency
                ),
            )
        return latency

    def _bytes_for_blocks(self, blocks: int) -> int:
        if self.cache_config is None:
            return 0
        return int(
            blocks * self.cache_config.block_size * self.cache_config.size_per_token
        )

    def _keys_for_requests(
        self,
        requests: list[Request],
        request_block_counts: dict[int, int],
    ) -> list[str]:
        keys: list[str] = []
        for req in requests:
            block_count = request_block_counts.get(
                req.id,
                req.num_physical_token_blocks,
            )
            physical_blocks = list(getattr(req, "_physical_token_blocks", []))
            for block_index in range(block_count):
                block = (
                    physical_blocks[block_index]
                    if block_index < len(physical_blocks)
                    else None
                )
                block_hash = getattr(block, "block_hash", None)
                if block_hash is not None:
                    keys.append(str(block_hash))
                    continue
                block_number = getattr(block, "block_number", block_index)
                keys.append(f"req:{req.id}:block:{block_number}")
        return keys

    def _own_plans(
        self, plans: list[ConnectorTransferPlan]
    ) -> list[ConnectorTransferPlan]:
        return [
            plan
            for plan in plans
            if plan.extra.get("connector") == self.name or plan.tier == "p2p"
        ]
