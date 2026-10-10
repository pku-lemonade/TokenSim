from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import MooncakeConfig


@dataclass
class OffloadOperation:
    tier: str
    blocks: int
    bytes: int
    latency: float
    evicted_keys: list[str] = field(default_factory=list)


SSDOperation = OffloadOperation


class OffloadTier:
    """Offload tier simulated at block-count granularity.

    Only per-key block counts are tracked; per-block offsets/addressing are
    intentionally not simulated because no exported metric consumes them.
    """

    def __init__(self, config: MooncakeConfig, block_bytes: int):
        self.config = config
        self.tier = config.offload_tier
        self.block_bytes = max(1, int(block_bytes))
        if config.ssd_capacity_blocks is not None:
            self.capacity_blocks = int(config.ssd_capacity_blocks)
        else:
            self.capacity_blocks = config.ssd_capacity_bytes // self.block_bytes
        self._blocks: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._used_blocks = 0
        self._protected_until = {}

    def contains(self, key: str) -> bool:
        return key in self._blocks

    def read(self, key: str) -> SSDOperation:
        blocks = self._blocks.get(key, 0)
        bytes_ = blocks * self.block_bytes
        if key in self._lru:
            self._lru.move_to_end(key)
        return OffloadOperation(
            tier=self.tier,
            blocks=blocks,
            bytes=bytes_,
            latency=self._read_latency(bytes_),
        )

    def pin(self, key: str, until: float) -> None:
        self._protected_until[key] = max(self._protected_until.get(key, 0), until)

    def write(self, key: str, blocks: int, *, now: float = 0) -> OffloadOperation:
        if self.capacity_blocks <= 0:
            raise ConfigurationError(
                f"{self.tier.upper()} tier is not enabled or has zero capacity"
            )
        blocks = int(blocks)
        if blocks > self.capacity_blocks:
            return OffloadOperation(
                tier=self.tier,
                blocks=0,
                bytes=0,
                latency=0.0,
                evicted_keys=[key],
            )
        evicted: list[str] = []
        extra = blocks - self._blocks.get(key, 0)
        candidates, available = [], self.capacity_blocks - self.used_blocks
        if extra > available:
            for candidate in self._lru:
                if candidate != key and self._protected_until.get(candidate, 0) <= now:
                    candidates.append(candidate)
                    available += self._blocks[candidate]
                    if extra <= available:
                        break
        if extra > available:
            return OffloadOperation(
                tier=self.tier,
                blocks=0,
                bytes=0,
                latency=0.0,
                evicted_keys=[key],
            )
        for candidate in candidates:
            self._remove_existing(candidate)
            evicted.append(candidate)
        self._remove_existing(key)
        self._blocks[key] = blocks
        self._used_blocks += blocks
        self._lru[key] = None
        bytes_ = blocks * self.block_bytes
        return OffloadOperation(
            tier=self.tier,
            blocks=blocks,
            bytes=bytes_,
            latency=self._write_latency(bytes_),
            evicted_keys=evicted,
        )

    @property
    def used_blocks(self) -> int:
        return self._used_blocks

    def _remove_existing(self, key: str) -> None:
        self._used_blocks -= self._blocks.pop(key, 0)
        self._lru.pop(key, None)
        self._protected_until.pop(key, None)

    def _read_latency(self, bytes_: int) -> float:
        if bytes_ <= 0:
            return 0.0
        return (
            self.config.ssd_read_latency_us / 1e6
            + bytes_ / self.config.ssd_read_bandwidth_bytes_per_s
        )

    def _write_latency(self, bytes_: int) -> float:
        if bytes_ <= 0:
            return 0.0
        return (
            self.config.ssd_write_latency_us / 1e6
            + bytes_ / self.config.ssd_write_bandwidth_bytes_per_s
        )


SSDTier = OffloadTier
