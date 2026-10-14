from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.storage import IOCost, IORequest, StorageBackendFactory
from TokenSim.storage.ssd import legacy_ssd_backend, legacy_ssd_profile
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
        self.block_bytes = max(1, int(block_bytes))
        self.legacy_ssd = config.offload is None
        self.tier = "ssd" if self.legacy_ssd else config.offload.name
        self.reference = "ssd" if self.legacy_ssd else config.offload.backend
        self.backend = (
            legacy_ssd_backend(config, self.block_bytes)
            if self.legacy_ssd
            else StorageBackendFactory.create(config.offload)
        )
        self.capacity_blocks = self.backend.capacity_bytes // self.block_bytes
        self.profile = deepcopy(self.backend.describe())
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
        bytes_ = blocks * self.block_bytes
        latency = self._write_latency(bytes_)
        for candidate in candidates:
            self._remove_existing(candidate)
            evicted.append(candidate)
        self._remove_existing(key)
        self._blocks[key] = blocks
        self._used_blocks += blocks
        self._lru[key] = None
        return OffloadOperation(
            tier=self.tier,
            blocks=blocks,
            bytes=bytes_,
            latency=latency,
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
        return self._latency("read", bytes_)

    def _write_latency(self, bytes_: int) -> float:
        return self._latency("write", bytes_)

    def _latency(self, operation, nbytes):
        if nbytes <= 0:
            return 0.0
        cost = self.backend.estimate_io(IORequest(operation, nbytes))
        if not isinstance(cost, IOCost):
            raise SimulationStateError("storage backend estimate_io must return IOCost")
        return float(cost.service_seconds)

    def describe(self):
        if self.legacy_ssd:
            return legacy_ssd_profile(self.config)
        return {
            "tier": self.tier,
            "backend": self.reference,
            "memory_capacity_blocks": self.config.memory_capacity_blocks,
            "capacity_blocks": self.capacity_blocks,
            "profile": deepcopy(self.profile),
            "write_blocking": True,
            "write_persistence": "blocking",
        }


SSDTier = OffloadTier
