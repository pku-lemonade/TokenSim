from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from TokenSim.mooncake.config import MooncakeConfig
from TokenSim.mooncake.metrics import MooncakeStats
from TokenSim.mooncake.pool_key import PoolKey
from TokenSim.mooncake.ssd import OffloadTier


@dataclass
class StoreObject:
    key: str
    blocks: int
    bytes: int
    tier: str = "memory"
    segment: str = "global"
    replicas: int = 1
    persisted: bool = False
    last_access: int = 0


@dataclass
class StoreHit:
    keys: list[PoolKey]
    hit_blocks: int
    tier: str | None

    @property
    def hit_tokens(self) -> int:
        if not self.keys:
            return 0
        block_size = getattr(self.keys[0], "_block_size", 0)
        return self.hit_blocks * block_size


@dataclass(frozen=True)
class StoreWriteTiming:
    accounting_latency: float = 0.0
    blocking_latency: float = 0.0

    def __add__(self, other: StoreWriteTiming) -> StoreWriteTiming:
        return StoreWriteTiming(
            accounting_latency=self.accounting_latency + other.accounting_latency,
            blocking_latency=self.blocking_latency + other.blocking_latency,
        )


class MooncakeStore:
    def __init__(self, config: MooncakeConfig, block_bytes: int):
        self.config = config
        self.block_bytes = max(1, int(block_bytes))
        if config.memory_capacity_blocks is not None:
            self.memory_capacity_bytes = config.memory_capacity_blocks * self.block_bytes
        else:
            self.memory_capacity_bytes = config.global_segment_size
        self.memory_used_bytes = 0
        self.objects: dict[str, StoreObject] = {}
        self.memory_lru: OrderedDict[str, None] = OrderedDict()
        self.stats = MooncakeStats()
        self.offload = OffloadTier(config, self.block_bytes) if config.enable_offload else None
        self.ssd = self.offload
        if self.offload is not None:
            self.stats.record_offload_tier(self.offload.tier)
            self.stats.record_offload_profile(self._offload_profile())
        self._access_counter = 0

    def lookup(self, keys: list[PoolKey], *, now: float = 0.0) -> StoreHit:
        hit_keys: list[PoolKey] = []
        slowest_tier: str | None = None
        for key in keys:
            obj = self.objects.get(key.to_string())
            if obj is None:
                break
            if (
                self._is_offload_tier(obj.tier)
                and self.offload is not None
                and not self.offload.contains(obj.key)
            ):
                break
            hit_keys.append(key)
            if self._is_offload_tier(obj.tier):
                slowest_tier = obj.tier
            elif slowest_tier is None:
                slowest_tier = "memory"
        if hit_keys:
            self.stats.store_hit_count += 1
            if self._is_offload_tier(slowest_tier):
                self.stats.disk_tier_hit_count += 1
            else:
                self.stats.memory_tier_hit_count += 1
        else:
            self.stats.store_miss_count += 1
        return StoreHit(keys=hit_keys, hit_blocks=len(hit_keys), tier=slowest_tier)

    def missing_indices(self, keys: list[PoolKey]) -> list[int]:
        """Indices of keys absent from the store.

        Save-side existence prefilter, mirroring the reference connector's
        ``batch_is_exist`` before ``batch_put``: checked per key (no prefix
        requirement) and without touching hit/miss statistics or LRU heat.
        """
        missing: list[int] = []
        for index, key in enumerate(keys):
            obj = self.objects.get(key.to_string())
            present = obj is not None and (
                not self._is_offload_tier(obj.tier)
                or self.offload is None
                or self.offload.contains(obj.key)
            )
            if not present:
                missing.append(index)
        return missing

    def get(
        self,
        keys: list[PoolKey],
        tier: str | None,
        *,
        now: float = 0.0,
    ) -> float:
        if not keys:
            return 0.0
        self.stats.get_count += 1
        latency = 0.0
        for key in keys:
            obj = self.objects.get(key.to_string())
            if obj is None:
                continue
            self._touch(obj)
            if self._is_offload_tier(obj.tier) and self.offload is not None:
                op = self.offload.read(obj.key)
                self._record_offload_read(op)
                latency += op.latency
        return latency

    def put(
        self,
        keys: list[PoolKey],
        blocks: int | None = None,
        *,
        now: float = 0.0,
        initial_delay: float = 0.0,
    ) -> float:
        return self.put_with_timing(
            keys,
            blocks,
            now=now,
            initial_delay=initial_delay,
        ).blocking_latency

    def put_with_timing(
        self,
        keys: list[PoolKey],
        blocks: int | None = None,
        *,
        now: float = 0.0,
        initial_delay: float = 0.0,
    ) -> StoreWriteTiming:
        if self.config.admission_policy == "never" or not keys:
            self.stats.admission_rejection_count += len(keys)
            return StoreWriteTiming()
        self.stats.put_count += 1
        timing = StoreWriteTiming()
        blocks = blocks if blocks is not None else len(keys)
        for key in keys[:blocks]:
            key_timing = self._put_one(
                key,
                now=now,
                initial_delay=initial_delay + timing.accounting_latency,
            )
            timing += key_timing
        return timing

    def _put_one(
        self,
        key: PoolKey,
        *,
        now: float,
        initial_delay: float,
    ) -> StoreWriteTiming:
        key_string = key.to_string()
        existing = self.objects.get(key_string)
        if existing is not None:
            self._touch(existing)
            return StoreWriteTiming()
        obj = StoreObject(
            key=key_string,
            blocks=1,
            bytes=self.block_bytes,
            replicas=self.config.replica_num,
            segment=self.config.preferred_segment or "global",
        )
        if not self._ensure_memory_capacity(obj.bytes, now=now):
            if self.config.enable_offload and self.offload is not None:
                obj.tier = self.offload.tier
                op = self.offload.write(obj.key, obj.blocks)
                if obj.key in op.evicted_keys:
                    self.stats.admission_rejection_count += 1
                    return StoreWriteTiming()
                timing = self._write_timing(
                    obj,
                    op.latency,
                    now=now,
                    initial_delay=initial_delay,
                )
                self._record_offload_write(op)
                self.objects[obj.key] = obj
                self.stats.admission_count += 1
                return timing
            self.stats.admission_rejection_count += 1
            return StoreWriteTiming()
        self.objects[obj.key] = obj
        self.memory_used_bytes += obj.bytes
        self.memory_lru[obj.key] = None
        self.stats.admission_count += 1
        timing = StoreWriteTiming()
        if self.config.enable_offload and self.offload is not None:
            op = self.offload.write(obj.key, obj.blocks)
            if obj.key not in op.evicted_keys:
                timing = self._write_timing(
                    obj,
                    op.latency,
                    now=now,
                    initial_delay=initial_delay,
                )
                self._record_offload_write(op)
        return timing

    def _ensure_memory_capacity(self, bytes_: int, *, now: float) -> bool:
        if bytes_ > self.memory_capacity_bytes:
            return False
        while self.memory_used_bytes + bytes_ > self.memory_capacity_bytes:
            if self.config.eviction_policy != "lru" or not self.memory_lru:
                return False
            self._evict_memory_lru(now=now)
        return True

    def _evict_memory_lru(self, *, now: float) -> None:
        key, _ = self.memory_lru.popitem(last=False)
        obj = self.objects.get(key)
        if obj is None or obj.tier != "memory":
            return
        self.memory_used_bytes -= obj.bytes
        self.stats.eviction_count += 1
        if self.config.enable_offload and self.offload is not None:
            obj.tier = self.offload.tier
            op = self.offload.write(obj.key, obj.blocks)
            self._write_timing(obj, op.latency, now=now, initial_delay=0.0)
            self._record_offload_write(op)
            for evicted_key in op.evicted_keys:
                if evicted_key != obj.key:
                    self.objects.pop(evicted_key, None)
        else:
            self.objects.pop(key, None)

    def _touch(self, obj: StoreObject) -> None:
        self._access_counter += 1
        obj.last_access = self._access_counter
        if obj.tier == "memory" and obj.key in self.memory_lru:
            self.memory_lru.move_to_end(obj.key)

    def _write_timing(
        self,
        obj: StoreObject,
        media_latency: float,
        *,
        now: float,
        initial_delay: float,
    ) -> StoreWriteTiming:
        del now, initial_delay
        obj.persisted = True
        return StoreWriteTiming(
            accounting_latency=media_latency,
            blocking_latency=media_latency,
        )

    def _record_offload_read(self, op) -> None:
        self.stats.ssd_read_blocks += op.blocks
        self.stats.ssd_read_bytes += op.bytes

    def _record_offload_write(self, op) -> None:
        self.stats.ssd_write_blocks += op.blocks
        self.stats.ssd_write_bytes += op.bytes
        self.stats.eviction_count += len(
            [key for key in op.evicted_keys if key in self.objects]
        )

    def _record_ssd_write(self, op) -> None:
        self._record_offload_write(op)

    @staticmethod
    def _is_offload_tier(tier: str | None) -> bool:
        return tier == "ssd"

    def _offload_profile(self) -> dict[str, object]:
        return {
            "tier": self.offload.tier if self.offload is not None else None,
            "memory_capacity_blocks": self.config.memory_capacity_blocks,
            "ssd_capacity_blocks": self.config.ssd_capacity_blocks,
            "ssd_capacity_gb": self.config.ssd_capacity_gb,
            "ssd_read_bw_gbps": self.config.ssd_read_bw_gbps,
            "ssd_write_bw_gbps": self.config.ssd_write_bw_gbps,
            "ssd_read_latency_us": self.config.ssd_read_latency_us,
            "ssd_write_latency_us": self.config.ssd_write_latency_us,
            "write_blocking": True,
            "write_persistence": "blocking",
        }
