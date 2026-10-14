from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from TokenSim.mooncake.config import MooncakeConfig
from TokenSim.mooncake.metrics import MooncakeStats
from TokenSim.mooncake.pool_key import PoolKey
from TokenSim.mooncake.ssd import OffloadTier
from TokenSim.errors import SimulationStateError


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
    ready_at: float = 0.0
    in_use_until: float = 0.0


@dataclass
class StoreHit:
    keys: list[PoolKey]
    hit_blocks: int
    tier: str | None


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
            self.memory_capacity_bytes = (
                config.memory_capacity_blocks * self.block_bytes
            )
        else:
            self.memory_capacity_bytes = config.global_segment_size
        self.memory_used_bytes = 0
        self.objects: dict[str, StoreObject] = {}
        self.memory_lru: OrderedDict[str, None] = OrderedDict()
        self.stats = MooncakeStats()
        self.offload = (
            OffloadTier(config, self.block_bytes) if config.enable_offload else None
        )
        self.ssd = self.offload
        self.register_storage_stats(self.stats)
        self._access_counter = 0

    def lookup(self, keys: list[PoolKey], *, now: float = 0.0, stats=None) -> StoreHit:
        stats = self.stats if stats is None else stats
        if not keys:
            return StoreHit(keys=[], hit_blocks=0, tier=None)
        hit_keys: list[PoolKey] = []
        slowest_tier: str | None = None
        for key in keys:
            obj = self.objects.get(key.to_string())
            if obj is None or obj.ready_at > now:
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
            stats.store_hit_count += 1
            if self._is_offload_tier(slowest_tier):
                stats.disk_tier_hit_count += 1
                self._backend_stats(stats).hit_count += 1
            else:
                stats.memory_tier_hit_count += 1
        else:
            stats.store_miss_count += 1
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
        initial_delay: float = 0.0,
        stats=None,
    ) -> float:
        if not keys:
            return 0.0
        stats = self.stats if stats is None else stats
        stats.get_count += 1
        latency = 0.0
        objects = []
        for key in keys:
            obj = self.objects.get(key.to_string())
            if obj is None or obj.ready_at > now:
                raise SimulationStateError(
                    "external KV read needs a ready resident object"
                )
            objects.append(obj)
            self._touch(obj)
            if self._is_offload_tier(obj.tier) and self.offload is not None:
                op = self.offload.read(obj.key)
                self._record_offload_read(op, stats)
                latency += op.latency
        for obj in objects:
            obj.in_use_until = max(obj.in_use_until, now + initial_delay + latency)
            if self._is_offload_tier(obj.tier):
                self.offload.pin(obj.key, obj.in_use_until)
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
        stats=None,
        admission_policy=None,
    ) -> StoreWriteTiming:
        stats = self.stats if stats is None else stats
        if (admission_policy or self.config.admission_policy) == "never" or not keys:
            stats.admission_rejection_count += len(keys)
            return StoreWriteTiming()
        stats.put_count += 1
        timing = StoreWriteTiming()
        blocks = blocks if blocks is not None else len(keys)
        # Install large DRAM-only batches without repeated admission scans.
        if (
            not self.config.enable_offload
            and blocks > 64
            and self.eviction_policy_allows_batch(blocks)
        ):
            selected = list(dict.fromkeys(key.to_string() for key in keys[:blocks]))
            fresh = [key for key in selected if key not in self.objects]
            if (
                self.memory_used_bytes + len(fresh) * self.block_bytes
                <= self.memory_capacity_bytes
            ):
                for key_string in fresh:
                    self.objects[key_string] = StoreObject(
                        key=key_string,
                        blocks=1,
                        bytes=self.block_bytes,
                        replicas=self.config.replica_num,
                        segment=self.config.preferred_segment or "global",
                        ready_at=now + initial_delay,
                    )
                self.memory_used_bytes += len(fresh) * self.block_bytes
                for key in selected:
                    self.memory_lru.setdefault(key, None)
                    self._touch(self.objects[key])
                stats.admission_count += len(fresh)
                return timing
        for key in keys[:blocks]:
            key_timing = self._put_one(
                key,
                now=now,
                initial_delay=initial_delay + timing.accounting_latency,
                stats=stats,
            )
            timing += key_timing
        return timing

    def eviction_policy_allows_batch(self, blocks: int) -> bool:
        """Whether a bulk write can preserve the configured LRU semantics."""
        return (
            self.config.eviction_policy == "lru"
            and blocks * self.block_bytes <= self.memory_capacity_bytes
        )

    def _put_one(
        self,
        key: PoolKey,
        *,
        now: float,
        initial_delay: float,
        stats,
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
            ready_at=now + initial_delay,
        )
        has_room, timing = self._ensure_memory_capacity(obj.bytes, now=now, stats=stats)
        if not has_room:
            if self.config.enable_offload and self.offload is not None:
                obj.tier = self.offload.tier
                op = self.offload.write(obj.key, obj.blocks, now=now)
                if obj.key in op.evicted_keys:
                    stats.admission_rejection_count += 1
                    return timing
                timing += self._write_timing(
                    obj,
                    op.latency,
                    now=now,
                    initial_delay=initial_delay + timing.accounting_latency,
                )
                self._record_offload_write(op, stats)
                self.objects[obj.key] = obj
                stats.admission_count += 1
                return timing
            stats.admission_rejection_count += 1
            return timing
        self.objects[obj.key] = obj
        self.memory_used_bytes += obj.bytes
        self.memory_lru[obj.key] = None
        stats.admission_count += 1
        obj.ready_at += timing.accounting_latency
        if self.config.enable_offload and self.offload is not None:
            op = self.offload.write(obj.key, obj.blocks, now=now)
            self._record_offload_write(op, stats)
            if obj.key not in op.evicted_keys:
                timing += self._write_timing(
                    obj,
                    op.latency,
                    now=now,
                    initial_delay=initial_delay + timing.accounting_latency,
                )
        return timing

    def _ensure_memory_capacity(self, bytes_: int, *, now: float, stats):
        timing = StoreWriteTiming()
        if bytes_ > self.memory_capacity_bytes:
            return False, timing
        while self.memory_used_bytes + bytes_ > self.memory_capacity_bytes:
            if self.config.eviction_policy != "lru" or not self.memory_lru:
                return False, timing
            candidate = next(
                (
                    key
                    for key in self.memory_lru
                    if max(self.objects[key].ready_at, self.objects[key].in_use_until)
                    <= now
                ),
                None,
            )
            if candidate is None:
                return False, timing
            timing += self._evict_memory(candidate, now=now, stats=stats)
        return True, timing

    def _evict_memory(self, key, *, now: float, stats):
        self.memory_lru.pop(key)
        obj = self.objects.get(key)
        if obj is None or obj.tier != "memory":
            return StoreWriteTiming()
        self.memory_used_bytes -= obj.bytes
        stats.eviction_count += 1
        if self.config.enable_offload and self.offload is not None:
            obj.tier = self.offload.tier
            if self.offload.contains(obj.key):
                return StoreWriteTiming()
            op = self.offload.write(obj.key, obj.blocks, now=now)
            self._record_offload_write(op, stats)
            if obj.key not in op.evicted_keys:
                return self._write_timing(obj, op.latency, now=now, initial_delay=0.0)
        else:
            self.objects.pop(key, None)
        return StoreWriteTiming()

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
        obj.persisted = True
        obj.ready_at = now + initial_delay + media_latency
        self.offload.pin(obj.key, obj.ready_at)
        return StoreWriteTiming(
            accounting_latency=media_latency,
            blocking_latency=media_latency,
        )

    def _record_offload_read(self, op, stats) -> None:
        self._backend_stats(stats).record_io("read", op.blocks, op.bytes, op.latency)
        if self.offload.legacy_ssd:
            stats.ssd_read_blocks += op.blocks
            stats.ssd_read_bytes += op.bytes

    def _record_offload_write(self, op, stats) -> None:
        self._backend_stats(stats).record_io("write", op.blocks, op.bytes, op.latency)
        if self.offload.legacy_ssd:
            stats.ssd_write_blocks += op.blocks
            stats.ssd_write_bytes += op.bytes
        for key in op.evicted_keys:
            obj = self.objects.get(key)
            if obj is not None:
                obj.persisted = False
                if self._is_offload_tier(obj.tier):
                    self.objects.pop(key)
                    stats.eviction_count += 1

    def _is_offload_tier(self, tier: str | None) -> bool:
        return self.offload is not None and tier == self.offload.tier

    def _backend_stats(self, stats):
        return stats.storage_metrics.instance(
            self.config.master_server_address,
            self.offload.tier,
            self.offload.reference,
            self.offload.profile,
        )

    def register_storage_stats(self, stats):
        if self.offload is not None:
            stats.record_offload_tier(self.offload.tier)
            stats.record_offload_profile(self._offload_profile())
            self._backend_stats(stats)

    def _offload_profile(self) -> dict[str, object]:
        return self.offload.describe()
