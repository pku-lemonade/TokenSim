from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_GB = 1 << 30

# Result export only surfaces a small sample of pool keys; keeping more than
# this in memory only grows RSS with simulation length.
POOL_KEY_SAMPLE_LIMIT = 64


@dataclass
class MooncakeStats:
    get_count: int = 0
    put_count: int = 0
    store_hit_count: int = 0
    store_miss_count: int = 0
    memory_tier_hit_count: int = 0
    disk_tier_hit_count: int = 0
    ssd_read_blocks: int = 0
    ssd_read_bytes: int = 0
    ssd_write_blocks: int = 0
    ssd_write_bytes: int = 0
    transferred_bytes: int = 0
    transfer_latency: float = 0.0
    p2p_latency: float = 0.0
    store_latency: float = 0.0
    admission_count: int = 0
    admission_rejection_count: int = 0
    eviction_count: int = 0
    pending_async_jobs: int = 0
    local_gpu_hit_tokens: int = 0
    mooncake_memory_hit_tokens: int = 0
    mooncake_disk_hit_tokens: int = 0
    load_wait_time: float = 0.0
    save_wait_time: float = 0.0
    pool_keys: list[str] = field(default_factory=list)
    offload_tiers: list[str] = field(default_factory=list)
    offload_profiles: list[dict[str, Any]] = field(default_factory=list)

    def aggregate(self, other: "MooncakeStats") -> "MooncakeStats":
        result = MooncakeStats()
        for field_name in self.__dataclass_fields__:
            left = getattr(self, field_name)
            right = getattr(other, field_name)
            if isinstance(left, list):
                setattr(result, field_name, [*left, *right][:POOL_KEY_SAMPLE_LIMIT])
            else:
                setattr(result, field_name, left + right)
        return result

    def record_transfer(
        self,
        *,
        bytes_: int,
        latency: float,
        kind: str,
        blocking_latency: float | None = None,
    ) -> None:
        self.transferred_bytes += bytes_
        self.transfer_latency += latency
        if kind == "p2p":
            self.p2p_latency += latency
        elif kind.startswith("store"):
            self.store_latency += latency
        if blocking_latency is None:
            blocking_latency = latency
        if kind in {"store", "store_load", "load"}:
            self.load_wait_time += blocking_latency
        elif kind in {"store_save", "save"}:
            self.save_wait_time += blocking_latency

    def record_pool_keys(self, keys: list[str]) -> None:
        remaining = POOL_KEY_SAMPLE_LIMIT - len(self.pool_keys)
        if remaining > 0:
            self.pool_keys.extend(keys[:remaining])

    def record_offload_tier(self, tier: str) -> None:
        if tier not in self.offload_tiers:
            self.offload_tiers.append(tier)

    def record_offload_profile(self, profile: dict[str, Any]) -> None:
        self.offload_profiles.append(dict(profile))

    def as_dict(self) -> dict[str, Any]:
        bandwidth = 0.0
        if self.transfer_latency > 0:
            bandwidth = self.transferred_bytes / _GB / self.transfer_latency
        offload_tiers = sorted(set(self.offload_tiers))
        offload_profiles = []
        seen_profiles = set()
        for profile in self.offload_profiles:
            fingerprint = json.dumps(profile, sort_keys=True, default=str)
            if fingerprint in seen_profiles:
                continue
            seen_profiles.add(fingerprint)
            offload_profiles.append(profile)
        return {
            "mooncake_get_count": self.get_count,
            "mooncake_put_count": self.put_count,
            "mooncake_store_hit_count": self.store_hit_count,
            "mooncake_store_miss_count": self.store_miss_count,
            "mooncake_memory_tier_hit_count": self.memory_tier_hit_count,
            "mooncake_disk_tier_hit_count": self.disk_tier_hit_count,
            "mooncake_ssd_read_blocks": self.ssd_read_blocks,
            "mooncake_ssd_read_bytes": self.ssd_read_bytes,
            "mooncake_ssd_write_blocks": self.ssd_write_blocks,
            "mooncake_ssd_write_bytes": self.ssd_write_bytes,
            "mooncake_transferred_bytes": self.transferred_bytes,
            "mooncake_effective_transfer_bandwidth_gbps": bandwidth,
            "mooncake_transfer_latency": self.transfer_latency,
            "mooncake_p2p_latency": self.p2p_latency,
            "mooncake_store_latency": self.store_latency,
            "mooncake_admission_count": self.admission_count,
            "mooncake_admission_rejection_count": self.admission_rejection_count,
            "mooncake_eviction_count": self.eviction_count,
            "mooncake_pending_async_jobs": self.pending_async_jobs,
            "mooncake_local_gpu_hit_tokens": self.local_gpu_hit_tokens,
            "mooncake_memory_hit_tokens": self.mooncake_memory_hit_tokens,
            "mooncake_disk_hit_tokens": self.mooncake_disk_hit_tokens,
            "mooncake_load_wait_time": self.load_wait_time,
            "mooncake_save_wait_time": self.save_wait_time,
            "mooncake_pool_keys": self.pool_keys[:64],
            "mooncake_offload_tiers": offload_tiers,
            "mooncake_offload_profiles": offload_profiles,
        }
