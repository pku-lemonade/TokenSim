from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MoEStats:
    moe_compute_latency: float = 0.0
    moe_all2all_latency: float = 0.0
    moe_straggler_latency: float = 0.0
    moe_all2all_event_count: int = 0
    routed_token_count: int = 0
    per_expert_load: dict[int, int] = field(default_factory=dict)
    per_rank_expert_load: dict[int, int] = field(default_factory=dict)
    expert_placement: dict[str, Any] | None = None
    effective_moe_config: dict[str, Any] | None = None

    def record_step(
        self,
        *,
        histogram: dict[int, int],
        per_rank_load: dict[int, int],
        compute_latency: float,
        all2all_latency: float,
        straggler_latency: float,
        all2all_event_count: int,
    ) -> None:
        self.moe_compute_latency += compute_latency
        self.moe_all2all_latency += all2all_latency
        self.moe_straggler_latency += straggler_latency
        self.moe_all2all_event_count += all2all_event_count
        self.routed_token_count += sum(histogram.values())
        for expert_id, count in histogram.items():
            self.per_expert_load[expert_id] = self.per_expert_load.get(expert_id, 0) + count
        for ep_rank, count in per_rank_load.items():
            self.per_rank_expert_load[ep_rank] = (
                self.per_rank_expert_load.get(ep_rank, 0) + count
            )

    def aggregate(self, other: "MoEStats") -> "MoEStats":
        result = MoEStats(
            moe_compute_latency=self.moe_compute_latency + other.moe_compute_latency,
            moe_all2all_latency=self.moe_all2all_latency + other.moe_all2all_latency,
            moe_straggler_latency=(
                self.moe_straggler_latency + other.moe_straggler_latency
            ),
            moe_all2all_event_count=(
                self.moe_all2all_event_count + other.moe_all2all_event_count
            ),
            routed_token_count=self.routed_token_count + other.routed_token_count,
            expert_placement=self.expert_placement or other.expert_placement,
            effective_moe_config=self.effective_moe_config or other.effective_moe_config,
        )
        result.per_expert_load = _merge_loads(
            self.per_expert_load,
            other.per_expert_load,
        )
        result.per_rank_expert_load = _merge_loads(
            self.per_rank_expert_load,
            other.per_rank_expert_load,
        )
        return result

    @property
    def max_expert_load(self) -> int:
        return max(self.per_expert_load.values(), default=0)

    @property
    def mean_expert_load(self) -> float:
        if not self.per_expert_load:
            return 0.0
        return sum(self.per_expert_load.values()) / len(self.per_expert_load)

    @property
    def expert_load_imbalance_ratio(self) -> float:
        mean = self.mean_expert_load
        if mean == 0:
            return 0.0
        return self.max_expert_load / mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective_moe_config": self.effective_moe_config or {},
            "moe_ep_rank_count": _ep_rank_count(self.expert_placement),
            "moe_expert_placement": self.expert_placement or {},
            "moe_per_expert_load": {
                str(expert_id): count
                for expert_id, count in sorted(self.per_expert_load.items())
            },
            "moe_per_rank_expert_load": {
                str(rank): count for rank, count in sorted(self.per_rank_expert_load.items())
            },
            "moe_max_expert_load": self.max_expert_load,
            "moe_mean_expert_load": self.mean_expert_load,
            "moe_expert_load_imbalance_ratio": self.expert_load_imbalance_ratio,
            "moe_compute_latency": self.moe_compute_latency,
            "moe_all2all_latency": self.moe_all2all_latency,
            "moe_straggler_latency": self.moe_straggler_latency,
            "moe_all2all_event_count": self.moe_all2all_event_count,
            "moe_routed_token_count": self.routed_token_count,
        }


def _merge_loads(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) + value
    return result


def _ep_rank_count(placement: dict[str, Any] | None) -> int:
    if not placement:
        return 1
    return int(placement.get("ep_rank_count", 1))
