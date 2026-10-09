from __future__ import annotations

from typing import Any


class MoEStats:
    def __init__(
        self,
        moe_compute_latency: float = 0.0,
        moe_all2all_latency: float = 0.0,
        moe_straggler_latency: float = 0.0,
        moe_all2all_event_count: int = 0,
        routed_token_count: int = 0,
        per_expert_load: dict[int, int] | None = None,
        per_rank_expert_load: dict[int, int] | None = None,
        expert_placement: dict[str, Any] | None = None,
        effective_moe_config: dict[str, Any] | None = None,
    ) -> None:
        self.moe_compute_latency = moe_compute_latency
        self.moe_all2all_latency = moe_all2all_latency
        self.moe_straggler_latency = moe_straggler_latency
        self.moe_all2all_event_count = moe_all2all_event_count
        self.routed_token_count = routed_token_count
        self._per_expert_load = per_expert_load if per_expert_load is not None else {}
        self._per_rank_expert_load = (
            per_rank_expert_load if per_rank_expert_load is not None else {}
        )
        self.expert_placement = expert_placement
        self.effective_moe_config = effective_moe_config
        # Routing supplies immutable cached histograms. Hold their references
        # and count repeated steps, expanding the loads only when read.
        self._pending: dict[tuple[int, int], list] = {}

    @property
    def per_expert_load(self) -> dict[int, int]:
        self.flush()
        return self._per_expert_load

    @per_expert_load.setter
    def per_expert_load(self, value: dict[int, int]) -> None:
        self.flush()
        self._per_expert_load = value

    @property
    def per_rank_expert_load(self) -> dict[int, int]:
        self.flush()
        return self._per_rank_expert_load

    @per_rank_expert_load.setter
    def per_rank_expert_load(self, value: dict[int, int]) -> None:
        self.flush()
        self._per_rank_expert_load = value

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
        identity = (id(histogram), id(per_rank_load))
        entry = self._pending.get(identity)
        if entry is not None and entry[0] is histogram and entry[1] is per_rank_load:
            entry[2] += 1
            self.routed_token_count += entry[3]
            return
        if entry is not None or len(self._pending) >= 4096:
            self.flush()
        routed = sum(histogram.values())
        self.routed_token_count += routed
        # The entry holds both objects, so their ids stay unique meanwhile.
        self._pending[identity] = [histogram, per_rank_load, 1, routed]

    def flush(self) -> None:
        experts, ranks = self._per_expert_load, self._per_rank_expert_load
        for histogram, per_rank_load, count, _ in self._pending.values():
            for expert_id, load in histogram.items():
                experts[expert_id] = experts.get(expert_id, 0) + load * count
            for ep_rank, load in per_rank_load.items():
                ranks[ep_rank] = ranks.get(ep_rank, 0) + load * count
        self._pending.clear()

    def aggregate(self, other: "MoEStats") -> "MoEStats":
        self.flush()
        other.flush()
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
            effective_moe_config=self.effective_moe_config
            or other.effective_moe_config,
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
        self.flush()
        return max(self.per_expert_load.values(), default=0)

    @property
    def mean_expert_load(self) -> float:
        self.flush()
        if not self.per_expert_load:
            return 0.0
        # Sparse batches still have unloaded experts. Excluding them from
        # the denominator would report a single hot expert as balanced.
        count = int(
            (self.expert_placement or {}).get("num_experts")
            or (self.effective_moe_config or {}).get("num_experts")
            or len(self.per_expert_load)
        )
        return sum(self.per_expert_load.values()) / count

    @property
    def expert_load_imbalance_ratio(self) -> float:
        mean = self.mean_expert_load
        if mean == 0:
            return 0.0
        return self.max_expert_load / mean

    def as_dict(self) -> dict[str, Any]:
        self.flush()
        return {
            "effective_moe_config": self.effective_moe_config or {},
            "moe_ep_rank_count": _ep_rank_count(self.expert_placement),
            "moe_expert_placement": self.expert_placement or {},
            "moe_per_expert_load": {
                str(expert_id): count
                for expert_id, count in sorted(self.per_expert_load.items())
            },
            "moe_per_rank_expert_load": {
                str(rank): count
                for rank, count in sorted(self.per_rank_expert_load.items())
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
