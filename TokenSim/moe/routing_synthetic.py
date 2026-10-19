"""Routing synthetic."""

from __future__ import annotations

import random
from typing import Mapping

from TokenSim.moe.routing_models import RoutingAssignment, RoutingPhase


def _scaled_histogram(
    histogram: dict[int, int],
    share: float,
    *,
    target_count: int | None = None,
) -> dict[int, int]:
    raw = {expert_id: count * share for expert_id, count in histogram.items()}
    scaled = {expert_id: int(value) for expert_id, value in raw.items()}
    target = (
        round(sum(histogram.values()) * share) if target_count is None else target_count
    )
    remainder = target - sum(scaled.values())
    for expert_id in sorted(raw, key=lambda item: (-(raw[item] - scaled[item]), item)):
        if remainder <= 0:
            break
        scaled[expert_id] += 1
        remainder -= 1
    return {expert_id: count for expert_id, count in scaled.items() if count > 0}


class SyntheticRoutingMixin:
    def _synthetic_assignments(
        self,
        request_tokens: Mapping[int, tuple[int, ...]],
        request_owners: Mapping[int, tuple[int, int]],
        *,
        iteration_id: int,
        phase: RoutingPhase,
        decode_step: int,
        layer_id: int,
        request_decode_steps: Mapping[int, int] | None = None,
    ) -> tuple[RoutingAssignment, ...]:
        original_tokens = tuple(
            (request_id, token_id)
            for request_id, token_ids in request_tokens.items()
            for token_id in token_ids
        )
        top_k = self.moe_config.num_experts_per_tok
        assignments: list[RoutingAssignment] = []
        for request_id, token_id in original_tokens:
            expert_ids = self._experts_for_token(request_id, token_id, phase, layer_id)
            owner = request_owners[request_id]
            for top_k_index in range(top_k):
                expert_id = expert_ids[top_k_index]
                assignments.append(
                    RoutingAssignment(
                        iteration_id=iteration_id,
                        phase=phase,
                        decode_step=(request_decode_steps or {}).get(
                            request_id, decode_step
                        ),
                        layer_id=layer_id,
                        request_id=request_id,
                        token_id=token_id,
                        top_k_index=top_k_index,
                        expert_id=expert_id,
                        routing_weight=1.0 / top_k,
                        a_dp_rank=owner[0],
                        a_tp_rank=owner[1],
                    )
                )
        return tuple(assignments)

    def synthetic_histogram(
        self, token_count: int, *, offset: int = 0
    ) -> dict[int, int]:
        route_count = max(0, int(token_count)) * self.moe_config.num_experts_per_tok
        if route_count == 0:
            return {}
        distribution = self.routing_config.distribution
        offset = (
            offset % self.moe_config.num_experts if distribution == "uniform" else 0
        )
        identity = (route_count, offset)
        cached = self._synthetic_cache.get(identity)
        if cached is None:
            if distribution in {"skew", "hot", "burst"}:
                cached = self._hot_histogram(route_count)
            else:
                cached = self._uniform_histogram(route_count, offset)
            self._synthetic_cache[identity] = cached
        # Callers treat histograms as read-only; returning the cached dict is safe.
        return cached

    def _experts_for_token(
        self, request_id: int, token_id: int, phase: RoutingPhase, layer_id: int
    ) -> tuple[int, ...]:
        """Stateless synthetic routing; scheduling/batch identities are excluded."""
        count = self.moe_config.num_experts
        top_k = self.moe_config.num_experts_per_tok
        if self.routing_config.distribution == "uniform":
            if self.routing_config.uniform_selection == "random_without_replacement":
                rng = random.Random(
                    f"{self.seed}:{phase}:{layer_id}:{request_id}:{token_id}"
                )
                return tuple(rng.sample(range(count), top_k))
            # Consecutive request IDs at the same token position cover experts
            # cyclically, independently of how those requests are partitioned.
            start = (
                request_id * top_k
                + token_id * top_k
                + layer_id
                + self.seed
                + (1 if phase == "prefill" else 0)
            ) % count
            return tuple((start + index) % count for index in range(top_k))

        # Local RNG makes hot/skew routing independent of call order as well.
        rng = random.Random(f"{self.seed}:{phase}:{layer_id}:{request_id}:{token_id}")
        hot = sorted(
            {
                expert
                for expert in (self.routing_config.hot_experts or [0])
                if 0 <= expert < count
            }
        ) or [0]
        hot_set = set(hot)
        cold = [expert for expert in range(count) if expert not in hot_set]
        chosen = []
        for _ in range(top_k):
            prefer_hot = rng.random() < self.routing_config.hot_expert_fraction
            pool = (hot if prefer_hot else cold) or (cold if prefer_hot else hot)
            expert = pool.pop(rng.randrange(len(pool)))
            chosen.append(expert)
        return tuple(chosen)

    def _uniform_histogram(self, route_count: int, offset: int = 0) -> dict[int, int]:
        base, remainder = divmod(route_count, self.moe_config.num_experts)
        histogram = {expert_id: base for expert_id in self.expert_order} if base else {}
        for index in range(remainder):
            expert_id = self.expert_order[
                (offset + index) % self.moe_config.num_experts
            ]
            histogram[expert_id] = base + 1
        return histogram

    def _hot_histogram(self, route_count: int) -> dict[int, int]:
        hot_experts = self.routing_config.hot_experts or [0]
        hot_experts = [
            expert_id
            for expert_id in hot_experts
            if 0 <= expert_id < self.moe_config.num_experts
        ]
        if not hot_experts:
            hot_experts = [0]
        hot_count = int(route_count * self.routing_config.hot_expert_fraction)
        cold_count = route_count - hot_count
        histogram: dict[int, int] = {}
        for index in range(hot_count):
            expert_id = hot_experts[index % len(hot_experts)]
            histogram[expert_id] = histogram.get(expert_id, 0) + 1
        cold_experts = [
            expert_id
            for expert_id in range(self.moe_config.num_experts)
            if expert_id not in set(hot_experts)
        ] or hot_experts
        for index in range(cold_count):
            expert_id = cold_experts[index % len(cold_experts)]
            histogram[expert_id] = histogram.get(expert_id, 0) + 1
        return histogram
