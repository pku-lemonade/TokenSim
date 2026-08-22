from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from typing import Any

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe.config import MoEModelConfig, RoutingConfig


def normalize_expert_histogram(
    value: Any,
    *,
    num_experts: int,
    request_id: int | None = None,
) -> dict[int, int]:
    if value is None:
        return {}
    items: list[tuple[Any, Any]]
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        if all(isinstance(item, int) for item in value):
            counter = Counter(int(item) for item in value)
            items = list(counter.items())
        elif all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in value
        ):
            items = [(item[0], item[1]) for item in value]
        else:
            raise _invalid_histogram(request_id, "must be a dict or [expert, count] list")
    else:
        raise _invalid_histogram(request_id, "must be a dict or list")

    histogram: dict[int, int] = {}
    for expert_id_raw, count_raw in items:
        try:
            expert_id = int(expert_id_raw)
            count = int(count_raw)
        except (TypeError, ValueError) as exc:
            raise _invalid_histogram(request_id, "expert ids and counts must be integers") from exc
        if expert_id < 0 or expert_id >= num_experts:
            raise _invalid_histogram(
                request_id,
                f"expert id {expert_id} outside [0, {num_experts})",
            )
        if count < 0:
            raise _invalid_histogram(request_id, "expert counts must be non-negative")
        if count:
            histogram[expert_id] = histogram.get(expert_id, 0) + count
    return histogram


class ExpertRouting:
    # Synthetic histograms are deterministic in route_count; cap the memo so
    # pathological prefill length diversity cannot grow it unboundedly.
    _SYNTHETIC_CACHE_LIMIT = 4096

    def __init__(
        self,
        moe_config: MoEModelConfig,
        *,
        seed: int = 0,
        routing_config: RoutingConfig | None = None,
    ) -> None:
        self.moe_config = moe_config
        self.routing_config = replace(routing_config or moe_config.routing or RoutingConfig())
        self.random = random.Random(seed)
        self._synthetic_cache: dict[int, dict[int, int]] = {}
        self._normalized_cache: dict[int, dict[int, int]] = {}

    def histogram_for_step(self, requests: list[Request]) -> dict[int, int]:
        histogram: dict[int, int] = {}
        if not self.moe_config.enabled or not requests:
            return histogram
        for request in requests:
            request_histogram = self._request_histogram(request)
            for expert_id, count in request_histogram.items():
                histogram[expert_id] = histogram.get(expert_id, 0) + count
        return histogram

    def _request_histogram(self, request: Request) -> dict[int, int]:
        provided = None
        if request.is_prefill:
            provided = getattr(request, "prefill_expert_histogram", None)
        else:
            provided = getattr(request, "decode_expert_histogram", None)
        if provided is None:
            provided = getattr(request, "expert_histogram", None)
        if provided is not None:
            # Decode re-reads the same trace-provided histogram every step;
            # normalize once per request. Callers must not mutate the result.
            cached = self._normalized_cache.get(request.id)
            if cached is None:
                cached = normalize_expert_histogram(
                    provided,
                    num_experts=self.moe_config.num_experts,
                    request_id=request.id,
                )
                if len(self._normalized_cache) >= self._SYNTHETIC_CACHE_LIMIT:
                    self._normalized_cache.clear()
                self._normalized_cache[request.id] = cached
            return cached
        if getattr(request, "needs_recompute", False):
            token_count = request.recompute_tokens
        else:
            token_count = request.prefill_compute_len if request.is_prefill else 1
        return self.synthetic_histogram(token_count)

    def synthetic_histogram(self, token_count: int) -> dict[int, int]:
        route_count = max(0, int(token_count)) * self.moe_config.num_experts_per_tok
        if route_count == 0:
            return {}
        cached = self._synthetic_cache.get(route_count)
        if cached is None:
            distribution = self.routing_config.distribution
            if distribution in {"skew", "hot", "burst"}:
                cached = self._hot_histogram(route_count)
            else:
                cached = self._uniform_histogram(route_count)
            if len(self._synthetic_cache) >= self._SYNTHETIC_CACHE_LIMIT:
                self._synthetic_cache.clear()
            self._synthetic_cache[route_count] = cached
        # Callers treat histograms as read-only; returning the cached dict is safe.
        return cached

    def _uniform_histogram(self, route_count: int) -> dict[int, int]:
        histogram = {expert_id: 0 for expert_id in range(self.moe_config.num_experts)}
        for index in range(route_count):
            expert_id = index % self.moe_config.num_experts
            histogram[expert_id] += 1
        return {expert_id: count for expert_id, count in histogram.items() if count}

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


def _invalid_histogram(request_id: int | None, detail: str) -> WorkloadValidationError:
    prefix = "expert histogram"
    if request_id is not None:
        prefix += f" for request {request_id}"
    return WorkloadValidationError(f"{prefix} {detail}")
