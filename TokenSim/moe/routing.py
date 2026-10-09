from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from typing import Any, Mapping

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
        elif all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
            items = [(item[0], item[1]) for item in value]
        else:
            raise _invalid_histogram(
                request_id, "must be a dict or [expert, count] list"
            )
    else:
        raise _invalid_histogram(request_id, "must be a dict or list")

    histogram: dict[int, int] = {}
    for expert_id_raw, count_raw in items:
        try:
            expert_id = int(expert_id_raw)
            count = int(count_raw)
        except (TypeError, ValueError) as exc:
            raise _invalid_histogram(
                request_id, "expert ids and counts must be integers"
            ) from exc
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
        rank_to_experts: Mapping[int, list[int]] | None = None,
    ) -> None:
        self.moe_config = moe_config
        self.routing_config = replace(
            routing_config or moe_config.routing or RoutingConfig()
        )
        self.random = random.Random(seed)
        self._synthetic_cache: dict[tuple[int, int], dict[int, int]] = {}
        self._normalized_cache: dict[tuple[int, str, int], dict[int, int]] = {}
        count = self.moe_config.num_experts
        if rank_to_experts:
            # Distribute a partial round across EP ranks before giving a
            # second route to a neighboring expert on the same rank.
            groups = [rank_to_experts[r] for r in sorted(rank_to_experts)]
            self.expert_order = tuple(
                experts[i]
                for i in range(max(map(len, groups), default=0))
                for experts in groups
                if i < len(experts)
            )
            if sorted(self.expert_order) != list(range(count)):
                raise WorkloadValidationError(
                    "expert placement must cover every expert once"
                )
        else:
            self.expert_order = tuple(range(count))
        # Continue the partial round across steps so small batches do not
        # permanently favor the first ranks. This cursor affects stats only
        # when the latency model uses a fixed measured workload distribution.
        self._route_cursor = seed % count if count else 0

    def histogram_for_step(self, requests: list[Request]) -> dict[int, int]:
        histogram: dict[int, int] = {}
        if not self.moe_config.enabled or not requests:
            return histogram
        synthetic_tokens = 0
        for request in requests:
            if self._provided_histogram(request) is None:
                synthetic_tokens += self._step_token_count(request)
                continue
            request_histogram = self._request_histogram(request)
            for expert_id, count in request_histogram.items():
                histogram[expert_id] = histogram.get(expert_id, 0) + count
        if synthetic_tokens:
            synthetic = self.synthetic_histogram(
                synthetic_tokens, offset=self._route_cursor
            )
            if histogram:
                for expert_id, count in synthetic.items():
                    histogram[expert_id] = histogram.get(expert_id, 0) + count
            else:
                # Histograms are read-only; share the cached one.
                histogram = synthetic
            self._route_cursor = (
                self._route_cursor
                + synthetic_tokens * self.moe_config.num_experts_per_tok
            ) % self.moe_config.num_experts
        return histogram

    @staticmethod
    def _provided_histogram(request: Request):
        provided = None
        if request.is_prefill:
            provided = getattr(request, "prefill_expert_histogram", None)
        else:
            provided = getattr(request, "decode_expert_histogram", None)
        if provided is None:
            provided = getattr(request, "expert_histogram", None)
        return provided

    def _request_histogram(self, request: Request) -> dict[int, int]:
        provided = self._provided_histogram(request)
        token_count = self._step_token_count(request)
        if provided is not None:
            # Decode re-reads the same trace-provided histogram every step;
            # normalize once per request. Callers must not mutate the result.
            identity = (
                request.id,
                "prefill" if request.is_prefill else "decode",
                id(provided),
            )
            cached = self._normalized_cache.get(identity)
            if cached is None:
                cached = normalize_expert_histogram(
                    provided,
                    num_experts=self.moe_config.num_experts,
                    request_id=request.id,
                )
                if len(self._normalized_cache) >= self._SYNTHETIC_CACHE_LIMIT:
                    self._normalized_cache.clear()
                self._normalized_cache[identity] = cached
            total = request.prefill_compute_len if request.is_prefill else 1
            if 0 < token_count < total:
                # A prefill chunk routes its share of the prompt's histogram.
                return _scaled_histogram(cached, token_count / total)
            return cached
        return self.synthetic_histogram(token_count)

    @staticmethod
    def _step_token_count(request: Request) -> int:
        """Tokens the request routes in this step (``Request.step_tokens``)."""
        return int(request.step_tokens)

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
            if len(self._synthetic_cache) >= self._SYNTHETIC_CACHE_LIMIT:
                self._synthetic_cache.clear()
            self._synthetic_cache[identity] = cached
        # Callers treat histograms as read-only; returning the cached dict is safe.
        return cached

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


def _scaled_histogram(histogram: dict[int, int], share: float) -> dict[int, int]:
    scaled = {
        expert_id: int(round(count * share)) for expert_id, count in histogram.items()
    }
    return {expert_id: count for expert_id, count in scaled.items() if count > 0}


def _invalid_histogram(request_id: int | None, detail: str) -> WorkloadValidationError:
    prefix = "expert histogram"
    if request_id is not None:
        prefix += f" for request {request_id}"
    return WorkloadValidationError(f"{prefix} {detail}")
