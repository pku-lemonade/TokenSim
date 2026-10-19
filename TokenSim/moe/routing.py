"""Routing."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random
from typing import Any, Mapping, Sequence

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe.config import MoEModelConfig, RoutingConfig
from TokenSim.moe.routing_models import (
    LayerRoutingSnapshot,
    RoutingAssignment,
    RoutingPhase,
)
from TokenSim.moe.routing_synthetic import SyntheticRoutingMixin, _scaled_histogram
from TokenSim.moe.routing_validation import (
    _invalid_assignments,
    _owner_rank,
    normalize_expert_histogram,
    normalize_routing_assignments,
)
from TokenSim.utils import BoundedCache


class ExpertRouting(SyntheticRoutingMixin):
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
        self.seed = seed
        self.random = random.Random(seed)
        self._synthetic_cache = BoundedCache(self._SYNTHETIC_CACHE_LIMIT)
        self._normalized_cache = BoundedCache(self._SYNTHETIC_CACHE_LIMIT)
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

    def snapshot_for_layer(
        self,
        requests: Sequence[Request],
        *,
        iteration_id: int,
        phase: RoutingPhase,
        decode_step: int,
        layer_id: int,
        a_dp_rank: int | Mapping[int, int] = 0,
        a_tp_rank: int | Mapping[int, int] = 0,
    ) -> LayerRoutingSnapshot:
        if not self.moe_config.enabled:
            raise WorkloadValidationError(
                "routing snapshot requires an enabled MoE model"
            )
        if not requests:
            raise WorkloadValidationError(
                "routing snapshot requires at least one request"
            )
        if phase not in {"prefill", "decode"}:
            raise WorkloadValidationError(
                f"routing phase must be 'prefill' or 'decode', got {phase!r}"
            )
        for field_name, value in (
            ("iteration_id", iteration_id),
            ("decode_step", decode_step),
            ("layer_id", layer_id),
        ):
            if type(value) is not int or value < 0:
                raise WorkloadValidationError(
                    f"routing snapshot {field_name} must be a non-negative integer"
                )

        request_tokens: dict[int, tuple[int, ...]] = {}
        request_owners: dict[int, tuple[int, int]] = {}
        request_steps = (
            {
                int(request.id): int(getattr(request, "generation_idx", decode_step))
                for request in requests
            }
            if phase == "decode"
            else {}
        )
        for request in requests:
            request_id = int(request.id)
            if request_id in request_tokens:
                raise WorkloadValidationError(
                    f"routing layer contains duplicate request id {request_id}"
                )
            request_phase = "prefill" if request.is_prefill else "decode"
            if request_phase != phase:
                raise _invalid_assignments(
                    request_id,
                    f"request phase={request_phase!r} does not match snapshot phase",
                    phase=phase,
                    decode_step=decode_step,
                    layer_id=layer_id,
                )
            request_tokens[request_id] = self._token_ids(request, phase)
            request_owners[request_id] = (
                _owner_rank(a_dp_rank, request_id, "a_dp_rank"),
                _owner_rank(a_tp_rank, request_id, "a_tp_rank"),
            )

        assignment_sources = {
            int(request.id): self._provided_assignments(request, phase)
            for request in requests
        }
        has_external_assignments = any(
            source is not None for source, _ in assignment_sources.values()
        )
        histogram_sources = {
            int(request.id): self._provided_histogram(request, phase)
            for request in requests
        }

        assignments: list[RoutingAssignment] = []
        if has_external_assignments:
            for request in requests:
                request_id = int(request.id)
                source, phase_is_implicit = assignment_sources[request_id]
                if source is None:
                    raise _invalid_assignments(
                        request_id,
                        "external assignments are present for other requests in the "
                        "same layer block, but missing for this request",
                        phase=phase,
                        decode_step=decode_step,
                        layer_id=layer_id,
                    )
                owner = request_owners[request_id]
                request_assignments = normalize_routing_assignments(
                    source,
                    iteration_id=iteration_id,
                    phase=phase,
                    decode_step=request_steps.get(request_id, decode_step),
                    layer_id=layer_id,
                    request_id=request_id,
                    expected_token_ids=request_tokens[request_id],
                    expected_a_dp_rank=owner[0],
                    expected_a_tp_rank=owner[1],
                    num_experts=self.moe_config.num_experts,
                    phase_is_implicit=phase_is_implicit,
                )
                if not request_assignments:
                    raise _invalid_assignments(
                        request_id,
                        "has no records for the current phase, decode_step, and layer",
                        phase=phase,
                        decode_step=decode_step,
                        layer_id=layer_id,
                    )
                assignments.extend(request_assignments)
        elif any(value is not None for value in histogram_sources.values()):
            request_ids = sorted(
                request_id
                for request_id, value in histogram_sources.items()
                if value is not None
            )
            raise WorkloadValidationError(
                "expert histogram is statistics-only and cannot drive a routing "
                "snapshot for ownership combine; missing token owners, Expert edges, "
                f"and routing weights for requests={request_ids}, phase={phase}, "
                f"decode_step={decode_step}, layer={layer_id}"
            )
        else:
            assignments.extend(
                self._synthetic_assignments(
                    request_tokens,
                    request_owners,
                    iteration_id=iteration_id,
                    phase=phase,
                    decode_step=decode_step,
                    layer_id=layer_id,
                    request_decode_steps=request_steps,
                )
            )

        snapshot = LayerRoutingSnapshot(
            iteration_id=iteration_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            top_k=self.moe_config.num_experts_per_tok,
            original_tokens=tuple(
                (request_id, token_id)
                for request_id, token_ids in request_tokens.items()
                for token_id in token_ids
            ),
            assignments=tuple(assignments),
            request_decode_steps=tuple(sorted(request_steps.items())),
        )
        if has_external_assignments:
            self._validate_external_histograms(
                snapshot,
                requests,
                request_tokens,
                histogram_sources,
            )
        return snapshot

    def histogram_for_step(
        self,
        requests: list[Request],
        *,
        phase: RoutingPhase | None = None,
        decode_step: int | None = None,
        layer_id: int = 0,
    ) -> dict[int, int]:
        """Return legacy routing statistics without creating execution ownership."""
        histogram: dict[int, int] = {}
        if not self.moe_config.enabled or not requests:
            return histogram
        synthetic_token_count = 0
        for request in requests:
            request_phase = phase or ("prefill" if request.is_prefill else "decode")
            provided = self._provided_histogram(request, request_phase)
            if provided is None:
                synthetic_token_count += self._token_count(request, request_phase)
                continue
            request_histogram = self._request_histogram(
                request,
                phase=request_phase,
                decode_step=(
                    0
                    if decode_step is None and request_phase == "prefill"
                    else (
                        int(request.generation_idx)
                        if decode_step is None
                        else decode_step
                    )
                ),
                layer_id=layer_id,
                provided=provided,
            )
            for expert_id, count in request_histogram.items():
                histogram[expert_id] = histogram.get(expert_id, 0) + count
        if synthetic_token_count:
            for expert_id, count in self.synthetic_histogram(
                synthetic_token_count, offset=self._route_cursor
            ).items():
                histogram[expert_id] = histogram.get(expert_id, 0) + count
            self._route_cursor = (
                self._route_cursor
                + synthetic_token_count * self.moe_config.num_experts_per_tok
            ) % self.moe_config.num_experts
        return histogram

    def _request_histogram(
        self,
        request: Request,
        *,
        phase: RoutingPhase,
        decode_step: int,
        layer_id: int,
        provided: Any,
    ) -> dict[int, int]:
        token_count = self._token_count(request, phase)
        expected_route_count = token_count * self.moe_config.num_experts_per_tok
        cache_key = (
            int(request.id),
            phase,
            int(decode_step),
            int(layer_id),
            expected_route_count,
        )
        cached = self._normalized_cache.get(cache_key)
        if cached is None:
            cached = self._normalize_step_histogram(
                request,
                phase,
                provided,
                token_count,
            )
            if len(self._normalized_cache) >= self._SYNTHETIC_CACHE_LIMIT:
                self._normalized_cache.clear()
            self._normalized_cache[cache_key] = cached
        return dict(cached)

    def _normalize_step_histogram(
        self,
        request: Request,
        phase: RoutingPhase,
        provided: Any,
        token_count: int,
    ) -> dict[int, int]:
        source_tokens = request.prefill_compute_len if phase == "prefill" else 1
        histogram = normalize_expert_histogram(
            provided,
            num_experts=self.moe_config.num_experts,
            request_id=request.id,
            expected_route_count=(source_tokens * self.moe_config.num_experts_per_tok),
        )
        if token_count != source_tokens:
            histogram = _scaled_histogram(
                histogram,
                token_count / max(1, source_tokens),
                target_count=token_count * self.moe_config.num_experts_per_tok,
            )
        return histogram

    @staticmethod
    def _provided_histogram(request: Request, phase: RoutingPhase) -> Any:
        field_name = (
            "prefill_expert_histogram"
            if phase == "prefill"
            else "decode_expert_histogram"
        )
        provided = getattr(request, field_name, None)
        if provided is None:
            provided = getattr(request, "expert_histogram", None)
        return provided

    @staticmethod
    def _provided_assignments(
        request: Request, phase: RoutingPhase
    ) -> tuple[Any | None, bool]:
        field_name = (
            "prefill_routing_assignments"
            if phase == "prefill"
            else "decode_routing_assignments"
        )
        provided = getattr(request, field_name, None)
        if provided is not None:
            return provided, True
        return getattr(request, "routing_assignments", None), False

    @staticmethod
    def _token_count(request: Request, phase: RoutingPhase) -> int:
        scheduled_tokens = int(getattr(request, "scheduled_tokens", 0))
        if scheduled_tokens > 0:
            return scheduled_tokens
        if getattr(request, "needs_recompute", False):
            return max(0, int(request.recompute_tokens))
        if phase == "prefill":
            return max(0, int(request.prefill_compute_len))
        return 1

    def _token_ids(self, request: Request, phase: RoutingPhase) -> tuple[int, ...]:
        token_count = self._token_count(request, phase)
        if int(getattr(request, "scheduled_tokens", 0)) > 0:
            start = max(0, int(getattr(request, "num_computed_tokens", 0)))
            return tuple(range(start, start + token_count))
        if getattr(request, "needs_recompute", False):
            start = max(0, int(request.context_len) - token_count)
            return tuple(range(start, start + token_count))
        if phase == "decode":
            return (max(0, int(request.context_len) - 1),)
        start = max(0, int(request.context_len) - token_count)
        return tuple(range(start, start + token_count))

    def _validate_external_histograms(
        self,
        snapshot: LayerRoutingSnapshot,
        requests: Sequence[Request],
        request_tokens: Mapping[int, tuple[int, ...]],
        histogram_sources: Mapping[int, Any],
    ) -> None:
        for request in requests:
            request_id = int(request.id)
            provided = histogram_sources[request_id]
            if provided is None:
                continue
            expected = self._normalize_step_histogram(
                request,
                snapshot.phase,
                provided,
                len(request_tokens[request_id]),
            )
            actual = Counter(
                assignment.expert_id
                for assignment in snapshot.assignments
                if assignment.request_id == request_id
            )
            if expected != dict(actual):
                raise _invalid_assignments(
                    request_id,
                    f"histogram mismatch: assignments aggregate to {dict(actual)!r}, "
                    f"provided histogram is {expected!r}",
                    phase=snapshot.phase,
                    decode_step=snapshot.decode_step,
                    layer_id=snapshot.layer_id,
                )
