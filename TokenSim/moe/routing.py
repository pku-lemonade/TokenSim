from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe.config import MoEModelConfig, RoutingConfig


RoutingPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class RoutingAssignment:
    iteration_id: int
    phase: RoutingPhase
    decode_step: int
    layer_id: int
    request_id: int
    token_id: int
    top_k_index: int
    expert_id: int
    routing_weight: float
    a_dp_rank: int
    a_tp_rank: int

    def __post_init__(self) -> None:
        for field_name in (
            "iteration_id",
            "decode_step",
            "layer_id",
            "request_id",
            "token_id",
            "top_k_index",
            "expert_id",
            "a_dp_rank",
            "a_tp_rank",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise WorkloadValidationError(
                    f"routing assignment {field_name} must be a non-negative integer"
                )
        if self.phase not in {"prefill", "decode"}:
            raise WorkloadValidationError(
                f"routing assignment phase must be 'prefill' or 'decode', got {self.phase!r}"
            )
        weight = self.routing_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise WorkloadValidationError(
                "routing assignment routing_weight must be finite and non-negative"
            )
        object.__setattr__(self, "routing_weight", float(weight))

    @property
    def identity(self) -> tuple[int, str, int, int, int, int, int]:
        return (
            self.iteration_id,
            self.phase,
            self.decode_step,
            self.layer_id,
            self.request_id,
            self.token_id,
            self.top_k_index,
        )


@dataclass(frozen=True)
class LayerRoutingSnapshot:
    iteration_id: int
    phase: RoutingPhase
    decode_step: int
    layer_id: int
    top_k: int
    original_tokens: tuple[tuple[int, int], ...]
    assignments: tuple[RoutingAssignment, ...]
    expert_histogram: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("iteration_id", "decode_step", "layer_id"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise WorkloadValidationError(
                    f"routing snapshot {field_name} must be a non-negative integer"
                )
        if self.phase not in {"prefill", "decode"}:
            raise WorkloadValidationError(
                f"routing snapshot phase must be 'prefill' or 'decode', got {self.phase!r}"
            )
        if type(self.top_k) is not int or self.top_k < 1:
            raise WorkloadValidationError("routing snapshot top_k must be positive")

        original_tokens: list[tuple[int, int]] = []
        for token_index, token in enumerate(self.original_tokens):
            if not isinstance(token, (list, tuple)) or len(token) != 2:
                raise WorkloadValidationError(
                    f"routing snapshot original token {token_index} must be "
                    "a (request_id, token_id) pair"
                )
            request_id, token_id = token
            if (
                type(request_id) is not int
                or request_id < 0
                or type(token_id) is not int
                or token_id < 0
            ):
                raise WorkloadValidationError(
                    f"routing snapshot original token {token_index} must contain "
                    "non-negative integer ids"
                )
            original_tokens.append((request_id, token_id))
        canonical_tokens = tuple(original_tokens)
        canonical_assignments = tuple(self.assignments)
        object.__setattr__(self, "original_tokens", canonical_tokens)
        object.__setattr__(self, "assignments", canonical_assignments)

        if len(canonical_tokens) != len(set(canonical_tokens)):
            raise WorkloadValidationError(
                "routing snapshot contains duplicate original token identities"
            )

        expected_key = (
            self.iteration_id,
            self.phase,
            self.decode_step,
            self.layer_id,
        )
        by_token: dict[tuple[int, int], list[RoutingAssignment]] = {
            token: [] for token in canonical_tokens
        }
        identities: set[tuple[int, str, int, int, int, int, int]] = set()
        histogram: Counter[int] = Counter()
        for assignment in canonical_assignments:
            if not isinstance(assignment, RoutingAssignment):
                raise WorkloadValidationError(
                    "routing snapshot assignments must contain RoutingAssignment objects"
                )
            actual_key = (
                assignment.iteration_id,
                assignment.phase,
                assignment.decode_step,
                assignment.layer_id,
            )
            if actual_key != expected_key:
                raise WorkloadValidationError(
                    "routing assignment belongs to a different phase, step, layer, "
                    f"or iteration: expected={expected_key!r}, actual={actual_key!r}"
                )
            token = (assignment.request_id, assignment.token_id)
            if token not in by_token:
                raise WorkloadValidationError(
                    f"routing assignment references unknown original token {token!r}"
                )
            if assignment.identity in identities:
                raise WorkloadValidationError(
                    f"duplicate routing assignment identity {assignment.identity!r}"
                )
            identities.add(assignment.identity)
            by_token[token].append(assignment)
            histogram[assignment.expert_id] += 1

        for token, token_assignments in by_token.items():
            if len(token_assignments) != self.top_k:
                raise WorkloadValidationError(
                    f"routing token {token!r} must have exactly {self.top_k} "
                    f"assignments; got {len(token_assignments)}"
                )
            indexes = {item.top_k_index for item in token_assignments}
            if indexes != set(range(self.top_k)):
                raise WorkloadValidationError(
                    f"routing token {token!r} top_k_index values must be "
                    f"{list(range(self.top_k))}; got {sorted(indexes)}"
                )
            experts = {item.expert_id for item in token_assignments}
            if len(experts) != self.top_k:
                raise WorkloadValidationError(
                    f"routing token {token!r} must select {self.top_k} distinct experts"
                )
            owners = {(item.a_dp_rank, item.a_tp_rank) for item in token_assignments}
            if len(owners) != 1:
                raise WorkloadValidationError(
                    f"routing token {token!r} has inconsistent A owners {sorted(owners)}"
                )

        object.__setattr__(
            self,
            "expert_histogram",
            tuple(sorted(histogram.items())),
        )

    @property
    def histogram(self) -> dict[int, int]:
        return dict(self.expert_histogram)

    @property
    def original_token_count(self) -> int:
        return len(self.original_tokens)

    @property
    def assignment_count(self) -> int:
        return len(self.assignments)


def normalize_expert_histogram(
    value: Any,
    *,
    num_experts: int,
    request_id: int | None = None,
    expected_route_count: int | None = None,
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
    if (
        expected_route_count is not None
        and sum(histogram.values()) != expected_route_count
    ):
        raise _invalid_histogram(
            request_id,
            f"total routed count must equal {expected_route_count}; "
            f"got {sum(histogram.values())}",
        )
    return histogram


def normalize_routing_assignments(
    value: Any,
    *,
    iteration_id: int,
    phase: RoutingPhase,
    decode_step: int,
    layer_id: int,
    request_id: int,
    expected_token_ids: Sequence[int],
    expected_a_dp_rank: int,
    expected_a_tp_rank: int,
    num_experts: int,
    phase_is_implicit: bool = False,
) -> tuple[RoutingAssignment, ...]:
    if not isinstance(value, (list, tuple)):
        raise _invalid_assignments(
            request_id,
            "must be a list of assignment objects",
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
        )

    expected_tokens = set(expected_token_ids)
    selected: list[RoutingAssignment] = []
    for record_index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise _invalid_assignments(
                request_id,
                f"record {record_index} must be an object",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        record_phase = record.get("phase")
        if record_phase is None and phase_is_implicit:
            record_phase = phase
        if record_phase not in {"prefill", "decode"}:
            raise _invalid_assignments(
                request_id,
                f"record {record_index} requires phase='prefill' or 'decode'",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        record_step = _required_non_negative_int(
            record,
            "decode_step",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        record_layer = _required_non_negative_int(
            record,
            "layer_id",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        if (record_phase, record_step, record_layer) != (
            phase,
            decode_step,
            layer_id,
        ):
            continue

        record_request_id = record.get("request_id", request_id)
        if record_request_id != request_id:
            raise _invalid_assignments(
                request_id,
                f"record {record_index} request_id={record_request_id!r} does not "
                f"match enclosing request",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        token_id = _required_non_negative_int(
            record,
            "token_id",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        if token_id not in expected_tokens:
            raise _invalid_assignments(
                request_id,
                f"record {record_index} token_id={token_id} is not in the current "
                f"token set {sorted(expected_tokens)}",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        expert_id = _required_non_negative_int(
            record,
            "expert_id",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        if expert_id >= num_experts:
            raise _invalid_assignments(
                request_id,
                f"record {record_index} expert_id={expert_id} outside "
                f"[0, {num_experts})",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        top_k_index = _required_non_negative_int(
            record,
            "top_k_index",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        a_dp_rank = _required_owner_rank(
            record,
            "a_dp_rank",
            "a_dp_owner",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        a_tp_rank = _required_owner_rank(
            record,
            "a_tp_rank",
            "a_tp_owner",
            request_id=request_id,
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
            record_index=record_index,
        )
        if (a_dp_rank, a_tp_rank) != (expected_a_dp_rank, expected_a_tp_rank):
            raise _invalid_assignments(
                request_id,
                f"record {record_index} A owner={(a_dp_rank, a_tp_rank)!r} does not "
                f"match runtime owner={(expected_a_dp_rank, expected_a_tp_rank)!r}",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        if "routing_weight" in record:
            routing_weight = record["routing_weight"]
        elif "weight" in record:
            routing_weight = record["weight"]
        else:
            raise _invalid_assignments(
                request_id,
                f"record {record_index} is missing routing_weight",
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
            )
        selected.append(
            RoutingAssignment(
                iteration_id=iteration_id,
                phase=phase,
                decode_step=decode_step,
                layer_id=layer_id,
                request_id=request_id,
                token_id=token_id,
                top_k_index=top_k_index,
                expert_id=expert_id,
                routing_weight=routing_weight,
                a_dp_rank=a_dp_rank,
                a_tp_rank=a_tp_rank,
            )
        )
    return tuple(selected)


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
        self.routing_config = replace(
            routing_config or moe_config.routing or RoutingConfig()
        )
        self.seed = seed
        self.random = random.Random(seed)
        self._synthetic_cache: dict[int, dict[int, int]] = {}
        self._normalized_cache: dict[tuple[int, str, int, int, int], dict[int, int]] = (
            {}
        )

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
                    decode_step=decode_step,
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
                synthetic_token_count
            ).items():
                histogram[expert_id] = histogram.get(expert_id, 0) + count
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
            expected_route_count=(
                source_tokens * self.moe_config.num_experts_per_tok
            ),
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

    def _synthetic_assignments(
        self,
        request_tokens: Mapping[int, tuple[int, ...]],
        request_owners: Mapping[int, tuple[int, int]],
        *,
        iteration_id: int,
        phase: RoutingPhase,
        decode_step: int,
        layer_id: int,
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
                        decode_step=decode_step,
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

    def _experts_for_token(
        self, request_id: int, token_id: int, phase: RoutingPhase, layer_id: int
    ) -> tuple[int, ...]:
        """Stateless synthetic routing; scheduling/batch identities are excluded."""
        count = self.moe_config.num_experts
        top_k = self.moe_config.num_experts_per_tok
        if self.routing_config.distribution == "uniform":
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


def _scaled_histogram(
    histogram: dict[int, int],
    share: float,
    *,
    target_count: int | None = None,
) -> dict[int, int]:
    raw = {expert_id: count * share for expert_id, count in histogram.items()}
    scaled = {expert_id: int(value) for expert_id, value in raw.items()}
    target = round(sum(histogram.values()) * share) if target_count is None else target_count
    remainder = target - sum(scaled.values())
    for expert_id in sorted(raw, key=lambda item: (-(raw[item] - scaled[item]), item)):
        if remainder <= 0:
            break
        scaled[expert_id] += 1
        remainder -= 1
    return {expert_id: count for expert_id, count in scaled.items() if count > 0}


def _invalid_histogram(request_id: int | None, detail: str) -> WorkloadValidationError:
    prefix = "expert histogram"
    if request_id is not None:
        prefix += f" for request {request_id}"
    return WorkloadValidationError(f"{prefix} {detail}")


def _invalid_assignments(
    request_id: int | None,
    detail: str,
    *,
    phase: str,
    decode_step: int,
    layer_id: int,
) -> WorkloadValidationError:
    request_text = "" if request_id is None else f" for request {request_id}"
    return WorkloadValidationError(
        f"routing assignments{request_text} at phase={phase}, "
        f"decode_step={decode_step}, layer={layer_id} {detail}"
    )


def _required_non_negative_int(
    record: Mapping[str, Any],
    field_name: str,
    *,
    request_id: int,
    phase: RoutingPhase,
    decode_step: int,
    layer_id: int,
    record_index: int,
) -> int:
    value = record.get(field_name)
    if type(value) is not int or value < 0:
        raise _invalid_assignments(
            request_id,
            f"record {record_index} field {field_name} must be a non-negative integer",
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
        )
    return value


def _required_owner_rank(
    record: Mapping[str, Any],
    field_name: str,
    alias: str,
    *,
    request_id: int,
    phase: RoutingPhase,
    decode_step: int,
    layer_id: int,
    record_index: int,
) -> int:
    if field_name in record:
        value = record[field_name]
    elif alias in record:
        value = record[alias]
    else:
        raise _invalid_assignments(
            request_id,
            f"record {record_index} is missing {field_name} owner",
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
        )
    if type(value) is not int or value < 0:
        raise _invalid_assignments(
            request_id,
            f"record {record_index} field {field_name} must be a non-negative integer",
            phase=phase,
            decode_step=decode_step,
            layer_id=layer_id,
        )
    return value


def _owner_rank(
    value: int | Mapping[int, int], request_id: int, field_name: str
) -> int:
    if isinstance(value, Mapping):
        if request_id not in value:
            raise WorkloadValidationError(
                f"routing owner mapping {field_name} is missing request {request_id}"
            )
        value = value[request_id]
    if type(value) is not int or value < 0:
        raise WorkloadValidationError(
            f"routing owner {field_name} must be a non-negative integer"
        )
    return value
