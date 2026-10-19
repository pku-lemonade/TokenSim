"""Immutable token-level expert assignments and routing snapshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Literal

from TokenSim.errors import WorkloadValidationError


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
    request_decode_steps: tuple[tuple[int, int], ...] = ()
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
        request_steps = dict(self.request_decode_steps)
        if len(request_steps) != len(self.request_decode_steps) or any(
            type(request) is not int or request < 0 or type(step) is not int or step < 0
            for request, step in request_steps.items()
        ):
            raise WorkloadValidationError("invalid per-request decode steps")
        if request_steps and set(request_steps) != {
            request for request, _ in canonical_tokens
        }:
            raise WorkloadValidationError(
                "per-request decode steps must cover snapshot requests"
            )
        object.__setattr__(
            self, "request_decode_steps", tuple(sorted(request_steps.items()))
        )

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
            expected_key = (
                self.iteration_id,
                self.phase,
                request_steps.get(assignment.request_id, self.decode_step),
                self.layer_id,
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
