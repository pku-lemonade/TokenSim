"""Routing validation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from TokenSim.errors import WorkloadValidationError
from TokenSim.moe.routing_models import RoutingAssignment, RoutingPhase


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
