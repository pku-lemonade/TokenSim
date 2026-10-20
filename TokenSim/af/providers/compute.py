"""Compute-query validation, component provenance and service composition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping

from TokenSim.af.caching.estimates import cached_estimate
from TokenSim.af.providers.queries import (
    AFComputeQuery,
    AFServiceEstimate,
    AF_MATCH_TYPES,
)
from TokenSim.comm.traffic import validated_bytes
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype
from TokenSim.latency.operator_table import OperatorEstimate, OperatorResolver


_MATCH_RANK = {name: rank for rank, name in enumerate(AF_MATCH_TYPES)}


_COLLECTIVE_MATCH = {"measured": "exact"}


def _worst_match(match_types: Iterable[str]) -> str:
    """The least trustworthy match type of a composition (exact < ... < analytical)."""
    worst = "exact"
    for match_type in match_types:
        if _MATCH_RANK.get(match_type, len(AF_MATCH_TYPES)) > _MATCH_RANK[worst]:
            worst = match_type if match_type in _MATCH_RANK else "analytical"
    return worst


def _compose_source_id(source_ids: Iterable[str]) -> str:
    unique = tuple(dict.fromkeys(source_ids))
    if not unique:
        return "composition:empty"
    if len(unique) == 1:
        return unique[0]
    return "compose:" + ":".join(unique)


def _compose_dataset_version(versions: Iterable[str]) -> str:
    unique = tuple(dict.fromkeys(v for v in versions if v))
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "compose:" + ":".join(unique)


def _estimate_dataset_version(
    resolver: OperatorResolver, estimate: OperatorEstimate
) -> str:
    """Label analytical estimates separately from a package row."""
    if estimate.match_type == "analytical":
        return f"analytical:{resolver.device.device_id}"
    return resolver.dataset_version or f"unknown:{resolver.device.device_id}"


@dataclass(frozen=True)
class OperatorComponent:
    """One operator of an AF stage and how it was priced."""

    name: str
    table: str
    key: Mapping[str, Any]
    latency_us: float
    match_type: str
    source_id: str
    dataset_version: str = ""
    covered_operations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "key": dict(self.key),
            "latency_us": self.latency_us,
            "match_type": self.match_type,
            "source_id": self.source_id,
            "dataset_version": self.dataset_version,
            "covered_operations": self.covered_operations,
        }


def prefill_components(provider, resolver, query, tokens, common):
    lengths = tuple(query.context_lengths) or (tokens,)
    queries = tuple(query.query_lengths)
    if not queries:
        if sum(lengths) == tokens:
            queries = lengths
        elif tokens % len(lengths) == 0:
            queries = (tokens // len(lengths),) * len(lengths)
        else:
            raise ValueError("uneven prefill requires per-request query_lengths")
    if (
        len(queries) != len(lengths)
        or sum(queries) != tokens
        or any(
            type(q) is not int or q < 1 or type(k) is not int or k < q
            for q, k in zip(queries, lengths)
        )
    ):
        raise ValueError("prefill query lengths must fit their per-request KV contexts")
    components = []
    for (q, k), count in sorted(Counter(zip(queries, lengths)).items()):
        key = {**common, "batch_size": count, "input_seq_len": q}
        if k > q:
            key["context_len"] = k
        component = provider._component(
            resolver, "attention_core", "context_attention", key
        )
        if k > q and component.match_type != "analytical":
            base = resolver.analytical_reference_us(
                "context_attention",
                {name: value for name, value in key.items() if name != "context_len"},
            )
            full = resolver.analytical_reference_us("context_attention", key)
            if base is None or full is None or base <= 0:
                raise ConfigurationError(
                    "cached prefill requires an analytical growth reference"
                )
            previous = component.match_type
            for counter in (
                resolver.stats.match_type_counts,
                resolver.stats.table_match_counts["context_attention"],
            ):
                counter[previous] -= 1
                if counter[previous] == 0:
                    del counter[previous]
            resolver.stats.record("context_attention", "extrapolated")
            component = replace(
                component,
                latency_us=component.latency_us * max(1.0, full / base),
                match_type="extrapolated",
                source_id=f"prefix_scaled:{component.source_id}",
            )
        components.append(component)
    return components


class ComputeEstimateMixin:
    @cached_estimate
    def estimate(self, query: AFComputeQuery) -> AFServiceEstimate:
        tokens, tp_size = self._validate_local_work(query)
        resolver = self.resolver_for(query.device_type)
        dtype = normalize_dtype(query.dtype)
        weight_dtype = normalize_dtype(query.weight_dtype or dtype)
        if query.stage in {"encode", "decode"}:
            source = normalize_dtype(query.source_dtype)
            target = normalize_dtype(query.target_dtype)
            elements = query.local_shape.get("elements")
            if (
                type(elements) is not int
                or elements < 1
                or elements != tokens * query.local_shape["hidden_size"]
            ):
                raise ValueError(
                    "dtype conversion requires an exact positive element count"
                )
            if source == target:
                raise ValueError(
                    "dtype conversion requires different source and target dtypes"
                )
            components = [
                self._elementwise(
                    resolver,
                    query.operation,
                    f"cast_{source}_to_{target}",
                    tokens,
                    elements // tokens,
                    target,
                )
            ]
        elif query.stage == "layout" and query.operation in {
            "embedding_lookup",
            "final_norm",
            "lm_head",
            "token_selection",
        }:
            if query.operation == "lm_head":
                weight_dtype = self.model.operator_dtype(
                    "embedding_lm_head", weight_dtype
                )
                if self.model.vocab_size % tp_size:
                    raise ConfigurationError("vocab_size must divide attention TP")
                components = [
                    self._gemm(
                        resolver,
                        "lm_head",
                        tokens,
                        self.model.vocab_size // tp_size,
                        self.model.hidden_size,
                        weight_dtype,
                    )
                ]
            else:
                operation = {
                    "embedding_lookup": "embedding_lookup",
                    "final_norm": "rmsnorm",
                    "token_selection": "argmax",
                }[query.operation]
                width = (
                    self.model.vocab_size
                    if operation == "argmax"
                    else self.model.hidden_size
                )
                components = [
                    self._elementwise(
                        resolver, query.operation, operation, tokens, width, dtype
                    )
                ]
        elif query.stage == "layout" and query.operation == "data_reorder":
            size = validated_bytes(query.local_shape["reorder_bytes"], query.dtype)
            components = [
                self._elementwise(
                    resolver,
                    "data_reorder",
                    "data_reorder",
                    1,
                    size // dtype_size_bytes(query.dtype, role="activation_dtype"),
                    dtype,
                )
            ]
        elif self.execution_protocol == "vllm_afd" and query.stage in {
            "router",
            "routed_expert",
            "weighted_combine",
        }:
            # These compositions preserve exact expert histograms and exclude
            # norm/residual work owned by A. The routed_experts contract does
            # not include combine, which is priced separately on F.
            components = self._mla_stage(resolver, query, tokens, tp_size, dtype)
        elif self.model.attention_type == "mla":
            components = self._mla_stage(resolver, query, tokens, tp_size, dtype)
        elif query.stage in {"a", "attention"}:
            components = self._attention_components(
                resolver, query, tokens, tp_size, dtype
            )
        elif query.stage == "router":
            components = self._router_components(resolver, query, tokens, dtype)
        elif query.stage in {"dense_ffn", "shared_expert"}:
            weight_dtype = self.model.operator_dtype(query.stage, weight_dtype)
            width = (
                self.model.intermediate_size
                if query.stage == "dense_ffn"
                else self.model.moe_intermediate_size
                * self.model.moe.num_shared_experts
            )
            local_width = math.ceil(width / tp_size)
            components = [
                self._gemm(
                    resolver,
                    "gate_up",
                    tokens,
                    self.model.ffn_projection_count * local_width,
                    self.model.hidden_size,
                    weight_dtype,
                )
            ]
            if self.model.gated:
                components.append(
                    self._elementwise(
                        resolver,
                        self.model.activation,
                        self.model.activation,
                        tokens,
                        local_width,
                        dtype,
                    )
                )
            components.append(
                self._gemm(
                    resolver,
                    "down",
                    tokens,
                    self.model.hidden_size,
                    local_width,
                    weight_dtype,
                )
            )
        elif query.stage == "layout" and query.operation == "post_attention_norm":
            components = [
                self._elementwise(
                    resolver,
                    "post_attention_norm",
                    "rmsnorm",
                    tokens,
                    self.model.hidden_size,
                    dtype,
                )
            ]
        elif query.stage in {"expert", "routed_expert"}:
            components = self._expert_components(
                resolver, query, tokens, tp_size, dtype
            )
        elif query.stage == "weighted_combine":
            edges = query.input_edge_count or tokens * max(
                1, self.model.moe.num_experts_per_tok
            )
            components = [
                self._elementwise(
                    resolver,
                    "weighted_combine",
                    "weighted_combine",
                    edges,
                    int(query.local_shape["hidden_size"]),
                    dtype,
                )
            ]
        elif query.stage == "residual":
            components = [
                self._elementwise(
                    resolver,
                    "residual_add",
                    "residual_add",
                    tokens,
                    int(query.local_shape["hidden_size"]),
                    dtype,
                )
            ]
        else:
            raise ValueError(f"unsupported AF compute stage={query.stage!r}")
        if self.model.architecture == "gemma4_text":
            h = self.model.hidden_size
            if query.stage == "layout" and query.operation == "post_attention_norm":
                components = [
                    self._elementwise(resolver, name, op, tokens, h, dtype)
                    for name, op in (
                        ("post_attn_norm", "rmsnorm"),
                        ("attn_residual", "residual_add"),
                        ("pre_ffn_norm", "rmsnorm"),
                    )
                ]
            elif query.stage == "residual":
                components.insert(
                    0,
                    self._elementwise(
                        resolver, "post_ffn_norm", "rmsnorm", tokens, h, dtype
                    ),
                )
        latency_us = sum(component.latency_us for component in components)
        match_type = _worst_match(component.match_type for component in components)
        return AFServiceEstimate(
            service_duration=latency_us / 1e6,
            source_id=_compose_source_id(
                component.source_id for component in components
            ),
            dataset_version=(
                _compose_dataset_version(
                    component.dataset_version for component in components
                )
                if self.model.attention_type == "mla"
                else resolver.dataset_version
                or f"analytical:{resolver.device.device_id}"
            ),
            match_type=match_type,
            details={
                "provider": "operator_table",
                "device_id": resolver.device.device_id,
                "operator_backend": resolver.operator_backend,
                "dataset_version": resolver.dataset_version,
                "match_type": match_type,
                "stage": query.stage,
                "operation": query.operation,
                "logical_participant": query.logical_participant,
                "physical_worker": query.physical_worker,
                "local_shape": dict(query.local_shape),
                "local_token_count": tokens,
                "expert_ids": tuple(query.expert_ids),
                "expert_workload": dict(query.expert_workload),
                "components": [component.as_dict() for component in components],
                "total_latency_us": latency_us,
                "covered_operations": tuple(
                    sorted(
                        {
                            operation
                            for component in components
                            for operation in component.covered_operations
                        }
                    )
                ),
            },
        )

    def _validate_local_work(self, query: AFComputeQuery) -> tuple[int, int]:
        context = (
            f"operation={query.operation!r}, stage={query.stage!r}, "
            f"domain={query.source_domain!r}, dp_rank={query.dp_rank}, "
            f"tp_rank={query.tp_rank}, ep_rank={query.ep_rank}"
        )
        if query.local_token_count is None or query.local_token_count < 1:
            raise ValueError(
                f"AF compute query missing positive local_token_count; {context}"
            )
        if query.token_count != query.local_token_count:
            raise ValueError(
                f"AF compute query token_count={query.token_count} does not match "
                f"local_token_count={query.local_token_count}; {context}"
            )
        shape = query.local_shape
        required = ("tokens", "hidden_size", "global_hidden_size", "tp_size")
        missing = sorted(set(required) - set(shape))
        if missing:
            raise ValueError(
                f"AF compute query missing local_shape fields {missing}; {context}"
            )
        values = {name: shape[name] for name in required}
        if any(type(value) is not int or value < 1 for value in values.values()):
            raise ValueError(
                f"AF compute query local_shape must contain positive integers; "
                f"local_shape={dict(shape)!r}; {context}"
            )
        if values["tokens"] != query.local_token_count:
            raise ValueError(
                f"AF compute query local_shape tokens={values['tokens']} does not "
                f"match local_token_count={query.local_token_count}; {context}"
            )
        if values["global_hidden_size"] != self.model.hidden_size:
            raise ValueError(
                f"AF compute query global_hidden_size={values['global_hidden_size']} "
                f"does not match model hidden_size={self.model.hidden_size}; {context}"
            )
        expected_hidden = (
            values["hidden_size"]
            if query.stage in {"encode", "decode"}
            else (
                self.model.hidden_size
                if query.stage
                in {
                    "router",
                    "attention",
                    "dense_ffn",
                    "shared_expert",
                    "routed_expert",
                }
                or (
                    query.stage == "layout"
                    and query.operation
                    in {
                        "post_attention_norm",
                        "embedding_lookup",
                        "final_norm",
                        "lm_head",
                        "token_selection",
                    }
                )
                else math.ceil(self.model.hidden_size / values["tp_size"])
            )
        )
        if values["hidden_size"] != expected_hidden:
            raise ValueError(
                f"AF compute query local hidden_size={values['hidden_size']} does not "
                f"match expected={expected_hidden}; {context}"
            )
        return query.local_token_count, values["tp_size"]

    @staticmethod
    def _validate_expert_workload(
        query: AFComputeQuery, tokens: int, num_experts: int
    ) -> tuple[tuple[int, int], ...]:
        workload = tuple(query.expert_workload)
        if not workload:
            raise ValueError(
                "MoE expert_rank_compute requires non-empty expert_workload; "
                f"domain={query.source_domain!r}, ep_rank={query.ep_rank}, "
                f"local_token_count={tokens}"
            )
        expert_ids = [expert_id for expert_id, _ in workload]
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError("expert_workload contains duplicate expert ids")
        if any(expert_id < 0 or count < 1 for expert_id, count in workload):
            raise ValueError(
                "expert_workload requires non-negative ids and positive counts"
            )
        if any(expert_id >= num_experts for expert_id in expert_ids):
            raise ValueError(
                f"expert_workload contains expert id outside model range [0, {num_experts})"
            )
        if tuple(query.expert_ids) != tuple(expert_ids):
            raise ValueError(
                f"expert_ids={tuple(query.expert_ids)!r} does not match "
                f"expert_workload ids={tuple(expert_ids)!r}"
            )
        workload_tokens = sum(count for _, count in workload)
        if workload_tokens != tokens:
            raise ValueError(
                f"expert_workload total={workload_tokens} does not match local_token_count={tokens}"
            )
        return workload
