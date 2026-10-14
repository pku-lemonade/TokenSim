"""AF (attention-FFN disaggregation) latency providers on the operator-table stack.

The AF executor asks two providers for service times: one for rank-local
compute (``a`` = attention block, ``router`` = gating on the A leader,
``expert`` = routed experts or dense FFN on an F rank) and one for the
transfers between the two domains (``af_a_to_f`` / ``af_f_to_a``) plus the
TP collectives inside a domain.

Both providers reuse the project's hardware catalog:

* :class:`OperatorTableAFComputeProvider` decomposes a stage into the same
  GEMM / attention / MoE / elementwise operators as the step-level latency
  backend and answers each one through :class:`OperatorResolver` (device
  operator tables first, device-family analytical model as fallback).
* :class:`TopologyAFTransferProvider` prices physical traffic with node-domain
  phases, and flat traffic with the catalog link's alpha-beta parameters.

Service times are pure kernel/link times; queueing on shared workers and
links is produced by the executor's SimPy resources.
"""

from __future__ import annotations

import math
import json
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.af.latency import (
    AF_MATCH_TYPES,
    AFComputeQuery,
    AFServiceEstimate,
    AFTransferQuery,
)
from TokenSim.af.snapshot import ConfigurationOwner, HardwareSnapshot, snapshot
from TokenSim.af.estimate_cache import (
    cached_estimate,
    cached_phase_estimate,
    cached_plan,
    initialize_cache,
    traffic_feature_fingerprint,
)
from TokenSim.comm.traffic import (
    all_to_all_plan,
    collective_plan,
    point_to_point_plan,
    ring_volume,
    validated_bytes,
    validate_all_to_all_parameters,
)
from TokenSim.hardware.domains import (
    CommunicationDomainConfig,
    MODEL_VERSION,
    physical_domain_config,
)
from TokenSim.hardware.placement import validate_domain_placement
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.hardware.device import normalize_dtype
from TokenSim.hardware.links import LinkClass
from TokenSim.latency.operator_table import (
    FALLBACK_POLICIES,
    _MOE_DISTRIBUTION_CANDIDATES,
    OperatorEstimate,
    OperatorResolver,
    OperatorStats,
)
from TokenSim.operator_data.lookup import LookupPolicy

TRANSFER_OPERATIONS = (
    "af_a_to_f",
    "af_f_to_a",
    "af_control_length",
    "af_control_payload",
    "af_router_metadata",
    "ep_dispatch",
    "ep_combine",
    "tp_collective",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
)
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


class OperatorTableAFComputeProvider(ConfigurationOwner):
    """Price AF compute stages from the device operator tables.

    Stage decomposition (``T`` = this rank's tokens, ``h`` = hidden size,
    ``tp`` = the domain's tensor-parallel size taken from ``local_shape``):

    ``a``      rmsnorm(T, h) + QKV GEMM(T, (q+2kv)/tp, h) + RoPE + attention
               core (prefill: ``context_attention`` per request length;
               decode: ``generation_attention`` per context length) +
               output GEMM(T, h, q/tp) + residual add
    ``router`` rmsnorm(T, h) + router GEMM(T, experts, h) for MoE models
    ``expert`` MoE: one ``moe`` lookup whose ``num_tokens`` is the balanced
               input-token count that would produce this rank's routed
               assignments (``assignments * ep_size / top_k``); dense FFN:
               gate/up GEMM + SwiGLU + down GEMM; plus residual add

    Shared experts are not priced separately (they are counted in the
    runtime plan's memory ledger only).
    """

    _configuration_fields = (
        "hardware",
        "model",
        "fallback",
        "lookup_policy",
        "operator_backend",
        "backend_by_device",
        "cluster_config",
        "placement",
        "dtype",
        "execution_protocol",
        "require_device_scope",
    )

    def __init__(
        self,
        hardware: Any,
        model: ModelSpec,
        *,
        operator_backend: str | None = None,
        backend_by_device: Mapping[str, str | None] | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
        cache_enabled: bool = True,
        execution_protocol: str = "partial_return",
        require_device_scope: bool = False,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(
                f"AF compute provider fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}"
            )
        self.hardware = HardwareSnapshot(hardware)
        self.require_device_scope = (
            require_device_scope or execution_protocol == "vllm_afd"
        )
        self.model = snapshot(model)
        self.operator_backend = operator_backend
        self.backend_by_device = snapshot(dict(backend_by_device or {}))
        self.fallback = fallback
        self.lookup_policy = snapshot(lookup_policy)
        self.execution_protocol = execution_protocol
        initialize_cache(self, cache_enabled)
        self._resolvers: dict[str, OperatorResolver] = {}
        routing = model.moe.routing.distribution if model.moe.routing else "uniform"
        self.moe_distributions = _MOE_DISTRIBUTION_CANDIDATES.get(
            routing, ("uniform", "balanced")
        )

    # -- resolvers ----------------------------------------------------------

    def resolver_for(self, device_type: str) -> OperatorResolver:
        """The per-device resolver (loads the operator package on first use)."""
        resolver = self._resolvers.get(device_type)
        if resolver is not None:
            return resolver
        device = self.hardware.device(device_type)
        backend = self.backend_by_device.get(device_type, self.operator_backend)
        package = None
        if self.fallback != "analytical_only":
            package = self.hardware.operator_package(device.device_id, backend)
        resolver = OperatorResolver(
            device=device,
            package=package,
            fallback=self.fallback,
            lookup_policy=self.lookup_policy,
            gated=self.model.gated,
        )
        self._resolvers[device_type] = resolver
        return resolver

    @property
    def stats(self) -> OperatorStats:
        aggregate = OperatorStats()
        for resolver in self._resolvers.values():
            aggregate = aggregate.aggregate(resolver.stats)
        return aggregate

    def stats_dict(self) -> dict[str, Any]:
        stats = self.stats
        return {
            **stats.as_dict(),
            "operator_missing_shapes": stats.missing_shape_records()[:200],
            "estimate_requests": self.estimate_requests,
            "estimate_cache_hits": self.estimate_cache_hits,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "operator_table",
            "model_id": self.model.model_id,
            "execution_protocol": self.execution_protocol,
            "fallback": self.fallback,
            "devices": {
                device_type: {
                    "device_id": resolver.device.device_id,
                    "operator_backend": resolver.operator_backend,
                    "dataset_version": resolver.dataset_version,
                }
                for device_type, resolver in sorted(self._resolvers.items())
            },
        }

    # -- estimates ------------------------------------------------------------

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
                        resolver, "swiglu", "swiglu", tokens, local_width, dtype
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

    def _mla_stage(self, resolver, query, tokens, tp, dtype):
        model, h = self.model, self.model.hidden_size
        if dtype != model.dtype or (query.weight_dtype and query.weight_dtype != dtype):
            raise ConfigurationError(
                "MLA adapter currently requires identical BF16 weight/activation dtype"
            )
        if query.stage in {"a", "attention"}:
            return self._mla_attention(resolver, query, tokens, tp, dtype)
        if query.stage == "router":
            return [
                self._gemm(resolver, "router", tokens, model.moe.num_experts, h, dtype),
                self._elementwise(
                    resolver,
                    "router_topk",
                    "topk",
                    tokens,
                    model.moe.num_experts,
                    dtype,
                ),
            ]
        if query.stage in {"dense_ffn", "shared_expert"}:
            width = (
                model.intermediate_size
                if query.stage == "dense_ffn"
                else model.moe_intermediate_size * model.moe.num_shared_experts
            )
            if width % tp:
                raise ConfigurationError(f"MLA FFN width={width} must divide TP={tp}")
            width //= tp
            return [
                self._gemm(resolver, "gate_up", tokens, 2 * width, h, dtype),
                self._elementwise(resolver, "swiglu", "swiglu", tokens, width, dtype),
                self._gemm(resolver, "down", tokens, h, width, dtype),
            ]
        if query.stage in {"expert", "routed_expert"}:
            workload = self._validate_expert_workload(
                query, tokens, model.moe.num_experts
            )
            if model.moe_intermediate_size % tp:
                raise ConfigurationError("MLA expert intermediate must divide TP")
            key = dict(
                dtype=dtype,
                hidden_size=h,
                inter_size=model.moe_intermediate_size // tp,
                expert_tokens=json.dumps(
                    sorted(count for _, count in workload), separators=(",", ":")
                ),
            )
            return [
                self._component(
                    resolver, "routed_grouped_swiglu", "routed_experts", key
                )
            ]
        if query.stage == "weighted_combine":
            edges = query.input_edge_count or tokens * model.moe.num_experts_per_tok
            return [
                self._elementwise(
                    resolver,
                    "weighted_combine",
                    "weighted_combine",
                    edges,
                    int(query.local_shape["hidden_size"]),
                    dtype,
                )
            ]
        if query.stage == "residual":
            return [
                self._elementwise(
                    resolver,
                    "residual_add",
                    "residual_add",
                    tokens,
                    int(query.local_shape["hidden_size"]),
                    dtype,
                )
            ]
        if query.stage == "layout" and query.operation == "post_attention_norm":
            return [
                self._elementwise(
                    resolver, "post_attention_norm", "rmsnorm", tokens, h, dtype
                )
            ]
        raise ConfigurationError(
            f"Unsupported MLA stage={query.stage} operation={query.operation}"
        )

    def _mla_attention(self, resolver, query, tokens, tp, dtype):
        model = self.model
        model.validate_mla_tp(tp)
        dims, h = model.mla_dimensions, model.hidden_size
        heads = model.num_attention_heads // tp
        lengths = tuple(query.context_lengths)
        if not lengths or any(type(v) is not int or v <= 0 for v in lengths):
            raise ConfigurationError("MLA requires positive context lengths")
        if query.phase == "decode":
            if len(lengths) != tokens:
                raise ConfigurationError(
                    "MLA decode requires one context length per token"
                )
            query_len, mode = 1, "absorbed"
        elif query.phase == "prefill" and tokens % len(lengths) == 0:
            query_len, mode = tokens // len(lengths), "expanded"
            if query_len < 1 or any(length < query_len for length in lengths):
                raise ConfigurationError(
                    "MLA prefill requires each context length to cover the computed query"
                )
        else:
            raise ConfigurationError(
                "MLA prefill requires an equal positive query length per request"
            )
        components = []

        def gemm(name, m, n, k, groups=None):
            key = dict(dtype=dtype, m=m, n=n, k=k)
            if groups is not None:
                key["groups"] = groups
            components.append(
                self._component(
                    resolver, name, "batched_gemm" if groups else "gemm", key
                )
            )

        def elem(name, op, width):
            components.append(
                self._elementwise(resolver, name, op, tokens, width, dtype)
            )

        elem("a_rmsnorm", "rmsnorm", h)
        gemm("mla_q_down", tokens, dims.q_lora_rank, h)
        elem("mla_q_norm", "rmsnorm", dims.q_lora_rank)
        gemm("mla_q_up", tokens, heads * dims.head_dim, dims.q_lora_rank)
        gemm("mla_kv_down", tokens, dims.cache_width, h)
        elem("mla_kv_norm", "rmsnorm", dims.kv_lora_rank)
        elem("mla_rope", "rope", (heads + 1) * dims.rope_head_dim)
        if query.phase == "decode":
            gemm("mla_q_absorb", tokens, dims.kv_lora_rank, dims.nope_head_dim, heads)
        else:
            gemm(
                "mla_kv_expand",
                tokens,
                heads * (dims.nope_head_dim + dims.v_head_dim),
                dims.kv_lora_rank,
            )
        histogram = Counter(lengths)
        key = dict(
            attn_dtype=dtype,
            kv_cache_dtype=normalize_dtype(
                query.kv_cache_dtype or model.kv_cache_dtype
            ),
            phase=query.phase,
            execution_mode=mode,
            batch_size=len(lengths),
            query_len=query_len,
            context_lengths=json.dumps(
                sorted((v, c / len(lengths)) for v, c in histogram.items()),
                separators=(",", ":"),
            ),
            num_heads=heads,
            kv_lora_rank=dims.kv_lora_rank,
            rope_head_dim=dims.rope_head_dim,
            nope_head_dim=dims.nope_head_dim,
            v_head_dim=dims.v_head_dim,
        )
        components.append(
            self._component(resolver, "mla_attention_core", "mla_attention", key)
        )
        if query.phase == "decode":
            gemm("mla_value_expand", tokens, dims.v_head_dim, dims.kv_lora_rank, heads)
        gemm("output_projection", tokens, h, heads * dims.v_head_dim)
        elem("attention_residual", "residual_add", h)
        return components

    def _attention_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        tp_size: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        weight_dtype = normalize_dtype(query.weight_dtype or dtype)
        h = model.hidden_size
        heads_local = math.ceil(model.num_attention_heads / tp_size)
        kv_heads_local = local_kv_heads(model.num_key_value_heads, tp_size)
        q_local = heads_local * model.head_dim
        kv_local = kv_heads_local * model.head_dim
        components = [
            self._elementwise(resolver, "a_rmsnorm", "rmsnorm", tokens, h, dtype),
            self._gemm(
                resolver,
                "qkv_projection",
                tokens,
                q_local + 2 * kv_local,
                h,
                weight_dtype,
            ),
            self._elementwise(
                resolver, "rope", "rope", tokens, q_local + kv_local, dtype
            ),
        ]
        if self.execution_protocol == "vllm_afd":
            components[2:2] = [
                self._elementwise(
                    resolver,
                    "q_norm",
                    "rmsnorm",
                    tokens * heads_local,
                    model.head_dim,
                    dtype,
                ),
                self._elementwise(
                    resolver,
                    "k_norm",
                    "rmsnorm",
                    tokens * kv_heads_local,
                    model.head_dim,
                    dtype,
                ),
            ]
        components.extend(
            self._attention_core(
                resolver, query, tokens, heads_local, kv_heads_local, dtype
            )
        )
        components.append(
            self._gemm(resolver, "output_projection", tokens, h, q_local, weight_dtype)
        )
        components.append(
            self._elementwise(
                resolver, "a_residual_add", "residual_add", tokens, h, dtype
            )
        )
        return [component for component in components if component is not None]

    def _attention_core(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        heads_local: int,
        kv_heads_local: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        common = {
            "attn_dtype": dtype,
            "kv_cache_dtype": normalize_dtype(
                query.kv_cache_dtype or model.kv_cache_dtype
            ),
            "num_heads": heads_local,
            "num_kv_heads": kv_heads_local,
            "head_dim": model.head_dim,
            "window_size": model.sliding_window,
        }
        lengths = [max(1, int(length)) for length in query.context_lengths]
        components: list[OperatorComponent] = []
        if query.phase == "prefill":
            if lengths and sum(lengths) == tokens:
                groups: Counter = Counter(lengths)
            else:
                batch = max(1, len(lengths))
                groups = Counter({max(1, math.ceil(tokens / batch)): batch})
            for seq_len, count in sorted(groups.items()):
                key = {
                    **common,
                    "batch_size": int(count),
                    "input_seq_len": int(seq_len),
                }
                components.append(
                    self._component(
                        resolver, "attention_core", "context_attention", key
                    )
                )
            return components
        groups = Counter(lengths) if lengths else Counter({1: max(1, tokens)})
        for context_len, count in sorted(groups.items()):
            key = {**common, "batch_size": int(count), "context_len": int(context_len)}
            components.append(
                self._component(resolver, "attention_core", "generation_attention", key)
            )
        return components

    def _router_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        weight_dtype = normalize_dtype(query.weight_dtype or dtype)
        h = model.hidden_size
        components = (
            []
            if query.layer_type is not None
            else [self._elementwise(resolver, "f_rmsnorm", "rmsnorm", tokens, h, dtype)]
        )
        if model.is_moe:
            components.append(
                self._gemm(
                    resolver, "router", tokens, model.moe.num_experts, h, weight_dtype
                )
            )
            if query.layer_type is not None:
                components.append(
                    self._elementwise(
                        resolver,
                        "router_topk",
                        "topk",
                        tokens,
                        model.moe.num_experts,
                        dtype,
                    )
                )
        return [component for component in components if component is not None]

    def _expert_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        tp_size: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        weight_dtype = normalize_dtype(query.weight_dtype or dtype)
        h = model.hidden_size
        components: list[OperatorComponent | None] = []
        if model.is_moe:
            workload = self._validate_expert_workload(
                query, tokens, model.moe.num_experts
            )
            ep_size = max(1, len(query.ep_group))
            top_k = max(1, model.moe.num_experts_per_tok)
            # ``tokens`` are routed assignments already expanded by top-k. A
            # balanced EP group turns N input tokens into N * top_k / ep_size
            # assignments per rank, so this rank's load equals a balanced layer
            # fed with tokens * ep_size / top_k input tokens.
            equivalent_tokens = max(1, int(round(tokens * ep_size / top_k)))
            base_key = {
                "dtype": weight_dtype,
                "hidden_size": h,
                "inter_size": model.moe_intermediate_size,
                "top_k": top_k,
                "num_experts": model.moe.num_experts,
                "tp_size": tp_size,
                "ep_size": ep_size,
                "num_tokens": equivalent_tokens,
            }
            estimate = resolver.query_moe(base_key, self.moe_distributions)
            components.append(
                OperatorComponent(
                    name="moe_expert",
                    table="moe",
                    key={
                        **base_key,
                        "distribution": self.moe_distributions[0],
                        "routed_assignments": tokens,
                        "per_expert": [list(item) for item in workload],
                    },
                    latency_us=estimate.latency_us,
                    match_type=estimate.match_type,
                    source_id=estimate.source_id,
                    dataset_version=_estimate_dataset_version(resolver, estimate),
                )
            )
        else:
            inter_local = math.ceil(model.intermediate_size / tp_size)
            components.append(
                self._gemm(
                    resolver,
                    "dense_gate_up",
                    tokens,
                    model.ffn_projection_count * inter_local,
                    h,
                    weight_dtype,
                )
            )
            if model.gated:
                components.append(
                    self._elementwise(
                        resolver, "dense_swiglu", "swiglu", tokens, inter_local, dtype
                    )
                )
            components.append(
                self._gemm(resolver, "dense_down", tokens, h, inter_local, weight_dtype)
            )
        if query.stage == "expert":
            components.append(
                self._elementwise(
                    resolver, "f_residual_add", "residual_add", tokens, h, dtype
                )
            )
        return [component for component in components if component is not None]

    # -- operator queries -----------------------------------------------------

    def _gemm(
        self, resolver: OperatorResolver, name: str, m: int, n: int, k: int, dtype: str
    ) -> OperatorComponent | None:
        if m <= 0 or n <= 0 or k <= 0:
            return None
        return self._component(
            resolver,
            name,
            "gemm",
            {"dtype": dtype, "m": int(m), "n": int(n), "k": int(k)},
        )

    def _elementwise(
        self,
        resolver: OperatorResolver,
        name: str,
        op_name: str,
        tokens: int,
        hidden: int,
        dtype: str,
    ) -> OperatorComponent | None:
        if tokens <= 0 or hidden <= 0:
            return None
        key = {
            "op_name": op_name,
            "dtype": dtype,
            "num_tokens": int(tokens),
            "hidden_size": int(hidden),
        }
        return self._component(resolver, name, "elementwise", key)

    def _component(
        self, resolver: OperatorResolver, name: str, table: str, key: Mapping[str, Any]
    ) -> OperatorComponent:
        estimate: OperatorEstimate = resolver.query(table, key)
        covered = ()
        if self.require_device_scope and estimate.match_type != "analytical":
            metadata = (
                resolver.package.meta.extra if resolver.package is not None else {}
            )
            if metadata.get("timing_scopes", {}).get(table) != "device_service":
                raise ConfigurationError(
                    f"{table}: native/profile mode requires measured device_service scope excluding host/wait"
                )
            coverage = metadata.get("operation_coverage", {}).get(table)
            if table == "routed_experts":
                if not isinstance(coverage, (list, tuple)) or tuple(coverage) not in (
                    ("routed_experts",),
                    ("routed_experts", "weighted_combine"),
                ):
                    raise ConfigurationError(
                        "routed_experts: declare exact operation_coverage including optional weighted_combine"
                    )
                covered = tuple(coverage)
        return OperatorComponent(
            name=name,
            table=table,
            key=dict(key),
            latency_us=estimate.latency_us,
            match_type=estimate.match_type,
            source_id=estimate.source_id,
            dataset_version=_estimate_dataset_version(resolver, estimate),
            covered_operations=covered,
        )

    # -- validation ------------------------------------------------------------

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


class TopologyAFTransferProvider(ConfigurationOwner):
    """Price static AF traffic with node domains or flat alpha-beta links.

    Physical collectives expose ring subgroup phases and matching measured
    transport services. Runtime reservations and queues belong to the executor.
    """

    _configuration_fields = OperatorTableAFComputeProvider._configuration_fields + (
        "domain_config",
        "node_by_worker",
        "phase_measurements",
        "_scope_by_worker",
        "transport_profile",
    )

    def __init__(
        self,
        hardware: Any,
        cluster_config: Any,
        worker_configs: Sequence[Any],
        placement: Any | None = None,
        *,
        dtype: str = "fp16",
        operator_backend: str | None = None,
        backend_by_device: Mapping[str, str | None] | None = None,
        fallback: str = "table_first",
    ) -> None:
        self.hardware = HardwareSnapshot(hardware)
        self.cluster_config = snapshot(cluster_config)
        self.placement = snapshot(placement)
        self.dtype = normalize_dtype(dtype)
        self.operator_backend = operator_backend
        self.backend_by_device = snapshot(dict(backend_by_device or {}))
        self.fallback = fallback
        self._worker_ids: dict[str, int] = {}
        initialize_cache(self, cluster_config.af.service_estimate_cache)
        self._hardware_by_identity: dict[str, str] = {}
        for worker_id, worker in enumerate(worker_configs):
            ref = getattr(worker, "physical_worker_ref", None)
            if ref is None:
                continue
            self._worker_ids[ref.identity] = worker_id
            self._hardware_by_identity[ref.identity] = worker.hardware
        # Fail before the event loop when an AF network has no link class.
        self.link_classes = {
            link.network: self.link_class(link.network)
            for link in cluster_config.af.links
        }
        if (
            placement is None
            and cluster_config.af.transfer_topology_mode != "flat_link"
        ):
            raise ConfigurationError(
                "AF physical topology requires placement; explicitly select flat_link for legacy link pricing"
            )
        physical = physical_domain_config(cluster_config.af)
        self.domain_config = snapshot(
            cluster_config.af.communication_domains or CommunicationDomainConfig()
        )
        if physical is not None:
            validate_domain_placement(self.placement, worker_configs, physical)
        self.node_by_worker = (
            snapshot(
                {
                    worker_id: (
                        self.placement.node_id(worker_id)
                        if physical is not None
                        else self.placement.device_index(worker_id)
                        // self.domain_config.node_size
                    )
                    for worker_id in self._worker_ids.values()
                }
            )
            if placement is not None
            else {}
        )
        records = []
        resolved_backends = {}
        if fallback != "analytical_only":
            for hardware_name in sorted(set(self._hardware_by_identity.values())):
                device = self.hardware.device(hardware_name)
                backend = self.backend_by_device.get(
                    hardware_name, self.operator_backend
                )
                package = self.hardware.operator_package(device.device_id, backend)
                if package is not None:
                    backend = getattr(package.meta, "backend", backend)
                    resolved_backends[hardware_name] = backend
                    for record in package.meta.extra.get(
                        "communication_phase_services", ()
                    ):
                        explicit_scope = record.get("participant_hardware_backends")
                        count = record.get("participants")
                        if type(count) is not int or count < 1:
                            raise ConfigurationError(
                                "phase measurement participants must be a positive integer"
                            )
                        if explicit_scope is None:
                            scope = ((device.device_id, backend),) * count
                        else:
                            if (
                                not isinstance(explicit_scope, (list, tuple))
                                or len(explicit_scope) != count
                                or any(
                                    not isinstance(item, (list, tuple))
                                    or len(item) != 2
                                    for item in explicit_scope
                                )
                            ):
                                raise ConfigurationError(
                                    "phase measurement hardware/backend scope must match participants"
                                )
                            scope = tuple(
                                (self.hardware.device(item[0]).device_id, item[1])
                                for item in explicit_scope
                            )
                        if any(
                            value is not None
                            and (not isinstance(value, str) or not value)
                            for _, value in scope
                        ):
                            raise ConfigurationError(
                                "phase measurement package backend must be a nonempty string or unspecified"
                            )
                        transport = record.get("transport_backend", "unspecified")
                        if not isinstance(transport, str) or not transport:
                            raise ConfigurationError(
                                "phase measurement transport backend must be a nonempty string"
                            )
                        candidate = {
                            **record,
                            "dataset_version": package.meta.dataset_version,
                            "measurement_device": device.device_id,
                            "measurement_backend": backend,
                            "participant_hardware_backends": scope,
                            "transport_backend": transport,
                            "applicability_origin": (
                                "explicit"
                                if explicit_scope is not None
                                else "owning_package_homogeneous"
                            ),
                        }
                        # A/F device aliases may expose the same measurement package.
                        if candidate not in records:
                            records.append(candidate)
        self.phase_measurements = snapshot(tuple(records))
        self._scope_by_worker = snapshot(
            {
                worker_id: (
                    self.hardware.device(
                        self._hardware_by_identity[identity]
                    ).device_id,
                    resolved_backends.get(
                        self._hardware_by_identity[identity],
                        self.backend_by_device.get(
                            self._hardware_by_identity[identity], self.operator_backend
                        ),
                    ),
                )
                for identity, worker_id in self._worker_ids.items()
            }
        )
        self._estimate_namespace += (
            self._fingerprints(self.phase_measurements),
            self._fingerprints(self._scope_by_worker),
        )
        transport_profile = None
        profile = cluster_config.af.transport_service_profile
        if profile is not None:
            if records:
                raise ConfigurationError(
                    "transport profile conflicts with operator-package phase measurements"
                )
            if (
                cluster_config.af.transfer_topology_mode != "physical"
                or physical is None
                or len(set(self.node_by_worker.values())) != 1
            ):
                raise ConfigurationError(
                    "measured direct transport profile requires explicit same-node physical domains"
                )
            from TokenSim.af.transport_profile import TransportServiceProfile

            transport_profile = TransportServiceProfile(
                profile,
                environment=cluster_config.af.calibration_environment,
                domains=self.domain_config,
                device_indices={
                    identity: self.placement.device_index(worker)
                    for identity, worker in self._worker_ids.items()
                },
            )
            self._estimate_namespace += (
                ("transport_service", transport_profile.fingerprint),
            )
        self.transport_profile = transport_profile
        if cluster_config.af.transfer_topology_mode == "physical":
            for domain in self.cluster_config.af.execution_domains:
                for instance in domain.logical_instances:
                    by_tp = {
                        mapping.tp_rank: mapping for mapping in instance.rank_mapping
                    }
                    for collective in domain.communication_plan:
                        ranks = tuple(
                            self._worker_ids[by_tp[tp].physical_worker.identity]
                            for tp in collective.participants
                        )
                        collective_plan(
                            collective.operation,
                            ranks,
                            collective.payload_bytes or 0,
                            self.node_by_worker,
                            self.dtype,
                        )

    def link_class(self, network: str) -> LinkClass:
        link_id = self.cluster_config.networks.get(network)
        if link_id is None:
            raise ConfigurationError(
                f"AF network {network!r} is not declared in the cluster's networks"
            )
        try:
            return self.hardware.link(link_id)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"AF network {network!r} maps to {link_id!r}, which is not a link class "
                f"in the hardware catalog (data/topologies/links.yaml)"
            ) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "topology_links",
            "transport_service_profile": (
                self.transport_profile.fingerprint if self.transport_profile else None
            ),
            "dtype": self.dtype,
            "links": {
                network: link.link_id
                for network, link in sorted(self.link_classes.items())
            },
            "placement": (
                self.placement.topology.topology_id
                if self.placement is not None
                else None
            ),
            "model_version": MODEL_VERSION,
            "domain_config": dict(vars(self.domain_config)),
            "worker_nodes": dict(self.node_by_worker),
            "worker_device_indices": (
                {
                    worker: self.placement.device_index(worker)
                    for worker in self.node_by_worker
                }
                if self.placement is not None
                else {}
            ),
            "placement_source": (
                "explicit:af.communication_domains"
                if self.cluster_config.af.enabled
                and self.cluster_config.af.transfer_topology_mode == "physical"
                else "legacy:topology"
            ),
            "configuration_fingerprint": traffic_feature_fingerprint(
                (
                    {
                        "model_version": MODEL_VERSION,
                        "domains": dict(vars(self.domain_config)),
                        "placement": self.node_by_worker,
                        "mode": self.cluster_config.af.transfer_topology_mode,
                        "measurements": self.phase_measurements,
                        "hardware_backends": self._scope_by_worker,
                    },
                )
            ),
        }

    def validate_query(self, query):
        if query.operation != "all_to_all":
            return
        if type(query.payload_bytes) is not int:
            raise ConfigurationError(
                "all-to-all payload_bytes must be a non-negative integer"
            )
        validated_bytes(query.payload_bytes, query.dtype)
        endpoints = query.physical_endpoints
        if query.group_size is not None and (
            type(query.group_size) is not int or query.group_size < 1
        ):
            raise ConfigurationError("all-to-all group_size must be a positive integer")
        if (
            not endpoints
            or len(set(endpoints)) != len(endpoints)
            or any(item not in self._worker_ids for item in endpoints)
            or query.group_size not in (None, len(endpoints))
            or (
                query.participant_ranks
                and (
                    len(set(query.participant_ranks)) != len(endpoints)
                    or len(query.participant_ranks) != len(endpoints)
                )
            )
        ):
            raise ConfigurationError(
                "all-to-all needs actual unique mapped participants matching group_size"
            )
        validate_all_to_all_parameters(
            query.local_fraction,
            query.remote_fraction,
            query.load_imbalance,
            query.destination_redistribution,
        )
        if self.cluster_config.af.transfer_topology_mode == "flat_link" and (
            query.local_fraction is not None
            or query.remote_fraction is not None
            or query.load_imbalance != 1
            or query.destination_redistribution
        ):
            raise ConfigurationError(
                "flat all-to-all supports uniform defaults only; use physical for locality, imbalance or redistribution"
            )

    def communication_plan(self, query):
        """Static traffic only. Invocation identities and readiness stay outside."""
        self.validate_query(query)
        endpoints = tuple(query.physical_endpoints)
        if (
            not endpoints
            or len(set(endpoints)) != len(endpoints)
            or any(item not in self._worker_ids for item in endpoints)
        ):
            raise ConfigurationError(
                "communication participants must be nonempty, distinct and mapped"
            )
        ranks = tuple(self._worker_ids[item] for item in endpoints)
        operation = (
            "all_reduce" if query.operation == "tp_collective" else query.operation
        )
        collective = operation in {
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
        }
        if (
            collective
            and query.group_size is not None
            and query.group_size != len(ranks)
        ):
            raise ConfigurationError(
                "collective group_size differs from actual participants"
            )
        if (
            collective
            and query.participant_ranks
            and len(query.participant_ranks) != len(set(query.participant_ranks))
        ):
            raise ConfigurationError("duplicate collective participant ranks")
        if (
            collective
            and query.participant_ranks
            and len(query.participant_ranks) != len(ranks)
        ):
            raise ConfigurationError("collective participant/endpoints mismatch")
        if not collective and len(ranks) != 2:
            raise ConfigurationError(
                "endpoint communication requires exactly two ranks"
            )
        payload = validated_bytes(query.payload_bytes)
        # Byte-oriented metadata need not align to dtype; measured service still does.
        key = (
            MODEL_VERSION,
            operation,
            payload,
            query.dtype,
            endpoints,
            query.participant_ranks,
            self.node_by_worker,
            self.placement,
            query.logical_input_bytes_per_rank,
            query.logical_output_bytes_per_rank,
            traffic_feature_fingerprint(query.routing_edges),
            traffic_feature_fingerprint(query.tensor_slices),
            query.topology,
        )
        transport_backend = getattr(query, "transport_backend", "unspecified")
        if not isinstance(transport_backend, str) or not transport_backend:
            raise ConfigurationError("transport backend must be a nonempty string")
        key += (
            query.local_fraction,
            query.remote_fraction,
            query.load_imbalance,
            query.destination_redistribution,
            transport_backend,
        )

        def build():
            if operation == "all_to_all":
                plan = all_to_all_plan(
                    ranks,
                    payload,
                    self.node_by_worker,
                    local_fraction=query.local_fraction,
                    remote_fraction=query.remote_fraction,
                    load_imbalance=query.load_imbalance,
                    destination_redistribution=query.destination_redistribution,
                    dtype=query.dtype,
                )
            elif collective:
                plan = collective_plan(
                    operation, ranks, payload, self.node_by_worker, query.dtype
                )
            else:
                plan = point_to_point_plan(
                    ranks[0],
                    ranks[1],
                    payload,
                    self.node_by_worker,
                    wire_dtype=query.dtype,
                )
            return replace(
                plan,
                phases=tuple(
                    replace(phase, transport_backend=transport_backend)
                    for phase in plan.phases
                ),
            )

        return cached_plan(self, key, build)

    @cached_phase_estimate
    def phase_estimate(self, phase):
        if self.transport_profile is not None:
            if phase.operation != "send_recv" or len(phase.participants) != 2:
                raise ConfigurationError(
                    "direct transport profile cannot price collective phases"
                )
            source, target = phase.participants
            duration = self.transport_profile.estimate_us(
                source=self.placement.device_index(source),
                target=self.placement.device_index(target),
                backend=phase.transport_backend,
                dtype=phase.dtype,
                size=phase.wire_bytes,
            )
            exact = any(
                row["source_device"] == self.placement.device_index(source)
                and row["target_device"] == self.placement.device_index(target)
                and row["backend"] == phase.transport_backend
                and row["dtype"] == phase.dtype
                and any(point["bytes"] == phase.wire_bytes for point in row["points"])
                for row in self.transport_profile.document["records"]
            )
            return AFServiceEstimate(
                duration * 1e-6,
                self.transport_profile.document["profile_id"],
                phase.logical_input_bytes,
                self.transport_profile.fingerprint,
                "exact" if exact else "interpolated",
                {
                    "timing_scope": "transport_service",
                    "transport_backend": phase.transport_backend,
                    "model_version": MODEL_VERSION,
                    "phase_id": phase.phase_id,
                    "resource_identities": phase.resource_ids,
                    "resource_wire_bytes": phase.resource_wire_bytes,
                    "contains_cpu_submit": False,
                    "contains_external_wait": False,
                    "measurement_boundary": self.transport_profile.document["evidence"][
                        "measurement_boundary"
                    ],
                    "interpolation": "linear_within_training_range",
                },
            )
        algorithm = (
            "direct" if phase.operation in {"send_recv", "all_to_all"} else "ring"
        )
        nodes = {
            node: index
            for index, node in enumerate(
                sorted({self.node_by_worker[rank] for rank in phase.participants})
            )
        }
        signature = {
            "model_version": MODEL_VERSION,
            "algorithm": algorithm,
            "participant_hardware_backends": tuple(
                self._scope_by_worker[rank] for rank in phase.participants
            ),
            "transport_backend": phase.transport_backend,
            "phase_kind": phase.phase_id.split(":")[0],
            "operation": phase.operation,
            "dtype": phase.dtype,
            "participants": len(phase.participants),
            "nodes": len({self.node_by_worker[rank] for rank in phase.participants}),
            "logical_input_bytes": phase.logical_input_bytes,
            "wire_bytes": phase.wire_bytes,
            "wire_schedule": tuple(
                tuple(
                    (
                        nodes[message.source_node],
                        nodes[message.target_node],
                        message.wire_bytes,
                    )
                    for message in messages
                )
                for messages in phase.rounds
            ),
            "domain_parameters": tuple(
                (key, getattr(self.domain_config, key))
                for key in (
                    "local_bandwidth_bytes_per_s",
                    "inter_tx_bandwidth_bytes_per_s",
                    "inter_rx_bandwidth_bytes_per_s",
                    "local_latency_us",
                    "inter_latency_us",
                )
            ),
        }
        matches = [
            record
            for record in self.phase_measurements
            if phase.wire_bytes
            and all(record.get(key) == value for key, value in signature.items())
        ]
        if len(matches) > 1:
            raise ConfigurationError(
                "conflicting matching communication phase measurements"
            )
        if matches:
            record = matches[0]
            if record.get("timing_scope") != "transport_service":
                raise ConfigurationError(
                    "communication phase measurement must declare transport_service timing scope"
                )
            duration = record.get("latency_us")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ConfigurationError("invalid measured communication phase latency")
            if not isinstance(record.get("source_id"), str) or not record["source_id"]:
                raise ConfigurationError(
                    "measured communication phase requires a nonempty source_id"
                )
            return AFServiceEstimate(
                duration * 1e-6,
                record["source_id"],
                phase.logical_input_bytes,
                record["dataset_version"],
                "exact",
                {
                    **signature,
                    "timing_scope": "transport_service",
                    "phase_id": phase.phase_id,
                    "measurement_device": record["measurement_device"],
                    "measurement_backend": record["measurement_backend"],
                    "applicability_origin": record["applicability_origin"],
                    "resource_wire_bytes": phase.resource_wire_bytes,
                    "resource_identities": phase.resource_ids,
                },
            )
        if self.fallback == "table_only" and phase.wire_bytes:
            raise ConfigurationError(
                "measured-only communication needs matching algorithm and phase transport_service records; whole collective timing cannot be split"
            )
        return AFServiceEstimate(
            phase.service(self.domain_config),
            f"analytical:{MODEL_VERSION}:{phase.phase_id}",
            phase.logical_input_bytes,
            MODEL_VERSION,
            "analytical",
            {
                "timing_scope": "transport_service",
                "model_version": MODEL_VERSION,
                "participant_hardware_backends": signature[
                    "participant_hardware_backends"
                ],
                "transport_backend": phase.transport_backend,
                "phase_id": phase.phase_id,
                "algorithm": algorithm,
                "participants": phase.participants,
                "fan": phase.fan,
                "logical_bytes": phase.logical_input_bytes,
                "wire_bytes": phase.wire_bytes,
                "resource_wire_bytes": phase.resource_wire_bytes,
                "resource_identities": phase.resource_ids,
                "rounds": len(phase.rounds),
                "relative_timetable": phase.relative_timetable(self.domain_config),
                "deprecated_fields_ignored": (
                    "half_bandwidth_bytes",
                    "collective_efficiency",
                ),
            },
        )

    @cached_estimate
    def estimate(self, query: AFTransferQuery) -> AFServiceEstimate:
        if query.operation not in TRANSFER_OPERATIONS:
            raise ValueError(
                f"unsupported AF communication operation={query.operation!r}"
            )
        if query.payload_bytes is None:
            raise ValueError(
                f"operation {query.operation!r} requires explicit payload_bytes"
            )
        if type(query.payload_bytes) is not int or query.payload_bytes < 0:
            raise ValueError("transfer payload_bytes must be a non-negative integer")
        if query.operation in {
            "tp_collective",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
        }:
            endpoints = query.physical_endpoints
            if (
                not endpoints
                or len(set(endpoints)) != len(endpoints)
                or any(item not in self._worker_ids for item in endpoints)
            ):
                raise ConfigurationError(
                    "collective participants must be nonempty, distinct and mapped"
                )
            if query.participant_ranks and (
                len(query.participant_ranks) != len(endpoints)
                or len(set(query.participant_ranks)) != len(endpoints)
            ):
                raise ConfigurationError(
                    "collective participant ranks must be unique and match endpoints"
                )
        if self.cluster_config.af.transfer_topology_mode == "physical":
            self.link_class(query.network)
            plan = self.communication_plan(query)
            services = {
                phase.phase_id: self.phase_estimate(phase) for phase in plan.phases
            }
            match_type = _worst_match(item.match_type for item in services.values())
            resource_wire = Counter()
            for phase in plan.phases:
                for resource, size in phase.resource_wire_bytes:
                    resource_wire[resource] += size
            return AFServiceEstimate(
                plan.service(
                    self.domain_config,
                    lambda phase: services[phase.phase_id].service_duration,
                ),
                _compose_source_id(item.source_id for item in services.values())
                or f"analytical:{MODEL_VERSION}",
                query.payload_bytes,
                _compose_dataset_version(
                    item.dataset_version for item in services.values()
                )
                or MODEL_VERSION,
                match_type,
                {
                    "timing_scope": "transport_service",
                    "model_version": MODEL_VERSION,
                    "operation": query.operation,
                    "logical_bytes": query.payload_bytes,
                    "output_bytes_per_rank": plan.output_bytes_per_rank,
                    "expected_output_bytes": plan.expected_output_bytes,
                    "wire_bytes": plan.wire_bytes,
                    "routing_provenance": plan.provenance,
                    "resource_wire_bytes": tuple(sorted(resource_wire.items())),
                    "phase_ids": tuple(phase.phase_id for phase in plan.phases),
                },
            )
        if query.operation in {
            "tp_collective",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
        }:
            return self._collective(query)
        if query.operation == "all_to_all":
            count = len(query.physical_endpoints)
            if (
                count < 1
                or len(set(query.physical_endpoints)) != count
                or query.group_size not in (None, count)
            ):
                raise ConfigurationError(
                    "flat all-to-all needs actual unique participants matching group_size"
                )
            link = self.link_class(query.network)
            wire_bytes = query.payload_bytes * (count - 1) / count
            return self._collective_estimate(
                link.analytical_transfer_us(wire_bytes),
                f"link:{link.link_id}:{link.source_id}",
                f"links:{link.link_id}",
                "analytical",
                query,
                {
                    "algorithm": "direct",
                    "wire_bytes": wire_bytes,
                    "wire_bytes_scope": "per_rank",
                    "group_size": count,
                    "model_version": MODEL_VERSION,
                    "routing_provenance": "uniform_fallback",
                },
            )
        link = self.link_class(query.network)
        payload = query.payload_bytes
        service_us = link.analytical_transfer_us(payload)
        source_id = f"link:{link.link_id}:{link.source_id}"
        dataset_version = f"links:{link.link_id}"
        return AFServiceEstimate(
            service_duration=service_us / 1e6,
            source_id=source_id,
            payload_bytes=payload,
            dataset_version=dataset_version,
            match_type="analytical",
            details={
                "provider": "topology_links",
                "model_version": MODEL_VERSION,
                "timing_scope": "transport_service",
                "transport_backend": query.transport_backend,
                "backend_calibrated": False,
                "link": (link.link_id if link is not None else None),
                "fixed_latency_us": link.latency_us if link is not None else None,
                "bandwidth_bytes_per_s": (
                    link.bandwidth_per_direction_bytes_per_s
                    if link is not None
                    else None
                ),
                "efficiency": 1.0,
                "deprecated_fields_ignored": (
                    "half_bandwidth_bytes",
                    "collective_efficiency",
                ),
                "payload_bytes": payload,
                "operation": query.operation,
                "direction": query.direction,
                "logical_participants": query.logical_participants,
                "participant_ranks": query.participant_ranks,
                "physical_endpoints": query.physical_endpoints,
                "resource_identities": query.resource_identities,
                "topology": dict(query.topology),
                "total_latency_us": service_us,
            },
        )

    def _collective(self, query: AFTransferQuery) -> AFServiceEstimate:
        payload = query.payload_bytes or 0
        operation = (
            "all_reduce" if query.operation == "tp_collective" else query.operation
        )
        group_size = (
            query.group_size
            if query.group_size is not None
            else len(query.physical_endpoints)
        )
        if (
            not query.physical_endpoints
            or len(set(query.physical_endpoints)) != len(query.physical_endpoints)
            or group_size != len(query.physical_endpoints)
        ):
            raise ConfigurationError(
                "collective needs actual unique participants matching group_size"
            )
        volume = ring_volume(operation, group_size, payload, query.dtype)
        if payload == 0:
            return self._collective_estimate(
                0.0, "collective:empty", "collective", "exact", query, {}
            )
        # No placement: a ring all-reduce over the AF link class.
        link = self.link_class(query.network)
        ring_bytes = volume.send_bytes_per_rank
        service_us = (
            (
                volume.rounds * link.latency_us
                + volume.send_bytes_per_rank
                / link.bandwidth_per_direction_bytes_per_s
                * 1e6
            )
            if payload
            else 0.0
        )
        return self._collective_estimate(
            service_us,
            f"link:{link.link_id}:{link.source_id}",
            f"links:{link.link_id}",
            "analytical",
            query,
            {
                "algorithm": "ring",
                "operation": operation,
                "logical_bytes": payload,
                "input_bytes_per_rank": volume.input_bytes,
                "output_bytes_per_rank": volume.output_bytes,
                "rounds": volume.rounds,
                "wire_bytes": ring_bytes,
                "wire_bytes_scope": "per_rank",
                "group_size": group_size,
                "link": link.link_id,
            },
        )

    def _collective_estimate(
        self,
        latency_us: float,
        source_id: str,
        dataset_version: str,
        match_type: str,
        query: AFTransferQuery,
        extra: Mapping[str, Any],
    ) -> AFServiceEstimate:
        return AFServiceEstimate(
            service_duration=latency_us / 1e6,
            source_id=source_id,
            payload_bytes=query.payload_bytes,
            dataset_version=dataset_version,
            match_type=match_type,
            details={
                "provider": "collective_model",
                "model_version": MODEL_VERSION,
                "timing_scope": "transport_service",
                "operation": query.operation,
                "direction": query.direction,
                "payload_bytes": query.payload_bytes,
                "logical_participants": query.logical_participants,
                "participant_ranks": query.participant_ranks,
                "physical_endpoints": query.physical_endpoints,
                "resource_identities": query.resource_identities,
                "total_latency_us": latency_us,
                **dict(extra),
            },
        )


def build_operator_table_af_providers(
    *,
    hardware: Any,
    model: ModelSpec,
    cluster_config: Any,
    worker_configs: Sequence[Any],
    placement: Any | None,
    dtype: str = "fp16",
    operator_backend: str | None = None,
    fallback: str = "table_first",
    latency_backend_type: str = "operator_table",
) -> tuple[OperatorTableAFComputeProvider, TopologyAFTransferProvider]:
    """Build the pair of providers the engine uses for ``af.latency_provider = operator_table``.

    Every AF device is resolved eagerly so that a missing device, an unknown
    operator backend or an unmapped network fails before the event loop.
    """
    if latency_backend_type == "analytical":
        fallback = "analytical_only"
    backend_by_device = {
        worker.hardware: worker.operator_backend
        for worker in worker_configs
        if getattr(worker, "operator_backend", None)
    }
    compute = OperatorTableAFComputeProvider(
        hardware,
        model,
        operator_backend=operator_backend,
        backend_by_device=backend_by_device,
        fallback=fallback,
        cache_enabled=cluster_config.af.service_estimate_cache,
        execution_protocol=cluster_config.af.execution_protocol,
        require_device_scope=cluster_config.af.engine_overhead_profile is not None,
    )
    transfer = TopologyAFTransferProvider(
        hardware,
        cluster_config,
        worker_configs,
        placement,
        dtype=dtype,
        operator_backend=operator_backend,
        backend_by_device=backend_by_device,
        fallback=fallback,
    )
    for hardware_name in sorted({worker.hardware for worker in worker_configs}):
        compute.resolver_for(hardware_name)
    if cluster_config.af.engine_overhead_profile is not None:
        from TokenSim.af.calibration import content_fingerprint

        fingerprint = content_fingerprint(cluster_config.af.engine_overhead_profile)
        for provider in (compute, transfer):
            provider._estimate_namespace += (fingerprint,)
    if cluster_config.af.operation_service_profile is not None:
        from TokenSim.af.calibration import content_fingerprint

        fingerprint = content_fingerprint(cluster_config.af.operation_service_profile)
        for provider in (compute, transfer):
            provider._estimate_namespace += (("operation_service", fingerprint),)
    return compute, transfer


__all__ = [
    "OperatorComponent",
    "OperatorTableAFComputeProvider",
    "TRANSFER_OPERATIONS",
    "TopologyAFTransferProvider",
    "build_operator_table_af_providers",
]
