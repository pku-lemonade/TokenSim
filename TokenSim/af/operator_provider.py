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
* :class:`TopologyAFTransferProvider` prices cross-domain payloads with the
  link class named by the cluster's ``networks`` mapping and TP collectives
  with the hierarchical :class:`CollectiveModel`.

Service times are pure kernel/link times; queueing on shared workers and
links is produced by the executor's SimPy resources.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.af.latency import (
    AF_MATCH_TYPES,
    AFComputeQuery,
    AFServiceEstimate,
    AFTransferQuery,
)
from TokenSim.comm.collectives import CollectiveModel, CollectiveQuery
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype
from TokenSim.hardware.links import LinkClass
from TokenSim.latency.operator_table import (
    FALLBACK_POLICIES,
    _MOE_DISTRIBUTION_CANDIDATES,
    OperatorEstimate,
    OperatorResolver,
    OperatorStats,
    collective_model_for,
)
from TokenSim.operator_data.lookup import LookupPolicy

TRANSFER_OPERATIONS = ("af_a_to_f", "af_f_to_a", "ep_dispatch", "ep_combine", "tp_collective")
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


@dataclass(frozen=True)
class OperatorComponent:
    """One operator of an AF stage and how it was priced."""

    name: str
    table: str
    key: Mapping[str, Any]
    latency_us: float
    match_type: str
    source_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "key": dict(self.key),
            "latency_us": self.latency_us,
            "match_type": self.match_type,
            "source_id": self.source_id,
        }


class OperatorTableAFComputeProvider:
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

    def __init__(
        self,
        hardware: Any,
        model: ModelSpec,
        *,
        operator_backend: str | None = None,
        backend_by_device: Mapping[str, str | None] | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(
                f"AF compute provider fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}"
            )
        self.hardware = hardware
        self.model = model
        self.operator_backend = operator_backend
        self.backend_by_device = dict(backend_by_device or {})
        self.fallback = fallback
        self.lookup_policy = lookup_policy
        self._resolvers: dict[str, OperatorResolver] = {}
        routing = model.moe.routing.distribution if model.moe.routing else "uniform"
        self.moe_distributions = _MOE_DISTRIBUTION_CANDIDATES.get(routing, ("uniform", "balanced"))

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
        return {**stats.as_dict(), "operator_missing_shapes": stats.missing_shape_records()[:200]}

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "operator_table",
            "model_id": self.model.model_id,
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

    def estimate(self, query: AFComputeQuery) -> AFServiceEstimate:
        tokens, tp_size = self._validate_local_work(query)
        resolver = self.resolver_for(query.device_type)
        dtype = normalize_dtype(query.dtype)
        if query.stage == "a":
            components = self._attention_components(resolver, query, tokens, tp_size, dtype)
        elif query.stage == "router":
            components = self._router_components(resolver, query, tokens, dtype)
        elif query.stage == "expert":
            components = self._expert_components(resolver, query, tokens, tp_size, dtype)
        else:
            raise ValueError(f"unsupported AF compute stage={query.stage!r}")
        latency_us = sum(component.latency_us for component in components)
        match_type = _worst_match(component.match_type for component in components)
        return AFServiceEstimate(
            service_duration=latency_us / 1e6,
            source_id=_compose_source_id(component.source_id for component in components),
            dataset_version=resolver.dataset_version or f"analytical:{resolver.device.device_id}",
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
            },
        )

    def _attention_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        tp_size: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        h = model.hidden_size
        heads_local = math.ceil(model.num_attention_heads / tp_size)
        kv_heads_local = local_kv_heads(model.num_key_value_heads, tp_size)
        q_local = heads_local * model.head_dim
        kv_local = kv_heads_local * model.head_dim
        components = [
            self._elementwise(resolver, "a_rmsnorm", "rmsnorm", tokens, h, dtype),
            self._gemm(resolver, "qkv_projection", tokens, q_local + 2 * kv_local, h, dtype),
            self._elementwise(resolver, "rope", "rope", tokens, q_local + kv_local, dtype),
        ]
        components.extend(
            self._attention_core(resolver, query, tokens, heads_local, kv_heads_local, dtype)
        )
        components.append(self._gemm(resolver, "output_projection", tokens, h, q_local, dtype))
        components.append(
            self._elementwise(resolver, "a_residual_add", "residual_add", tokens, h, dtype)
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
            "kv_cache_dtype": model.kv_cache_dtype,
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
                key = {**common, "batch_size": int(count), "input_seq_len": int(seq_len)}
                components.append(self._component(resolver, "attention_core", "context_attention", key))
            return components
        groups = Counter(lengths) if lengths else Counter({1: max(1, tokens)})
        for context_len, count in sorted(groups.items()):
            key = {**common, "batch_size": int(count), "context_len": int(context_len)}
            components.append(self._component(resolver, "attention_core", "generation_attention", key))
        return components

    def _router_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        h = model.hidden_size
        components = [self._elementwise(resolver, "f_rmsnorm", "rmsnorm", tokens, h, dtype)]
        if model.is_moe:
            components.append(self._gemm(resolver, "router", tokens, model.moe.num_experts, h, dtype))
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
        h = model.hidden_size
        components: list[OperatorComponent | None] = []
        if model.is_moe:
            workload = self._validate_expert_workload(query, tokens, model.moe.num_experts)
            ep_size = max(1, len(query.ep_group))
            top_k = max(1, model.moe.num_experts_per_tok)
            # ``tokens`` are routed assignments already expanded by top-k. A
            # balanced EP group turns N input tokens into N * top_k / ep_size
            # assignments per rank, so this rank's load equals a balanced layer
            # fed with tokens * ep_size / top_k input tokens.
            equivalent_tokens = max(1, int(round(tokens * ep_size / top_k)))
            base_key = {
                "dtype": dtype,
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
                )
            )
        else:
            inter_local = math.ceil(model.intermediate_size / tp_size)
            components.append(
                self._gemm(resolver, "dense_gate_up", tokens, model.ffn_projection_count * inter_local, h, dtype)
            )
            if model.gated:
                components.append(
                    self._elementwise(resolver, "dense_swiglu", "swiglu", tokens, inter_local, dtype)
                )
            components.append(self._gemm(resolver, "dense_down", tokens, h, inter_local, dtype))
        components.append(
            self._elementwise(resolver, "f_residual_add", "residual_add", tokens, h, dtype)
        )
        return [component for component in components if component is not None]

    # -- operator queries -----------------------------------------------------

    def _gemm(
        self, resolver: OperatorResolver, name: str, m: int, n: int, k: int, dtype: str
    ) -> OperatorComponent | None:
        if m <= 0 or n <= 0 or k <= 0:
            return None
        return self._component(resolver, name, "gemm", {"dtype": dtype, "m": int(m), "n": int(n), "k": int(k)})

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
        key = {"op_name": op_name, "dtype": dtype, "num_tokens": int(tokens), "hidden_size": int(hidden)}
        return self._component(resolver, name, "elementwise", key)

    @staticmethod
    def _component(
        resolver: OperatorResolver, name: str, table: str, key: Mapping[str, Any]
    ) -> OperatorComponent:
        estimate: OperatorEstimate = resolver.query(table, key)
        return OperatorComponent(
            name=name,
            table=table,
            key=dict(key),
            latency_us=estimate.latency_us,
            match_type=estimate.match_type,
            source_id=estimate.source_id,
        )

    # -- validation ------------------------------------------------------------

    def _validate_local_work(self, query: AFComputeQuery) -> tuple[int, int]:
        context = (
            f"operation={query.operation!r}, stage={query.stage!r}, "
            f"domain={query.source_domain!r}, dp_rank={query.dp_rank}, "
            f"tp_rank={query.tp_rank}, ep_rank={query.ep_rank}"
        )
        if query.local_token_count is None or query.local_token_count < 1:
            raise ValueError(f"AF compute query missing positive local_token_count; {context}")
        if query.token_count != query.local_token_count:
            raise ValueError(
                f"AF compute query token_count={query.token_count} does not match "
                f"local_token_count={query.local_token_count}; {context}"
            )
        shape = query.local_shape
        required = ("tokens", "hidden_size", "global_hidden_size", "tp_size")
        missing = sorted(set(required) - set(shape))
        if missing:
            raise ValueError(f"AF compute query missing local_shape fields {missing}; {context}")
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
            self.model.hidden_size
            if query.stage == "router"
            else math.ceil(self.model.hidden_size / values["tp_size"])
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
            raise ValueError("expert_workload requires non-negative ids and positive counts")
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


class TopologyAFTransferProvider:
    """Price AF transfers from the catalog's link classes and collective model.

    ``af_a_to_f`` / ``af_f_to_a`` (and the legacy ``ep_dispatch`` /
    ``ep_combine``) use the link class the cluster's ``networks`` mapping
    assigns to the AF link's network: ``latency_us + payload / (bandwidth *
    efficiency(payload))``. ``tp_collective`` runs an all-reduce through the
    hierarchical collective model over the participants' topology placement
    (measured NCCL curves when the device package has them); without a
    placement it falls back to a ring on the same link class.
    """

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
        self.hardware = hardware
        self.cluster_config = cluster_config
        self.placement = placement
        self.dtype = normalize_dtype(dtype)
        self.operator_backend = operator_backend
        self.backend_by_device = dict(backend_by_device or {})
        self.fallback = fallback
        self._worker_ids: dict[str, int] = {}
        self._hardware_by_identity: dict[str, str] = {}
        for worker_id, worker in enumerate(worker_configs):
            ref = getattr(worker, "physical_worker_ref", None)
            if ref is None:
                continue
            self._worker_ids[ref.identity] = worker_id
            self._hardware_by_identity[ref.identity] = worker.hardware
        self._collective_models: dict[str, CollectiveModel] = {}
        # Fail before the event loop when an AF network has no link class.
        self.link_classes = {link.network: self.link_class(link.network) for link in cluster_config.af.links}

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
            "dtype": self.dtype,
            "links": {network: link.link_id for network, link in sorted(self.link_classes.items())},
            "placement": self.placement.topology.topology_id if self.placement is not None else None,
        }

    def estimate(self, query: AFTransferQuery) -> AFServiceEstimate:
        if query.operation not in TRANSFER_OPERATIONS:
            raise ValueError(f"unsupported AF communication operation={query.operation!r}")
        if query.payload_bytes is None:
            raise ValueError(f"operation {query.operation!r} requires explicit payload_bytes")
        if type(query.payload_bytes) is not int or query.payload_bytes < 0:
            raise ValueError("transfer payload_bytes must be a non-negative integer")
        if query.operation == "tp_collective":
            return self._collective(query)
        link = self.link_class(query.network)
        payload = query.payload_bytes
        service_us = link.transfer_us(payload) if payload > 0 else 0.0
        return AFServiceEstimate(
            service_duration=service_us / 1e6,
            source_id=f"link:{link.link_id}:{link.source_id}",
            payload_bytes=payload,
            dataset_version=f"links:{link.link_id}",
            match_type="analytical",
            details={
                "provider": "topology_links",
                "link": link.link_id,
                "fixed_latency_us": link.latency_us,
                "bandwidth_bytes_per_s": link.bandwidth_per_direction_bytes_per_s,
                "efficiency": link.efficiency(payload),
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
        worker_ids = [self._worker_ids[item] for item in query.physical_endpoints if item in self._worker_ids]
        group_size = max(2, query.group_size or len(worker_ids) or 2)
        if payload == 0:
            return self._collective_estimate(0.0, "collective:empty", "collective", "exact", query, {})
        if self.placement is not None and len(worker_ids) >= 2 and len(set(worker_ids)) == len(worker_ids):
            model = self._collective_model_for(query.physical_endpoints[0])
            layout = self.placement.layout(worker_ids)
            estimate = model.estimate(CollectiveQuery("all_reduce", float(payload), layout, self.dtype))
            return self._collective_estimate(
                estimate.latency_us,
                estimate.source_id,
                f"collective:{self.placement.topology.topology_id}",
                _COLLECTIVE_MATCH.get(estimate.match_type, estimate.match_type),
                query,
                {
                    "algorithm_levels": [level.level for level in estimate.levels],
                    "group_size": estimate.group_size,
                    "launch_us": estimate.launch_us,
                },
            )
        # No placement: a ring all-reduce over the AF link class.
        link = self.link_class(query.network)
        ring_bytes = 2.0 * (group_size - 1) / group_size * payload
        service_us = link.transfer_us(ring_bytes)
        return self._collective_estimate(
            service_us,
            f"link:{link.link_id}:{link.source_id}",
            f"links:{link.link_id}",
            "analytical",
            query,
            {"algorithm": "ring", "group_size": group_size, "link": link.link_id},
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

    def _collective_model_for(self, physical_identity: str) -> CollectiveModel:
        hardware_name = self._hardware_by_identity[physical_identity]
        device = self.hardware.device(hardware_name)
        model = self._collective_models.get(device.device_id)
        if model is None:
            package = None
            if self.fallback != "analytical_only":
                backend = self.backend_by_device.get(hardware_name, self.operator_backend)
                package = self.hardware.operator_package(device.device_id, backend)
            model = collective_model_for(device, self.placement, self.hardware.links, package)
            self._collective_models[device.device_id] = model
        return model


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
    return compute, transfer


__all__ = [
    "OperatorComponent",
    "OperatorTableAFComputeProvider",
    "TRANSFER_OPERATIONS",
    "TopologyAFTransferProvider",
    "build_operator_table_af_providers",
]
