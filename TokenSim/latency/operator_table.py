"""Latency backend that composes per-operator table lookups into a step time.

For every scheduler step the backend decomposes the transformer layers owned
by this rank into GEMM, attention-core, MoE, elementwise and collective
operators, resolves each operator against the device's operator tables (exact
match, controlled interpolation, bounded extrapolation) and falls back to the
device family's analytical model when a table has no usable data. Every
estimate records where it came from, so results can report how much of the
simulated time rests on measurements versus formulas.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from TokenSim.comm.collectives import CollectiveModel
from TokenSim.config.cache_config import local_kv_heads, stage_layer_count
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec, dtype_bytes
from TokenSim.latency.base import (
    LatencyBackend,
    is_context_build,
    request_context_tokens,
    request_kv_len,
)
from TokenSim.llm.llm_request import Request
from TokenSim.moe.placement import ExpertPlacement
from TokenSim.moe.routing import ExpertRouting
from TokenSim.moe.stats import MoEStats
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.lookup import LookupPolicy, MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.workload import activation_bytes_for, context_attention_work
from TokenSim.parallel import ParallelCommunicator

logger = logging.getLogger(__name__)

FALLBACK_POLICIES = ("table_first", "table_only", "analytical_only")
COMPONENTS = ("gemm", "attention", "moe", "elementwise", "lm_head", "comm", "overhead")

_MOE_DISTRIBUTION_CANDIDATES = {
    "uniform": ("uniform", "balanced", "power_law_1.01"),
    "balanced": ("balanced", "uniform", "power_law_1.01"),
    "skew": ("power_law_1.2", "skew", "uniform", "balanced"),
    "hot": ("power_law_1.2", "hot", "uniform", "balanced"),
    "burst": ("power_law_1.2", "burst", "uniform", "balanced"),
}


@dataclass
class OperatorStats:
    """Aggregated provenance and time breakdown of every operator estimate."""

    match_type_counts: Counter = field(default_factory=Counter)
    table_match_counts: dict[str, Counter] = field(default_factory=dict)
    component_seconds: Counter = field(default_factory=Counter)
    missing_shapes: dict[tuple, dict[str, Any]] = field(default_factory=dict)
    step_count: int = 0
    _missing_limit: int = 4096

    def record(self, table: str, match_type: str) -> None:
        self.match_type_counts[match_type] += 1
        self.table_match_counts.setdefault(table, Counter())[match_type] += 1

    def record_missing(self, table: str, key: Mapping[str, Any], reason: str) -> None:
        identity = (table, tuple(sorted(key.items())))
        if identity in self.missing_shapes or len(self.missing_shapes) >= self._missing_limit:
            return
        self.missing_shapes[identity] = {"table": table, "key": dict(key), "reason": reason}

    def record_component(self, component: str, seconds: float) -> None:
        self.component_seconds[component] += seconds

    def aggregate(self, other: "OperatorStats") -> "OperatorStats":
        result = OperatorStats()
        result.match_type_counts = self.match_type_counts + other.match_type_counts
        for table, counter in list(self.table_match_counts.items()) + list(other.table_match_counts.items()):
            result.table_match_counts[table] = result.table_match_counts.get(table, Counter()) + counter
        result.component_seconds = self.component_seconds + other.component_seconds
        result.missing_shapes = {**self.missing_shapes, **other.missing_shapes}
        result.step_count = self.step_count + other.step_count
        return result

    def as_dict(self) -> dict[str, Any]:
        total = sum(self.match_type_counts.values())
        return {
            "operator_query_count": total,
            "operator_match_type_counts": dict(self.match_type_counts),
            "operator_table_match_counts": {
                table: dict(counter) for table, counter in sorted(self.table_match_counts.items())
            },
            "operator_component_seconds": {k: self.component_seconds.get(k, 0.0) for k in COMPONENTS},
            "operator_missing_shape_count": len(self.missing_shapes),
            "operator_step_count": self.step_count,
        }

    def missing_shape_records(self) -> list[dict[str, Any]]:
        return list(self.missing_shapes.values())


@dataclass(frozen=True)
class OperatorEstimate:
    latency_us: float
    match_type: str
    source_id: str


class OperatorTableLatencyBackend(LatencyBackend):
    _CACHE_LIMIT = 262144

    def __init__(
        self,
        *,
        device: DeviceSpec,
        model: ModelSpec,
        parallel_config: ParallelConfig | None = None,
        rank_info: ParallelRankInfo | None = None,
        communicator: ParallelCommunicator | None = None,
        expert_placement: ExpertPlacement | None = None,
        package: OperatorDataPackage | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
        decode_context_bucket: int = 128,
        random_seed: int = 0,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(f"fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}")
        if fallback != "analytical_only" and package is None and fallback == "table_only":
            raise ConfigurationError(
                f"latency backend 'table_only' requires operator data for device {device.device_id!r}"
            )
        self.device = device
        self.model = model
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.rank_info = rank_info or ParallelRankInfo()
        self.communicator = communicator
        if expert_placement is None and model.is_moe:
            from TokenSim.moe.placement import build_expert_placement

            expert_placement = build_expert_placement(
                model.moe, self.parallel_config, total_layers=model.num_layers
            )
        self.expert_placement = expert_placement
        self.package = package if fallback != "analytical_only" else None
        self.fallback = fallback
        self.analytical = analytical_model_for(device.family)
        # Out-of-range table queries keep the boundary row's measured efficiency
        # and follow the analytical model's growth (see LookupPolicy).
        self.lookup = (
            OperatorLookup(self.package, lookup_policy, analytical_scaler=self._analytical_reference_us)
            if self.package is not None
            else None
        )
        self.decode_context_bucket = max(1, int(decode_context_bucket))
        self.stats = OperatorStats()
        self.moe_stats = MoEStats(
            effective_moe_config=model.moe.to_dict(),
            expert_placement=expert_placement.to_dict() if expert_placement else None,
        )
        self.expert_routing = ExpertRouting(model.moe, seed=random_seed)
        self._expert_to_rank: dict[int, int] | None = None
        self._cache: dict[tuple, OperatorEstimate] = {}
        self._calibrations = {
            table: Calibration.from_mapping(values)
            for table, values in (self.package.meta.calibration.items() if self.package else [])
        }
        self._init_dimensions()

    # -- dimensions -------------------------------------------------------------

    def _init_dimensions(self) -> None:
        model = self.model
        tp = self.parallel_config.tensor_parallel_size
        pp = self.parallel_config.pipeline_parallel_size
        self.tp = tp
        self.heads_local = math.ceil(model.num_attention_heads / tp)
        self.kv_heads_local = local_kv_heads(model.num_key_value_heads, tp)
        self.q_dim_local = self.heads_local * model.head_dim
        self.kv_dim_local = (
            model.kv_dim // tp
            if model.kv_cache_dim is not None
            else self.kv_heads_local * model.head_dim
        )
        self.inter_local = math.ceil(model.intermediate_size / tp)
        self.vocab_local = math.ceil(model.vocab_size / tp)
        self.layers_local = stage_layer_count(model.num_layers, pp, self.rank_info.pp_rank)
        self.is_last_stage = self.rank_info.pp_rank == pp - 1
        self.moe_layers_local = (
            len(self.expert_placement.moe_layers_for_pp_rank(self.rank_info.pp_rank))
            if model.is_moe and self.expert_placement is not None
            else 0
        )
        self.dense_layers_local = max(0, self.layers_local - self.moe_layers_local)
        self.activation_bytes = activation_bytes_for(model.activation_dtype)
        ep_enabled = (
            self.parallel_config.enable_expert_parallel
            and self.expert_placement is not None
            and self.expert_placement.ep_rank_count > 1
        )
        self.ep_enabled = bool(ep_enabled)
        self.moe_tp = 1 if ep_enabled else tp
        self.moe_ep = self.expert_placement.ep_rank_count if ep_enabled else 1
        # All DP replicas dispatch into the same expert ranks; assume lockstep
        # steps so the EP group sees data_parallel_size times this rank's tokens.
        self.moe_token_multiplier = self.parallel_config.data_parallel_size if ep_enabled else 1
        routing = model.moe.routing.distribution if model.moe.routing else "uniform"
        self.moe_distributions = _MOE_DISTRIBUTION_CANDIDATES.get(routing, ("uniform", "balanced"))

    # -- public API -------------------------------------------------------------

    def estimate_step_latency(self, requests: list[Request]) -> float:
        if not requests:
            return 0.0
        if all(getattr(req, "needs_recompute", False) and req.recompute_tokens == 0 for req in requests):
            return 0.0
        self.stats.step_count += 1
        if is_context_build(requests):
            return self._context_step(requests)
        return self._decode_step(requests)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "operator_table",
            "fallback": self.fallback,
            "device_id": self.device.device_id,
            "device_family": self.device.family,
            "model_id": self.model.model_id,
            "dataset_version": self.package.dataset_version if self.package else None,
            "operator_backend": self.package.backend if self.package else "analytical",
            "tables": {name: len(rows) for name, rows in self.package.tables.items()} if self.package else {},
            "analytical_model": type(self.analytical).__name__,
            "decode_context_bucket": self.decode_context_bucket,
        }

    def stats_dict(self) -> dict[str, Any]:
        return self.stats.as_dict()

    # -- step composition -------------------------------------------------------

    def _context_step(self, requests: list[Request]) -> float:
        token_counts = [max(1, request_context_tokens(req)) for req in requests]
        kv_lens = [max(1, request_kv_len(req)) for req in requests]
        tokens = sum(token_counts)
        attention_us = 0.0
        # Group requests by (query_len, kv_len) so one lookup covers equal shapes.
        groups: Counter = Counter(zip(token_counts, kv_lens))
        for (q_len, kv_len), count in groups.items():
            attention_us += self._attention_prefill(count, q_len, kv_len)
        return self._compose(tokens=tokens, requests=len(requests), attention_us=attention_us, requests_list=requests)

    def _decode_step(self, requests: list[Request]) -> float:
        buckets: Counter = Counter()
        for req in requests:
            context = max(1, req.prefill_len + req.generation_idx)
            buckets[self._bucket(context)] += 1
        attention_us = sum(self._attention_decode(count, context) for context, count in buckets.items())
        return self._compose(tokens=len(requests), requests=len(requests), attention_us=attention_us, requests_list=requests)

    def _bucket(self, context_len: int) -> int:
        step = self.decode_context_bucket
        return int(math.ceil(context_len / step) * step)

    def _compose(self, *, tokens: int, requests: int, attention_us: float, requests_list: list[Request]) -> float:
        model = self.model
        h = model.hidden_size
        T = tokens
        stats = self.stats

        # Attention block shared by dense and MoE layers.
        attn_gemm_us = self._gemm(T, self.q_dim_local + 2 * self.kv_dim_local, h) + self._gemm(T, h, self.q_dim_local)
        attn_elem_us = (
            self._elementwise("rmsnorm", T, h)
            + self._elementwise("rope", T, self.q_dim_local + self.kv_dim_local)
            + self._elementwise("residual_add", T, h)
        )
        dense_ffn_gemm_us = 0.0
        dense_ffn_elem_us = 0.0
        if self.dense_layers_local:
            dense_ffn_gemm_us = self._gemm(T, model.ffn_projection_count * self.inter_local, h) + self._gemm(T, h, self.inter_local)
            dense_ffn_elem_us = self._elementwise("rmsnorm", T, h) + self._elementwise("residual_add", T, h)
            if model.gated:
                dense_ffn_elem_us += self._elementwise("swiglu", T, self.inter_local)

        moe_us = 0.0
        router_us = 0.0
        moe_elem_us = 0.0
        moe_comm_bytes = 0
        if self.moe_layers_local:
            router_us = self._gemm(T, model.moe.num_experts, h)
            moe_elem_us = self._elementwise("rmsnorm", T, h) + self._elementwise("residual_add", T, h)
            moe_us = self._moe_layers(T, requests_list)
            if self.ep_enabled:
                moe_comm_bytes = int(T * model.moe.num_experts_per_tok * h * self.activation_bytes)

        layer_gemm_us = self.layers_local * attn_gemm_us + self.dense_layers_local * dense_ffn_gemm_us + self.moe_layers_local * router_us
        layer_elem_us = self.layers_local * attn_elem_us + self.dense_layers_local * dense_ffn_elem_us + self.moe_layers_local * moe_elem_us
        layer_attention_us = self.layers_local * attention_us

        lm_head_us = 0.0
        if self.is_last_stage:
            lm_head_us = self._gemm(max(1, requests), self.vocab_local, h) + self._elementwise("rmsnorm", max(1, requests), h)

        overhead_us = self.device.step_overhead_us.value

        stats.record_component("gemm", layer_gemm_us * 1e-6)
        stats.record_component("attention", layer_attention_us * 1e-6)
        stats.record_component("elementwise", layer_elem_us * 1e-6)
        stats.record_component("moe", moe_us * 1e-6)
        stats.record_component("lm_head", lm_head_us * 1e-6)
        stats.record_component("overhead", overhead_us * 1e-6)

        compute_s = (layer_gemm_us + layer_attention_us + layer_elem_us + moe_us + lm_head_us + overhead_us) * 1e-6
        comm_s = self._communication(T, moe_comm_bytes)
        stats.record_component("comm", comm_s)
        return compute_s + comm_s

    def _communication(self, tokens: int, moe_comm_bytes: int) -> float:
        if self.communicator is None:
            return 0.0
        hidden_bytes = int(tokens * self.model.hidden_size * self.activation_bytes)
        latency = 0.0
        if self.tp > 1:
            self.communicator.record_tp_shard()
            # Row-parallel output projection and FFN down projection each
            # synchronize the hidden state: one all-reduce after attention in
            # every layer, one after the FFN in dense layers and in MoE layers
            # sharded with pure TP. Expert-parallel MoE layers replace the
            # FFN all-reduce with dispatch/combine all-to-all below.
            allreduce_count = self.layers_local + self.dense_layers_local
            if not self.ep_enabled:
                allreduce_count += self.moe_layers_local
            latency += self.communicator.estimate_tp_collective(hidden_bytes, count=allreduce_count)
        if self.ep_enabled and moe_comm_bytes > 0 and self.moe_layers_local:
            # One dispatch + one combine per MoE layer; measured DeepEP rows are
            # keyed by this rank's token count and the expert configuration.
            latency += self.communicator.estimate_ep_all2all(
                moe_comm_bytes,
                count=self.moe_layers_local,
                num_tokens=tokens,
                hidden_size=self.model.hidden_size,
                top_k=self.model.moe.num_experts_per_tok,
                num_experts=self.model.moe.num_experts,
                activation_bytes=self.activation_bytes,
            )
        if self.parallel_config.pipeline_parallel_size > 1:
            latency += self.communicator.estimate_pp_stage_transfer(hidden_bytes)
        return latency

    # -- operator queries -------------------------------------------------------

    def _gemm(self, m: int, n: int, k: int) -> float:
        if m <= 0 or n <= 0 or k <= 0:
            return 0.0
        return self._query("gemm", {"dtype": self.model.dtype, "m": int(m), "n": int(n), "k": int(k)}).latency_us

    def _elementwise(self, op_name: str, tokens: int, hidden: int) -> float:
        if tokens <= 0 or hidden <= 0:
            return 0.0
        return self._query(
            "elementwise",
            {"op_name": op_name, "dtype": self.model.activation_dtype, "num_tokens": int(tokens), "hidden_size": int(hidden)},
        ).latency_us

    def _attention_prefill(self, batch: int, q_len: int, kv_len: int) -> float:
        model = self.model
        key = {
            "attn_dtype": model.activation_dtype,
            "kv_cache_dtype": model.kv_cache_dtype,
            "batch_size": int(batch),
            "input_seq_len": int(q_len),
            "num_heads": self.heads_local,
            "num_kv_heads": self.kv_heads_local,
            "head_dim": model.head_dim,
            "window_size": model.sliding_window,
        }
        latency = self._query("context_attention", key).latency_us
        if kv_len > q_len:
            # Prefix-cache hit: queries attend to cached keys as well. Scale by
            # the analytical work ratio because tables assume kv_len == q_len.
            base = context_attention_work(batch, q_len, self.heads_local, self.kv_heads_local, model.head_dim, model.activation_dtype, model.kv_cache_dtype, model.sliding_window)
            full_keys = float(min(kv_len, model.sliding_window) if model.sliding_window else kv_len)
            # queries q_len each attend to (kv_len - q_len) cached keys + causal part of their own block
            full_flops = 4.0 * batch * q_len * ((kv_len - q_len) + 0.5 * q_len) * self.q_dim_local
            if model.sliding_window:
                full_flops = min(full_flops, 4.0 * batch * q_len * full_keys * self.q_dim_local)
            if base.flops > 0:
                latency *= max(1.0, full_flops / base.flops)
        return latency

    def _attention_decode(self, batch: int, context_len: int) -> float:
        model = self.model
        key = {
            "attn_dtype": model.activation_dtype,
            "kv_cache_dtype": model.kv_cache_dtype,
            "batch_size": int(batch),
            "context_len": int(context_len),
            "num_heads": self.heads_local,
            "num_kv_heads": self.kv_heads_local,
            "head_dim": model.head_dim,
            "window_size": model.sliding_window,
        }
        return self._query("generation_attention", key).latency_us

    def _moe_layers(self, tokens: int, requests: list[Request]) -> float:
        """Expert compute for all MoE layers on this rank, including straggler time."""
        model = self.model
        if self.expert_placement is None:
            return 0.0
        histogram = self.expert_routing.histogram_for_step(requests)
        per_rank_load = self._per_rank_load(histogram)
        imbalance = 1.0
        if per_rank_load:
            mean_load = sum(per_rank_load.values()) / max(1, self.expert_placement.ep_rank_count)
            max_load = max(per_rank_load.values())
            if mean_load > 0 and max_load > mean_load:
                imbalance = max_load / mean_load
        group_tokens = tokens * self.moe_token_multiplier
        base_key = {
            "dtype": model.dtype,
            "hidden_size": model.hidden_size,
            "inter_size": model.moe_intermediate_size,
            "top_k": model.moe.num_experts_per_tok,
            "num_experts": model.moe.num_experts,
            "tp_size": self.moe_tp,
            "ep_size": self.moe_ep,
        }
        balanced = self._query_moe({**base_key, "num_tokens": max(1, int(round(group_tokens)))})
        compute_s = balanced.latency_us * self.moe_layers_local * 1e-6
        straggler_s = 0.0
        if imbalance > 1.0:
            # The step waits for the busiest expert rank. Its load equals a
            # balanced layer fed with ``imbalance`` times as many tokens, which
            # keeps memory-bound small batches from being over-scaled.
            busiest = self._query_moe({**base_key, "num_tokens": max(1, int(round(group_tokens * imbalance)))})
            straggler_s = max(0.0, busiest.latency_us - balanced.latency_us) * self.moe_layers_local * 1e-6
        all2all_s = 0.0  # recorded by the communicator separately
        self.moe_stats.record_step(
            histogram=histogram,
            per_rank_load=per_rank_load,
            compute_latency=compute_s,
            all2all_latency=all2all_s,
            straggler_latency=straggler_s,
            all2all_event_count=1 if (self.ep_enabled and self.moe_layers_local) else 0,
        )
        return (compute_s + straggler_s) * 1e6

    def _query_moe(self, base_key: Mapping[str, Any]) -> OperatorEstimate:
        last_error: MissingOperatorDataError | None = None
        if self.lookup is not None and self.fallback != "analytical_only":
            for distribution in self.moe_distributions:
                key = {**base_key, "distribution": distribution}
                cached = self._cache.get(("moe", tuple(sorted(key.items()))))
                if cached is not None:
                    self.stats.record("moe", cached.match_type)
                    return cached
                try:
                    result = self.lookup.lookup("moe", key)
                except MissingOperatorDataError as exc:
                    last_error = exc
                    continue
                estimate = OperatorEstimate(result.latency_us, result.match_type, result.source_id)
                self._remember(("moe", tuple(sorted(key.items()))), estimate)
                self.stats.record("moe", estimate.match_type)
                return estimate
        key = {**base_key, "distribution": self.moe_distributions[0]}
        if self.fallback == "table_only":
            raise MissingOperatorDataError("moe", key, str(last_error) if last_error else "no moe table")
        if last_error is not None:
            self.stats.record_missing("moe", key, last_error.reason)
        return self._analytical("moe", key, gated=self.model.gated)

    def _query(self, table: str, key: Mapping[str, Any]) -> OperatorEstimate:
        identity = (table, tuple(sorted(key.items())))
        cached = self._cache.get(identity)
        if cached is not None:
            self.stats.record(table, cached.match_type)
            return cached
        estimate: OperatorEstimate | None = None
        if self.lookup is not None and self.fallback != "analytical_only":
            try:
                result = self.lookup.lookup(table, key)
                estimate = OperatorEstimate(result.latency_us, result.match_type, result.source_id)
            except MissingOperatorDataError as exc:
                if self.fallback == "table_only":
                    raise
                self.stats.record_missing(table, key, exc.reason)
        if estimate is None:
            estimate = self._analytical(table, key, gated=self.model.gated)
        self._remember(identity, estimate)
        self.stats.record(table, estimate.match_type)
        return estimate

    def _analytical(self, table: str, key: Mapping[str, Any], **context: Any) -> OperatorEstimate:
        calibration = self._calibrations.get(table) or self._calibrations.get("default")
        result = self.analytical.estimate(table, key, self.device, calibration, **context)
        return OperatorEstimate(result.latency_us, "analytical", result.source_id)

    def _analytical_reference_us(self, table: str, key: Mapping[str, Any]) -> float | None:
        """Uncalibrated analytical latency used as the growth reference for extrapolation."""
        try:
            return self.analytical.estimate(table, key, self.device, Calibration(), gated=self.model.gated).latency_us
        except Exception:
            return None

    def _remember(self, identity: tuple, estimate: OperatorEstimate) -> None:
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[identity] = estimate

    def _per_rank_load(self, histogram: Mapping[int, int]) -> dict[int, int]:
        if self.expert_placement is None:
            return {}
        if self._expert_to_rank is None:
            mapping: dict[int, int] = {}
            for ep_rank, experts in self.expert_placement.rank_to_experts.items():
                for expert_id in experts:
                    mapping[expert_id] = ep_rank
            self._expert_to_rank = mapping
        per_rank: dict[int, int] = {}
        for expert_id, count in histogram.items():
            ep_rank = self._expert_to_rank.get(expert_id)
            if ep_rank is None:
                continue
            per_rank[ep_rank] = per_rank.get(ep_rank, 0) + count
        return per_rank


def collective_model_for(device: DeviceSpec, placement, links, package: OperatorDataPackage | None) -> CollectiveModel:
    """Build the collective model a worker on ``device`` should use."""
    measured = OperatorLookup(package) if package is not None and "collective" in package.tables else None
    launch_us = device.analytical.get("collective_launch_us")
    return CollectiveModel(
        placement.topology,
        links,
        family=device.family,
        measured_lookup=measured,
        launch_us=float(launch_us) if launch_us is not None else None,
    )
