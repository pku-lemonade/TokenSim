"""Price primitive operators and route measured-table misses to formulas."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.analytical import Calibration
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.ragged_attention import generation_attention_query
from TokenSim.errors import ConfigurationError
from .stats import OperatorEstimate


class OperatorQueries:
    """Price primitive operators and route measured-table misses to formulas."""

    def shared_experts(self, rows: int, *, tp: int = 1) -> float:
        """One MoE layer's shared experts for ``rows`` tokens, TP-sharded ``tp`` ways."""
        model = self.model
        width = model.moe_intermediate_size * model.moe.num_shared_experts
        if rows <= 0 or not width:
            return 0.0
        width = math.ceil(width / tp)
        dtype = model.operator_dtype("shared_expert")
        latency = self.gemm(
            rows, model.ffn_projection_count * width, model.hidden_size, dtype=dtype
        )
        if model.gated:
            latency += self.elementwise("swiglu", rows, width)
        return latency + self.gemm(rows, model.hidden_size, width, dtype=dtype)

    def gemm(self, m: int, n: int, k: int, *, dtype: str | None = None) -> float:
        if m <= 0 or n <= 0 or k <= 0:
            return 0.0
        return self.query(
            "gemm",
            {"dtype": dtype or self.model.dtype, "m": int(m), "n": int(n), "k": int(k)},
        ).latency_us

    def elementwise(self, op_name: str, tokens: int, hidden: int) -> float:
        if tokens <= 0 or hidden <= 0:
            return 0.0
        return self.query(
            "elementwise",
            {
                "op_name": op_name,
                "dtype": self.model.activation_dtype,
                "num_tokens": int(tokens),
                "hidden_size": int(hidden),
            },
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
        if kv_len <= q_len:
            return self.query("context_attention", key).latency_us
        full = {**key, "context_len": int(kv_len)}
        if self.fallback == "analytical_only":
            return self.query("context_attention", full).latency_us
        estimate = self.query("context_attention", key)
        if estimate.match_type == "analytical":
            return self.query("context_attention", full).latency_us
        base_us = self._analytical_reference_us("context_attention", key)
        full_us = self._analytical_reference_us("context_attention", full)
        if base_us is None or full_us is None or base_us <= 0:
            raise ConfigurationError(
                "cached prefill requires an analytical latency reference"
            )
        # Scale measured device service by roofline growth, including KV reads.
        # Multiplying launch overhead by the FLOP ratio creates seconds of
        # fictitious latency for a short query over a long cached prefix.
        return estimate.latency_us * max(1.0, full_us / base_us)

    def _attention_decode(
        self, batch: int, context_len: int, query_width: int = 1
    ) -> float:
        return self._attention_decode_batch(
            {(int(context_len), int(query_width)): int(batch)}
        )

    def _attention_decode_batch(self, buckets) -> float:
        """Price one ragged q=1 call; verification extends retain context work."""
        contexts = Counter()
        latency = 0.0
        for (context, width), count in buckets.items():
            if width == 1:
                contexts[context] += count
            else:
                latency += self._attention_prefill(count, width, context)
        if contexts:
            table, key = generation_attention_query(
                {
                    "attn_dtype": self.model.activation_dtype,
                    "kv_cache_dtype": self.model.kv_cache_dtype,
                    "num_heads": self.heads_local,
                    "num_kv_heads": self.kv_heads_local,
                    "head_dim": self.model.head_dim,
                    "window_size": self.model.sliding_window,
                },
                contexts,
            )
            latency += self.query(table, key).latency_us
        return latency

    def _moe_layers(self, tokens: int, requests: list[Request]) -> float:
        """Expert compute for all MoE layers on this rank, including straggler time."""
        model = self.model
        if self.expert_placement is None:
            return 0.0
        histogram = self.expert_routing.histogram_for_step(requests)
        per_rank_load = self._per_rank_load(histogram)
        if self.native_moe:
            total_us, routed_us, shared_us = self.native_moe.estimate(tokens)
            self.moe_stats.record_step(
                histogram=histogram,
                per_rank_load=per_rank_load,
                compute_latency=total_us * 1e-6,
                straggler_latency=0,
                all2all_event_count=1,
            )
            return total_us
        imbalance = 1.0
        if per_rank_load:
            mean_load = sum(per_rank_load.values()) / max(
                1, self.expert_placement.ep_rank_count
            )
            max_load = max(per_rank_load.values())
            if mean_load > 0 and max_load > mean_load:
                imbalance = max_load / mean_load
        group_tokens = tokens * self.moe_token_multiplier
        base_key = {
            "dtype": self.moe_dtype,
            "hidden_size": model.hidden_size,
            "inter_size": model.moe_intermediate_size,
            "top_k": model.moe.num_experts_per_tok,
            "num_experts": model.moe.num_experts,
            "tp_size": self.moe_tp,
            "ep_size": self.moe_ep,
        }
        balanced = self.moe_query(
            {**base_key, "num_tokens": max(1, int(round(group_tokens)))}
        )
        compute_s = balanced.latency_us * self.moe_layers_local * 1e-6
        straggler_s = 0.0
        # A power-law measurement already includes its distribution's slow
        # ranks. When a model binding fixes the measured distribution,
        # synthetic or trace histograms must not change that cost.
        measured_skew = bool(
            balanced.workload_distribution
            and balanced.workload_distribution.startswith("power_law")
        )
        if imbalance > 1.0 and not measured_skew:
            # The step waits for the busiest expert rank. Its load equals a
            # balanced layer fed with ``imbalance`` times as many tokens, which
            # keeps memory-bound small batches from being over-scaled.
            busiest = self.moe_query(
                {**base_key, "num_tokens": max(1, int(round(group_tokens * imbalance)))}
            )
            straggler_s = (
                max(0.0, busiest.latency_us - balanced.latency_us)
                * self.moe_layers_local
                * 1e-6
            )
        # Shared experts see every token: the rank's sequence-parallel shard
        # with replicated weights, or all tokens with TP-sharded weights.
        if self.ep_enabled and self.parallel_config.sequence_parallel_moe(False):
            shared_us = self.shared_experts(math.ceil(tokens / self.tp))
        else:
            shared_us = self.shared_experts(tokens, tp=self.tp)
        compute_s += shared_us * self.moe_layers_local * 1e-6
        # Dispatch/combine time is recorded by the communicator.
        self.moe_stats.record_step(
            histogram=histogram,
            per_rank_load=per_rank_load,
            compute_latency=compute_s,
            straggler_latency=straggler_s,
            all2all_event_count=1 if (self.ep_enabled and self.moe_layers_local) else 0,
        )
        return (compute_s + straggler_s) * 1e6

    def moe_query(self, base_key: Mapping[str, Any]) -> OperatorEstimate:
        """Rows of the first routing distribution the table has, else analytical."""
        keys = [{**base_key, "distribution": d} for d in self.moe_distributions]
        for key in keys:
            cached = self._cache.get(("moe", tuple(sorted(key.items()))))
            if cached is not None:
                self.stats.record("moe", cached.match_type)
                return cached
        if self.fallback != "analytical_only":
            error = self._no_package("moe", keys[0])
            for key in keys if error is None else ():
                try:
                    result = self.lookup.lookup("moe", key)
                except MissingOperatorDataError as exc:
                    error = exc
                    continue
                estimate = OperatorEstimate(
                    result.latency_us,
                    result.match_type,
                    result.source_id,
                    workload_distribution=key["distribution"],
                )
                self._remember(("moe", tuple(sorted(key.items()))), estimate)
                self.stats.record("moe", estimate.match_type)
                return estimate
            if self.fallback == "table_only":
                raise error
            self.stats.record_missing(error)
        estimate = self._analytical("moe", keys[0], gated=self.model.gated)
        self._remember(("moe", tuple(sorted(keys[0].items()))), estimate)
        self.stats.record("moe", estimate.match_type)
        return estimate

    def query(self, table: str, key: Mapping[str, Any]) -> OperatorEstimate:
        identity = (table, tuple(sorted(key.items())))
        cached = self._cache.get(identity)
        if cached is not None:
            self.stats.record(table, cached.match_type)
            return cached
        estimate: OperatorEstimate | None = None
        if self.fallback != "analytical_only":
            try:
                error = self._no_package(table, key)
                if error is not None:
                    raise error
                result = self.lookup.lookup(table, key)
                estimate = OperatorEstimate(
                    result.latency_us, result.match_type, result.source_id
                )
            except MissingOperatorDataError as exc:
                if self.fallback == "table_only":
                    raise
                self.stats.record_missing(exc)
        if estimate is None:
            estimate = self._analytical(table, key, gated=self.model.gated)
        self._remember(identity, estimate)
        self.stats.record(table, estimate.match_type)
        return estimate

    def _no_package(
        self, table: str, key: Mapping[str, Any]
    ) -> MissingOperatorDataError | None:
        """The miss every query records when the device has no operator package."""
        if self.lookup is not None:
            return None
        return MissingOperatorDataError(
            table, key, "no operator package for this device", kind="table_absent"
        )

    def _analytical(
        self, table: str, key: Mapping[str, Any], **context: Any
    ) -> OperatorEstimate:
        calibration = self._calibrations.get(table) or self._calibrations.get("default")
        result = self.analytical.estimate(
            table, key, self.device, calibration, **context
        )
        return OperatorEstimate(result.latency_us, "analytical", result.source_id)

    def _analytical_reference_us(
        self, table: str, key: Mapping[str, Any]
    ) -> float | None:
        """Uncalibrated analytical latency used as the growth reference for extrapolation."""
        try:
            return self.analytical.estimate(
                table, key, self.device, Calibration(), gated=self.model.gated
            ).latency_us
        except Exception:
            return None

    def _remember(self, identity: tuple, estimate: OperatorEstimate) -> None:
        self._cache[identity] = estimate
