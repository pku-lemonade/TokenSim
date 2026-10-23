"""Ordinary and latent attention, FFN and elementwise operator components."""

from __future__ import annotations

from collections import Counter
import json
import math
from typing import Any, Mapping

from TokenSim.af.providers.attention import attention_core
from TokenSim.af.providers.context_parallel import (
    kv_parallel_size,
    local_attention_lengths,
)
from TokenSim.af.providers.compute import (
    OperatorComponent,
    _estimate_dataset_version,
    prefill_components,
)
from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype
from TokenSim.latency.operator_table import OperatorEstimate, OperatorResolver


class ComputeComponentsMixin:
    def _attention_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        tp_size: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        weight_dtype = model.operator_dtype("attention", query.weight_dtype or dtype)
        h = model.hidden_size
        geometry = model.attention_geometry(query.layer_id)
        kvp = kv_parallel_size(query)
        # KV-parallel ranks keep tp/kvp heads; attention_output applies O.
        attention_tp = tp_size // kvp
        heads_local = math.ceil(model.num_attention_heads / attention_tp)
        kv_heads_local = local_kv_heads(geometry.num_key_value_heads, attention_tp)
        q_local = heads_local * geometry.head_dim
        kv_local = kv_heads_local * geometry.head_dim
        components = [
            self._elementwise(resolver, "a_rmsnorm", "rmsnorm", tokens, h, dtype),
            self._gemm(
                resolver,
                "qkv_projection",
                tokens,
                q_local + geometry.kv_projections * kv_local,
                h,
                weight_dtype,
            ),
            self._elementwise(
                resolver, "rope", "rope", tokens, q_local + kv_local, dtype
            ),
        ]
        if model.qk_norm:
            components[2:2] = [
                self._elementwise(
                    resolver,
                    "q_norm",
                    "rmsnorm",
                    tokens * heads_local,
                    geometry.head_dim,
                    dtype,
                ),
                self._elementwise(
                    resolver,
                    "k_norm",
                    "rmsnorm",
                    tokens * kv_heads_local,
                    geometry.head_dim,
                    dtype,
                ),
            ]
        lengths = (
            local_attention_lengths(query, model.attention_window(query.layer_id))
            if kvp > 1
            else None
        )
        components.extend(
            attention_core(
                self,
                resolver,
                query,
                tokens,
                heads_local,
                kv_heads_local,
                dtype,
                lengths,
            )
        )
        if kvp == 1:
            components.append(
                self._gemm(
                    resolver, "output_projection", tokens, h, q_local, weight_dtype
                )
            )
        if model.architecture != "gemma4_text":
            components.append(
                self._elementwise(
                    resolver, "a_residual_add", "residual_add", tokens, h, dtype
                )
            )
        else:
            components.append(
                self._elementwise(
                    resolver,
                    "v_norm",
                    "rmsnorm",
                    tokens * kv_heads_local,
                    geometry.head_dim,
                    dtype,
                )
            )
        return [component for component in components if component is not None]

    def _router_components(
        self,
        resolver: OperatorResolver,
        query: AFComputeQuery,
        tokens: int,
        dtype: str,
    ) -> list[OperatorComponent]:
        model = self.model
        weight_dtype = model.operator_dtype("router", query.weight_dtype or dtype)
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
        weight_dtype = model.operator_dtype(
            "routed_experts", query.weight_dtype or dtype
        )
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
            table = "moe"
            exact_key = {}
            if estimate.match_type == "analytical":
                # The runtime already knows the local histogram. Preserve its
                # assignment and active-expert counts through analytical fallback.
                table = "routed_experts"
                exact_key = {
                    "dtype": weight_dtype,
                    "hidden_size": h,
                    "inter_size": model.moe_intermediate_size // tp_size,
                    "expert_tokens": json.dumps(
                        sorted(count for _, count in workload), separators=(",", ":")
                    ),
                }
                if weight_dtype in {"mxfp4", "mxfp4_wo"}:
                    exact_key["activation_dtype"] = dtype
                estimate = resolver.analytical_estimate(
                    table, exact_key, gated=model.gated
                )
            components.append(
                OperatorComponent(
                    name="moe_expert",
                    table=table,
                    key={
                        **base_key,
                        **exact_key,
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
