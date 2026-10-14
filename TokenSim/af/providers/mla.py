"""Operator mla."""

from __future__ import annotations

from collections import Counter
import json

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype


class MLAComputeMixin:
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
