from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from TokenSim.af.latency import AFComputeQuery, AFServiceEstimate, AFTransferQuery
from TokenSim.dtypes import canonicalize_dtype, dtype_size_bytes
from TokenSim.operator_data.lookup import (
    LookupResult,
    MissingOperatorDataError,
    OperatorDataLookup,
)
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import SCHEMA_VERSION, encode_positive_vector
from TokenSim.mla import MLADimensions

if TYPE_CHECKING:
    from TokenSim.config.cluster_config import AFEffectiveDTypes


@dataclass(frozen=True)
class ModelShape:
    model_id: str
    hidden_size: int
    dense_intermediate_size: int
    routed_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int = 0
    num_shared_experts: int = 0
    top_k: int = 1
    tp_size: int = 1
    ep_size: int = 1
    window_size: int = 0
    weight_dtype: str = "bf16"
    activation_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"
    attention_type: str = "auto"
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    rope_head_dim: int = 0
    nope_head_dim: int = 0
    distribution: str = "balanced"
    a_elementwise: tuple[str, ...] = (
        "rmsnorm",
        "rope",
        "residual_add",
    )
    f_elementwise: tuple[str, ...] = ("rmsnorm", "residual_add")
    component_includes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    post_attention_norm_on_a: bool = False

    v_head_dim: int = 0

    @property
    def mla_dimensions(self) -> MLADimensions:
        return MLADimensions.from_model(self)

    @property
    def head_dim(self) -> int:
        if self.resolved_attention_type == "mla":
            return self.mla_dimensions.head_dim
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"model {self.model_id!r} hidden_size must divide num_attention_heads"
            )
        return self.hidden_size // self.num_attention_heads

    @property
    def q_dimension(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dimension(self) -> int:
        if self.resolved_attention_type == "mla":
            return self.mla_dimensions.cache_width
        return self.num_key_value_heads * self.head_dim

    @property
    def shared_intermediate_size(self) -> int:
        return self.routed_intermediate_size * self.num_shared_experts

    @property
    def resolved_attention_type(self) -> str:
        value = self.attention_type.strip().lower()
        if value == "auto":
            if self.num_key_value_heads == self.num_attention_heads:
                return "mha"
            if self.num_key_value_heads == 1:
                return "mqa"
            return "gqa"
        return value


class ModelOperatorManifest:
    def __init__(self, models: Mapping[str, ModelShape]) -> None:
        self._models = dict(models)
        if not self._models:
            raise ValueError("model operator manifest must contain at least one model")
        for model in self._models.values():
            _validate_model(model)

    @classmethod
    def load(cls, path: str | Path) -> "ModelOperatorManifest":
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("model operator manifest requires PyYAML") from exc
        manifest_path = Path(path)
        try:
            document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"failed to parse model operator manifest {path}: {exc}"
            ) from exc
        if not isinstance(document, Mapping):
            raise ValueError("model operator manifest root must be a mapping")
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"operator-data schema migration required for model manifest "
                f"{manifest_path}: found schema_version="
                f"{document.get('schema_version')!r}, expected {SCHEMA_VERSION}; "
                "regenerate the manifest"
            )
        raw_models = document.get("models")
        if not isinstance(raw_models, Mapping) or not raw_models:
            raise ValueError(
                "model operator manifest models must be a non-empty mapping"
            )
        models: dict[str, ModelShape] = {}
        for model_id, raw in raw_models.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"model {model_id!r} entry must be a mapping")
            missing_shapes = [
                field_name
                for field_name in (
                    "dense_intermediate_size",
                    "routed_intermediate_size",
                    "num_shared_experts",
                )
                if field_name not in raw
            ]
            if missing_shapes:
                legacy_hint = (
                    "; legacy intermediate_size cannot describe Dense, Routed, "
                    "and Shared Expert shapes independently"
                    if "intermediate_size" in raw
                    else ""
                )
                raise ValueError(
                    f"operator-data schema migration required for model "
                    f"{model_id!r}: missing model shape fields {missing_shapes}"
                    f"{legacy_hint}; regenerate the manifest"
                )
            if "intermediate_size" in raw:
                raise ValueError(
                    f"operator-data schema migration required for model "
                    f"{model_id!r}: legacy intermediate_size is no longer "
                    "supported; use dense_intermediate_size, "
                    "routed_intermediate_size, and num_shared_experts"
                )
            missing_dtypes = [
                role
                for role in ("weight_dtype", "activation_dtype", "kv_cache_dtype")
                if role not in raw
            ]
            if missing_dtypes:
                raise ValueError(
                    f"operator-data schema migration required for model "
                    f"{model_id!r}: missing dtype coverage {missing_dtypes}; "
                    "regenerate the manifest"
                )
            values = dict(raw)
            for role in ("weight_dtype", "activation_dtype", "kv_cache_dtype"):
                values[role] = canonicalize_dtype(str(values[role]), role=role)
            for name in ("a_elementwise", "f_elementwise"):
                if name in values:
                    values[name] = tuple(values[name])
            includes = values.get("component_includes", {})
            values["component_includes"] = {
                str(name): tuple(logical_ops) for name, logical_ops in includes.items()
            }
            model = ModelShape(model_id=str(model_id), **values)
            _validate_model(model)
            models[str(model_id)] = model
        return cls(models)

    def model(self, model_id: str) -> ModelShape:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ValueError(
                f"model operator manifest has no model_id={model_id!r}"
            ) from exc

    def validate_effective_dtypes(
        self, model_id: str, effective_dtypes: "AFEffectiveDTypes"
    ) -> None:
        model = self.model(model_id)
        for role in ("weight_dtype", "activation_dtype", "kv_cache_dtype"):
            manifest_value = canonicalize_dtype(
                getattr(model, role), role=role
            )
            effective_value = canonicalize_dtype(
                getattr(effective_dtypes, role), role=role
            )
            if manifest_value != effective_value:
                raise ValueError(
                    f"operator manifest dtype mismatch: model={model_id!r}, "
                    f"role={role!r}, effective={effective_value!r}, "
                    f"manifest={manifest_value!r}; manifest coverage must not "
                    "override runtime semantics"
                )


DEFAULT_INCLUDES: Mapping[str, tuple[str, ...]] = {
    "a_rmsnorm": ("a_rmsnorm",),
    "qkv_projection": ("qkv_projection",),
    "rope": ("rope",),
    "attention_core": ("attention_core", "softmax"),
    "mla_attention_core": ("mla_attention_core", "softmax", "mla_cache_write"),
    "output_projection": ("output_projection",),
    "a_residual_add": ("a_residual_add",),
    "a_data_reorder": ("a_data_reorder",),
    "f_rmsnorm": ("f_rmsnorm",),
    "router": ("router",),
    "moe_expert": ("moe_expert", "swiglu"),
    "dense_gate_up": ("dense_gate_up",),
    "dense_swiglu": ("dense_swiglu",),
    "dense_down": ("dense_down",),
    "f_residual_add": ("f_residual_add",),
    "f_data_reorder": ("f_data_reorder",),
    "routed_expert": ("routed_expert", "swiglu"),
    "shared_expert": ("shared_expert", "swiglu"),
    "routing_weight_multiply": ("routing_weight_multiply",),
    "routed_reduce": ("routed_reduce",),
    "shared_sum": ("shared_sum",),
    "encode": ("encode",),
    "decode": ("decode",),
}


class OperatorDataComputeLatencyProvider:
    def __init__(
        self,
        lookup: OperatorDataLookup,
        manifest: ModelOperatorManifest,
        backend: str,
    ) -> None:
        self.lookup = lookup
        self.manifest = manifest
        self.backend = backend.strip().lower()

    def validate_effective_dtypes(
        self, model_id: str, effective_dtypes: "AFEffectiveDTypes"
    ) -> None:
        self.manifest.validate_effective_dtypes(model_id, effective_dtypes)

    def estimate(self, query: AFComputeQuery) -> AFServiceEstimate:
        model = self.manifest.model(query.model_id)
        tokens, local_hidden, global_hidden, tp_size = self._validate_local_work(
            query, model
        )
        self._validate_stage_semantics(query, model, tokens)
        if query.stage in {"a", "router", "expert"}:
            _validate_split_composition(model)
        try:
            if query.stage == "a":
                components = self._attention_components(
                    query,
                    model,
                    tokens,
                    local_hidden,
                    global_hidden,
                    tp_size,
                    include_tail_elementwise=True,
                )
            elif query.stage == "attention":
                components = self._attention_components(
                    query,
                    model,
                    tokens,
                    local_hidden,
                    global_hidden,
                    tp_size,
                    include_tail_elementwise=False,
                )
            elif query.stage == "router":
                components = self._router_components(
                    query, model, tokens, global_hidden
                )
            elif query.stage == "expert":
                components = self._expert_components(
                    query, model, tokens, global_hidden, tp_size
                )
            elif query.stage == "dense_ffn":
                components = self._dense_ffn_components(
                    query, model, tokens, global_hidden, tp_size
                )
            elif query.stage == "routed_expert":
                components = self._routed_expert_components(
                    query, model, tokens, global_hidden, tp_size
                )
            elif query.stage == "shared_expert":
                components = self._shared_expert_components(
                    query, model, tokens, global_hidden, tp_size
                )
            elif query.stage == "weighted_combine":
                components = self._weighted_combine_components(
                    query, tokens, local_hidden
                )
            elif query.stage == "residual":
                components = [
                    self._element_component(
                        "a_residual_add",
                        "residual_add",
                        query,
                        tokens,
                        local_hidden,
                    )
                ]
            elif query.stage == "layout":
                components = [
                    self._element_component(
                        "a_data_reorder",
                        "data_reorder",
                        query,
                        tokens,
                        local_hidden,
                    )
                ]
            elif query.stage in {"encode", "decode"}:
                components = [self._conversion_component(query, local_hidden)]
            else:
                raise ValueError(f"unsupported AF compute stage={query.stage!r}")
        except MissingOperatorDataError as exc:
            rank = (query.dp_rank, query.ep_rank, query.tp_rank)
            raise MissingOperatorDataError(
                exc.table_name,
                exc.key,
                f"{exc.reason}; stage={query.stage!r}, operation={query.operation!r}, "
                f"rank(dp,ep,tp)={rank}, local_shape={dict(query.local_shape)!r}, "
                f"local_token_workload={query.local_token_count}",
            ) from exc
        _validate_component_coverage(components, model.component_includes)
        latency_us = sum(float(component["latency_us"]) for component in components)
        match_type = (
            "interpolated"
            if any(item["match_type"] == "interpolated" for item in components)
            else "exact"
        )
        source_ids = tuple(dict.fromkeys(item["source_id"] for item in components))
        if not source_ids:
            source_id = "composition:empty"
        elif len(source_ids) == 1:
            source_id = source_ids[0]
        else:
            source_id = "compose:" + ":".join(source_ids)
        return AFServiceEstimate(
            service_duration=latency_us / 1e6,
            source_id=source_id,
            dataset_version=self.lookup.package.dataset_version,
            match_type=match_type,
            details={
                "dataset_version": self.lookup.package.dataset_version,
                "match_type": match_type,
                "stage": query.stage,
                "operation": query.operation,
                "logical_participant": query.logical_participant,
                "physical_worker": query.physical_worker,
                "local_shape": dict(query.local_shape),
                "local_token_count": tokens,
                "expert_ids": query.expert_ids,
                "expert_workload": dict(query.expert_workload),
                "layer_type": query.layer_type,
                "original_token_count": query.original_token_count,
                "assignment_count": query.assignment_count,
                "input_edge_count": query.input_edge_count,
                "output_token_count": query.output_token_count,
                "routing_edges": _routing_edge_details(query),
                "tensor_slices": _tensor_slice_details(query),
                "effective_dtypes": {
                    "weight_dtype": _query_compute_dtypes(query, model)[0],
                    "activation_dtype": _query_compute_dtypes(query, model)[1],
                    "kv_cache_dtype": _query_compute_dtypes(query, model)[2],
                    "source_dtype": getattr(query, "source_dtype", None),
                    "target_dtype": getattr(query, "target_dtype", None),
                },
                "components": components,
                "total_latency_us": latency_us,
            },
        )

    def _attention_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        local_hidden: int,
        global_hidden: int,
        tp_size: int,
        *,
        include_tail_elementwise: bool,
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        common = {"device": device, "backend": self.backend}
        weight_dtype, activation_dtype, kv_cache_dtype = _query_compute_dtypes(
            query, model
        )
        if model.resolved_attention_type == "mla":
            from TokenSim.operator_data.mla import attention_components
            return attention_components(self, query, model, tokens, global_hidden, tp_size,
                weight_dtype, activation_dtype, kv_cache_dtype, include_tail_elementwise)
        local_query_heads = _exact_tp_shard(
            model.num_attention_heads,
            tp_size,
            "Attention query heads",
        )
        local_kv_heads = _local_kv_heads(model.num_key_value_heads, tp_size)
        local_query_width = local_query_heads * model.head_dim
        local_kv_width = local_kv_heads * model.head_dim
        results: list[dict[str, Any]] = []
        element_names = {
            "rmsnorm": "a_rmsnorm",
            "rope": "rope",
            "residual_add": "a_residual_add",
            "data_reorder": "a_data_reorder",
        }
        for op in model.a_elementwise:
            if op == "rmsnorm":
                results.append(
                    self._element_component(
                        element_names[op], op, query, tokens, local_hidden
                    )
                )
        results.append(
            self._component(
                "qkv_projection",
                self.lookup.lookup(
                    "gemm",
                    {
                        **common,
                        "weight_dtype": weight_dtype,
                        "activation_dtype": activation_dtype,
                        "m": tokens,
                        "n": local_query_width + 2 * local_kv_width,
                        "k": global_hidden,
                    },
                ),
            )
        )
        for op in model.a_elementwise:
            if op == "rope":
                results.append(
                    self._element_component(
                        element_names[op], op, query, tokens, local_hidden
                    )
                )
        batch_size = max(1, len(query.context_lengths))
        if query.phase == "prefill":
            seq_len = max(1, math.ceil(tokens / batch_size))
            table_name = "context_attention"
            length_key = {"input_seq_len": seq_len}
            attention_component = self._component(
                "attention_core",
                self.lookup.lookup(
                    table_name,
                    {
                        **common,
                        "attention_type": model.resolved_attention_type,
                        "activation_dtype": activation_dtype,
                        "kv_cache_dtype": kv_cache_dtype,
                        "batch_size": batch_size,
                        **length_key,
                        "num_heads": local_query_heads,
                        "num_kv_heads": local_kv_heads,
                        "head_dim": model.head_dim,
                        "window_size": model.window_size,
                    },
                ),
            )
        else:
            attention_component = self._decode_attention_component(
                query,
                model,
                common,
                activation_dtype,
                kv_cache_dtype,
                local_query_heads,
                local_kv_heads,
            )
        results.append(attention_component)
        results.append(
            self._component(
                "output_projection",
                self.lookup.lookup(
                    "gemm",
                    {
                        **common,
                        "weight_dtype": weight_dtype,
                        "activation_dtype": activation_dtype,
                        "m": tokens,
                        "n": global_hidden,
                        "k": local_query_width,
                    },
                ),
            )
        )
        for op in model.a_elementwise:
            if include_tail_elementwise and op in {"residual_add", "data_reorder"}:
                results.append(
                    self._element_component(
                        element_names[op], op, query, tokens, local_hidden
                    )
                )
        if model.post_attention_norm_on_a and query.stage == "attention" and query.layer_type == "moe":
            results.append(self._element_component(
                "post_attention_rmsnorm", "rmsnorm", query, tokens, global_hidden))
        return results

    def _decode_attention_component(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        common: Mapping[str, Any],
        activation_dtype: str,
        kv_cache_dtype: str,
        local_query_heads: int,
        local_kv_heads: int,
    ) -> dict[str, Any]:
        contexts = query.context_lengths or (1,)
        encode_positive_vector(contexts)  # Validate before window clamping.
        if len(contexts) != query.local_token_count:
            raise ValueError("Decode Attention requires one context length per local token")
        effective = tuple(min(length, model.window_size) if model.window_size > 0
                          else length for length in contexts)
        key = {
            **common,
            "attention_type": model.resolved_attention_type,
            "activation_dtype": activation_dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "batch_size": len(contexts),
            "num_heads": local_query_heads,
            "num_kv_heads": local_kv_heads,
            "head_dim": model.head_dim,
            "window_size": model.window_size,
        }
        if len(set(effective)) == 1:
            table = "generation_attention"
            key["context_len"] = effective[0]
        else:
            table = "generation_attention_batch"
            key["context_lengths"] = encode_positive_vector(effective)
        result = self._component("attention_core", self.lookup.lookup(table, key))
        result.update({
            "execution_unit": "attention_batch",
            "table": table,
            "per_request": [{"request_index": index, "context_len": raw,
                             "effective_context_len": length}
                            for index, (raw, length) in enumerate(zip(contexts, effective))],
        })
        return result

    def _router_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        global_hidden: int,
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        common = {"device": device, "backend": self.backend}
        weight_dtype, activation_dtype, _ = _query_compute_dtypes(query, model)
        results: list[dict[str, Any]] = []
        for op in model.f_elementwise:
            if op == "rmsnorm" and not (model.post_attention_norm_on_a and query.layer_type == "moe"):
                results.append(
                    self._element_component(
                        "f_rmsnorm", op, query, tokens, global_hidden
                    )
                )
        if model.num_experts > 0:
            results.append(
                self._component(
                    "router",
                    self.lookup.lookup(
                        "gemm",
                        {
                            **common,
                            "weight_dtype": weight_dtype,
                            "activation_dtype": activation_dtype,
                            "m": tokens,
                            "n": model.num_experts,
                            "k": global_hidden,
                        },
                    ),
                )
            )
        return results

    def _dense_ffn_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        global_hidden: int,
        tp_size: int,
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        common = {"device": device, "backend": self.backend}
        weight_dtype, activation_dtype, _ = _query_compute_dtypes(query, model)
        local_intermediate = _positive_shape_value(
            query, "intermediate_size", "Dense FFN"
        )
        results: list[dict[str, Any]] = []
        if "rmsnorm" in model.f_elementwise:
            results.append(
                self._element_component(
                    "f_rmsnorm", "rmsnorm", query, tokens, global_hidden
                )
            )
        results.extend(
            (
                self._component(
                    "dense_gate_up",
                    self.lookup.lookup(
                        "gemm",
                        {
                            **common,
                            "weight_dtype": weight_dtype,
                            "activation_dtype": activation_dtype,
                            "m": tokens,
                            "n": 2 * local_intermediate,
                            "k": global_hidden,
                        },
                    ),
                ),
                self._element_component(
                    "dense_swiglu",
                    "swiglu",
                    query,
                    tokens,
                    local_intermediate,
                ),
                self._component(
                    "dense_down",
                    self.lookup.lookup(
                        "gemm",
                        {
                            **common,
                            "weight_dtype": weight_dtype,
                            "activation_dtype": activation_dtype,
                            "m": tokens,
                            "n": global_hidden,
                            "k": local_intermediate,
                        },
                    ),
                ),
            )
        )
        return results

    def _routed_expert_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        global_hidden: int,
        tp_size: int,
    ) -> list[dict[str, Any]]:
        workload = self._validate_expert_workload(query, tokens, model.num_experts)
        local_intermediate = _positive_shape_value(
            query, "intermediate_size", "Routed Expert"
        )
        return [
            self._moe_expert_component(
                query,
                model,
                workload,
                global_hidden,
                tp_size,
                {
                    "device": query.device_type.strip().lower(),
                    "backend": self.backend,
                },
                component_name="routed_expert",
                global_intermediate=local_intermediate * tp_size,
                ep_size=(1 if tp_size > 1 else max(1, len(query.ep_group))),
            )
        ]

    def _shared_expert_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        global_hidden: int,
        tp_size: int,
    ) -> list[dict[str, Any]]:
        local_intermediate = _positive_shape_value(
            query, "intermediate_size", "Shared Expert"
        )
        shared_intermediate = local_intermediate * tp_size
        weight_dtype, activation_dtype, _ = _query_compute_dtypes(query, model)
        result = self.lookup.lookup(
            "moe",
            {
                "device": query.device_type.strip().lower(),
                "backend": self.backend,
                "operation": "shared_expert",
                "weight_dtype": weight_dtype,
                "activation_dtype": activation_dtype,
                "phase": query.phase,
                "distribution": model.distribution,
                "num_tokens": tokens,
                "hidden_size": global_hidden,
                "inter_size": shared_intermediate,
                "top_k": 1,
                "num_experts": 1,
                "tp_size": tp_size,
                "ep_size": 1,
            },
        )
        component = self._component("shared_expert", result)
        component["original_token_count"] = tokens
        component["shared_intermediate_size"] = shared_intermediate
        component["local_intermediate_size"] = local_intermediate
        return [component]

    def _weighted_combine_components(
        self,
        query: AFComputeQuery,
        tokens: int,
        local_hidden: int,
    ) -> list[dict[str, Any]]:
        routing_edges = tuple(getattr(query, "routing_edges", ()))
        if not routing_edges:
            raise ValueError(
                "weighted_combine requires non-empty routing_edges so routing "
                "weights and token ownership are not lost"
            )
        edge_details = _routing_edge_details(query)
        routed_edge_count = len(routing_edges)
        if query.assignment_count != routed_edge_count:
            raise ValueError(
                "weighted_combine assignment_count must equal routing edge count; "
                f"assignment_count={query.assignment_count}, edges={routed_edge_count}"
            )
        token_ids = {
            (item["request_id"], item["token_id"]) for item in edge_details
        }
        if len(token_ids) != tokens:
            raise ValueError(
                "weighted_combine routing edges do not cover exactly the output "
                f"tokens; unique_tokens={len(token_ids)}, output_tokens={tokens}"
            )
        edges_per_token: dict[tuple[int, int], set[int]] = {}
        for item in edge_details:
            token_key = (item["request_id"], item["token_id"])
            top_k_indices = edges_per_token.setdefault(token_key, set())
            if item["top_k_index"] in top_k_indices:
                raise ValueError(
                    "weighted_combine contains duplicate top_k_index for token "
                    f"{token_key!r}"
                )
            top_k_indices.add(item["top_k_index"])
        edge_counts = {len(indices) for indices in edges_per_token.values()}
        if len(edge_counts) != 1:
            raise ValueError(
                "weighted_combine requires a consistent Top-K edge count per token; "
                f"counts={sorted(edge_counts)}"
            )

        slices = tuple(query.tensor_slices)
        routed_slices = tuple(
            item for item in slices if str(item.partial_id or "") != "shared"
        )
        shared_slices = tuple(
            item for item in slices if str(item.partial_id or "") == "shared"
        )
        if len(routed_slices) != routed_edge_count:
            raise ValueError(
                "weighted_combine requires one owner-local routed tensor slice per "
                f"routing edge; slices={len(routed_slices)}, edges={routed_edge_count}"
            )
        routed_slice_weights = [item.routing_weight for item in routed_slices]
        if any(weight is None for weight in routed_slice_weights):
            raise ValueError(
                "weighted_combine routed tensor slices must preserve routing_weight"
            )
        if sorted(float(weight) for weight in routed_slice_weights) != sorted(
            float(item["routing_weight"]) for item in edge_details
        ):
            raise ValueError(
                "weighted_combine routed tensor slice weights do not match "
                "routing_edges"
            )
        expected_inputs = routed_edge_count + len(shared_slices)
        if query.input_edge_count != expected_inputs:
            raise ValueError(
                "weighted_combine input_edge_count does not match routed/shared "
                f"partials; declared={query.input_edge_count}, actual={expected_inputs}"
            )
        if shared_slices and len(shared_slices) != tokens:
            raise ValueError(
                "weighted_combine Shared Expert inputs must contain one partial per "
                f"output token; shared={len(shared_slices)}, output_tokens={tokens}"
            )

        components: list[dict[str, Any]] = []
        weighted = self._element_component(
            "routing_weight_multiply",
            "routing_weight_multiply",
            query,
            routed_edge_count,
            local_hidden,
        )
        weighted["routing_edges"] = edge_details
        components.append(weighted)

        reduction_count = routed_edge_count - tokens
        if reduction_count < 0:
            raise ValueError(
                "weighted_combine has fewer routed edges than output tokens"
            )
        if reduction_count:
            reduced = self._element_component(
                "routed_reduce",
                "routed_reduce",
                query,
                reduction_count,
                local_hidden,
            )
            reduced["input_edge_count"] = routed_edge_count
            reduced["output_token_count"] = tokens
            components.append(reduced)
        if shared_slices:
            shared = self._element_component(
                "shared_sum",
                "shared_sum",
                query,
                len(shared_slices),
                local_hidden,
            )
            shared["shared_partial_ids"] = tuple(
                item.tensor_item_id for item in shared_slices
            )
            components.append(shared)
        return components

    def _conversion_component(
        self, query: AFComputeQuery, element_count: int
    ) -> dict[str, Any]:
        source_dtype = getattr(query, "source_dtype", None)
        target_dtype = getattr(query, "target_dtype", None)
        if source_dtype is None or target_dtype is None:
            raise ValueError(
                f"{query.stage} query requires explicit source_dtype and target_dtype"
            )
        source_dtype = canonicalize_dtype(source_dtype, role="activation_dtype")
        target_dtype = canonicalize_dtype(target_dtype, role="activation_dtype")
        if source_dtype == target_dtype:
            raise ValueError(
                f"{query.stage} is invalid when source_dtype equals "
                f"target_dtype={source_dtype!r}"
            )
        result = self.lookup.lookup(
            "elementwise",
            {
                "device": query.device_type,
                "backend": self.backend,
                "op_name": query.stage,
                "activation_dtype": query.activation_dtype or query.dtype,
                "source_dtype": source_dtype,
                "target_dtype": target_dtype,
                "num_tokens": 1,
                "hidden_size": element_count,
            },
        )
        component = self._component(query.stage, result)
        component["dtype_pair"] = {
            "source_dtype": source_dtype,
            "target_dtype": target_dtype,
        }
        component["element_count"] = element_count
        return component

    def _expert_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        global_hidden: int,
        tp_size: int,
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        common = {"device": device, "backend": self.backend}
        weight_dtype, activation_dtype, _ = _query_compute_dtypes(query, model)
        results: list[dict[str, Any]] = []
        element_names = {
            "residual_add": "f_residual_add",
            "data_reorder": "f_data_reorder",
        }
        if model.num_experts > 0:
            workload = self._validate_expert_workload(query, tokens, model.num_experts)
            results.append(
                self._moe_expert_component(
                    query,
                    model,
                    workload,
                    global_hidden,
                    tp_size,
                    common,
                )
            )
        else:
            local_intermediate = _exact_tp_shard(
                model.dense_intermediate_size,
                tp_size,
                "Dense FFN intermediate size",
            )
            results.extend(
                (
                    self._component(
                        "dense_gate_up",
                        self.lookup.lookup(
                            "gemm",
                            {
                                **common,
                                "weight_dtype": weight_dtype,
                                "activation_dtype": activation_dtype,
                                "m": tokens,
                                "n": 2 * local_intermediate,
                                "k": global_hidden,
                            },
                        ),
                    ),
                    self._element_component(
                        "dense_swiglu",
                        "swiglu",
                        query,
                        tokens,
                        local_intermediate,
                    ),
                    self._component(
                        "dense_down",
                        self.lookup.lookup(
                            "gemm",
                            {
                                **common,
                                "weight_dtype": weight_dtype,
                                "activation_dtype": activation_dtype,
                                "m": tokens,
                                "n": global_hidden,
                                "k": local_intermediate,
                            },
                        ),
                    ),
                )
            )
        for op in model.f_elementwise:
            if op in {"residual_add", "data_reorder"}:
                results.append(
                    self._element_component(
                        element_names[op], op, query, tokens, global_hidden
                    )
                )
        return results

    def _moe_expert_component(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        workload: tuple[tuple[int, int], ...],
        global_hidden: int,
        tp_size: int,
        common: Mapping[str, Any],
        *,
        operation: str = "routed_expert",
        component_name: str = "moe_expert",
        global_intermediate: int | None = None,
        ep_size: int | None = None,
    ) -> dict[str, Any]:
        weight_dtype, activation_dtype, _ = _query_compute_dtypes(query, model)
        intermediate = global_intermediate or model.routed_intermediate_size
        effective_ep_size = max(1, len(query.ep_group)) if ep_size is None else ep_size
        key = {
            **common,
            "operation": operation,
            "weight_dtype": weight_dtype,
            "activation_dtype": activation_dtype,
            "phase": query.phase,
            "distribution": model.distribution,
            "num_tokens": sum(count for _, count in workload),
            "hidden_size": global_hidden,
            "inter_size": intermediate,
            "top_k": 1,  # num_tokens already counts token-expert assignments.
            "num_experts": len(workload),
            "tp_size": tp_size,
            "ep_size": effective_ep_size,
        }
        table = "moe"
        if len(workload) > 1:
            table = "moe_group"
            key["expert_token_counts"] = encode_positive_vector(count for _, count in workload)
            key["execution_boundary"] = "unweighted_partials"
        result = self._component(component_name, self.lookup.lookup(table, key))
        result.update({
            "execution_unit": "expert_group" if len(workload) > 1 else "single_expert",
            "table": table,
            "token_semantics": "routed_assignments",
            "per_expert": [{"expert_id": expert, "num_tokens": count}
                           for expert, count in workload],
        })
        return result

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
                f"expert_workload contains expert id outside model range "
                f"[0, {num_experts})"
            )
        if tuple(query.expert_ids) != tuple(expert_ids):
            raise ValueError(
                f"expert_ids={tuple(query.expert_ids)!r} does not match "
                f"expert_workload ids={tuple(expert_ids)!r}"
            )
        workload_tokens = sum(count for _, count in workload)
        if workload_tokens != tokens:
            raise ValueError(
                f"expert_workload total={workload_tokens} does not match "
                f"local_token_count={tokens}"
            )
        return workload

    @staticmethod
    def _validate_stage_semantics(
        query: AFComputeQuery, model: ModelShape, tokens: int
    ) -> None:
        formal_stages = {
            "attention",
            "router",
            "dense_ffn",
            "routed_expert",
            "shared_expert",
            "weighted_combine",
            "residual",
            "encode",
            "decode",
            "layout",
        }
        if query.stage in {"a", "expert"} or (
            query.stage == "router" and query.layer_type is None
        ):
            return
        if query.stage in formal_stages and query.layer_type not in {"dense", "moe"}:
            raise ValueError(
                f"formal AF compute stage={query.stage!r} requires layer_type "
                "'dense' or 'moe'"
            )
        required_layer_type = {
            "router": "moe",
            "dense_ffn": "dense",
            "routed_expert": "moe",
            "shared_expert": "moe",
            "weighted_combine": "moe",
        }.get(query.stage)
        if required_layer_type is not None and query.layer_type != required_layer_type:
            raise ValueError(
                f"stage={query.stage!r} requires layer_type={required_layer_type!r}, "
                f"got {query.layer_type!r}"
            )
        if query.stage in {"router", "routed_expert", "shared_expert", "weighted_combine"}:
            if model.num_experts < 1:
                raise ValueError(
                    f"stage={query.stage!r} requires a model with Routed Experts"
                )

        if query.stage in {"attention", "dense_ffn", "shared_expert", "residual", "layout"}:
            _require_optional_count(query, "original_token_count", tokens)
        if query.stage == "router":
            _require_optional_count(query, "original_token_count", tokens)
            top_k = query.local_shape.get("top_k", model.top_k)
            if type(top_k) is not int or top_k < 1:
                raise ValueError("router local_shape top_k must be a positive integer")
            _require_optional_count(query, "assignment_count", tokens * top_k)
        if query.stage == "routed_expert":
            _require_optional_count(query, "assignment_count", tokens)
        if query.stage == "shared_expert":
            _require_optional_count(query, "output_token_count", tokens)
            if query.assignment_count not in {None, 0}:
                raise ValueError(
                    "Shared Expert workload is original tokens and must not carry "
                    "routed assignment_count"
                )
        if query.stage == "weighted_combine":
            _require_optional_count(query, "original_token_count", tokens)
            _require_optional_count(query, "output_token_count", tokens)
        if query.stage == "residual":
            _require_optional_count(query, "output_token_count", tokens)

    def _element_component(
        self,
        component_name: str,
        op_name: str,
        query: AFComputeQuery,
        token_count: int,
        hidden_size: int,
    ) -> dict[str, Any]:
        activation_dtype = canonicalize_dtype(
            query.activation_dtype or query.dtype,
            role="activation_dtype",
        )
        return self._component(
            component_name,
            self.lookup.lookup(
                "elementwise",
                {
                    "device": query.device_type,
                    "backend": self.backend,
                    "op_name": op_name,
                    "activation_dtype": activation_dtype,
                    "source_dtype": activation_dtype,
                    "target_dtype": activation_dtype,
                    "num_tokens": token_count,
                    "hidden_size": hidden_size,
                },
            ),
        )

    @staticmethod
    def _validate_local_work(
        query: AFComputeQuery, model: ModelShape
    ) -> tuple[int, int, int, int]:
        context = (
            f"model={model.model_id!r}, operation={query.operation!r}, "
            f"stage={query.stage!r}, "
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
        required = {"tokens"}
        if query.stage in {"encode", "decode"}:
            required.add("elements")
        elif query.stage == "router":
            required.add("hidden_size")
        elif query.stage in {"weighted_combine", "residual", "layout"}:
            required.add("hidden_size")
        else:
            required.update({"hidden_size", "global_hidden_size", "tp_size"})
        if query.stage == "attention":
            required.update(
                {
                    "num_heads",
                    "num_kv_heads",
                    "head_dim",
                    "query_width",
                    "kv_width",
                }
            )
        if query.stage in {"dense_ffn", "routed_expert", "shared_expert"}:
            required.add("intermediate_size")
        missing = sorted(required - set(shape))
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
        if query.stage in {"encode", "decode"}:
            if query.local_token_count != 1:
                raise ValueError(
                    f"AF {query.stage} query must describe one stable tensor item; "
                    f"local_token_count={query.local_token_count}; {context}"
                )
            return (query.local_token_count, values["elements"], model.hidden_size, 1)

        if query.stage == "router":
            if values["hidden_size"] != model.hidden_size:
                raise ValueError(
                    f"AF Router hidden_size={values['hidden_size']} does not match "
                    f"model hidden_size={model.hidden_size}; {context}"
                )
            return (query.local_token_count, model.hidden_size, model.hidden_size, 1)

        if query.stage in {"weighted_combine", "residual", "layout"}:
            tensor_width = _tensor_slice_width(query)
            if tensor_width is not None and values["hidden_size"] != tensor_width:
                raise ValueError(
                    f"AF {query.stage} local hidden_size={values['hidden_size']} does "
                    f"not match owned tensor slice width={tensor_width}; {context}"
                )
            return (
                query.local_token_count,
                tensor_width or values["hidden_size"],
                model.hidden_size,
                int(shape.get("tp_size", 1)),
            )

        if values["global_hidden_size"] != model.hidden_size:
            raise ValueError(
                f"AF compute query global_hidden_size={values['global_hidden_size']} "
                f"does not match model hidden_size={model.hidden_size}; {context}"
            )
        compute_tp_size = values["tp_size"]
        if query.stage == "attention":
            expected_tp_size = model.tp_size
            expected_num_heads = _exact_tp_shard(
                model.num_attention_heads,
                expected_tp_size,
                "Attention query heads",
            )
            expected_num_kv_heads = _local_kv_heads(
                model.num_key_value_heads,
                expected_tp_size,
            )
            expected_attention_shape = {
                "tp_size": expected_tp_size,
                "num_heads": expected_num_heads,
                "num_kv_heads": expected_num_kv_heads,
                "head_dim": model.head_dim,
                "query_width": expected_num_heads * model.head_dim,
                "kv_width": expected_num_kv_heads * model.head_dim,
            }
            if model.resolved_attention_type == "mla":
                expected_attention_shape = model.mla_dimensions.local_shape(
                    model.num_attention_heads, expected_tp_size)
            actual_attention_shape = {
                name: shape.get(name) for name in expected_attention_shape
            }
            if actual_attention_shape != expected_attention_shape:
                raise ValueError(
                    "AF Attention local shape does not match operator manifest; "
                    f"model={model.model_id!r}, operation={query.operation!r}, "
                    f"rank(dp,ep,tp)=({query.dp_rank},{query.ep_rank},"
                    f"{query.tp_rank}), expected={expected_attention_shape!r}, "
                    f"actual={actual_attention_shape!r}"
                )
            if query.tp_group and len(query.tp_group) != expected_tp_size:
                raise ValueError(
                    "AF Attention TP group does not match operator manifest; "
                    f"model={model.model_id!r}, operation={query.operation!r}, "
                    f"rank(dp,ep,tp)=({query.dp_rank},{query.ep_rank},"
                    f"{query.tp_rank}), expected_tp_size={expected_tp_size}, "
                    f"actual_tp_group={query.tp_group!r}"
                )
        is_routed_expert = query.stage in {"expert", "routed_expert"} and (
            model.num_experts > 0
        )
        if is_routed_expert:
            compute_tp_size = shape.get("expert_tp_size", values["tp_size"])
            if type(compute_tp_size) is not int or compute_tp_size < 1:
                raise ValueError(
                    "AF routed Expert query expert_tp_size must be a positive "
                    f"integer; local_shape={dict(shape)!r}; {context}"
                )
        mlp_stages = {"expert", "dense_ffn", "routed_expert", "shared_expert"}
        expected_hidden = model.hidden_size
        if query.stage == "a":
            expected_hidden = _exact_tp_shard(
                model.hidden_size,
                values["tp_size"],
                "hidden size",
            )
        if values["hidden_size"] != expected_hidden:
            raise ValueError(
                f"AF compute query local hidden_size={values['hidden_size']} does not "
                f"match expected={expected_hidden}; {context}"
            )
        if query.stage in mlp_stages and "intermediate_size" in shape:
            local_intermediate = shape["intermediate_size"]
            if query.stage == "shared_expert":
                if model.num_shared_experts < 1:
                    raise ValueError(
                        "AF Shared Expert query has no Shared Expert in the operator "
                        f"manifest; model={model.model_id!r}; {context}"
                    )
                expected_global_intermediate = model.shared_intermediate_size
                axis = "Shared Expert expanded intermediate size"
            else:
                expected_global_intermediate = (
                    model.routed_intermediate_size
                    if query.stage in {"expert", "routed_expert"}
                    and model.num_experts > 0
                    else model.dense_intermediate_size
                )
                axis = (
                    "Routed Expert intermediate size"
                    if query.stage in {"expert", "routed_expert"}
                    and model.num_experts > 0
                    else "Dense FFN intermediate size"
                )
            expected_intermediate = _exact_tp_shard(
                expected_global_intermediate,
                compute_tp_size,
                axis,
            )
            if local_intermediate != expected_intermediate:
                raise ValueError(
                    f"AF compute query local intermediate_size="
                    f"{local_intermediate} does not match expected="
                    f"{expected_intermediate} for model={model.model_id!r}, "
                    f"stage={query.stage!r}; {context}"
                )
            if type(local_intermediate) is not int or local_intermediate < 1:
                raise ValueError(
                    f"AF compute query local intermediate_size must be a positive "
                    f"integer; got {local_intermediate!r}; {context}"
                )
        return (
            query.local_token_count,
            values["hidden_size"],
            values["global_hidden_size"],
            compute_tp_size,
        )

    @staticmethod
    def _component(name: str, result: LookupResult) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": name,
            "key": dict(result.key),
            "latency_us": result.latency_us,
            "match_type": result.match_type,
            "source_id": result.source_id,
        }
        if result.interpolation is not None:
            value["interpolation"] = dict(result.interpolation)
        for field_name in ("timing_method", "calibration_applied", "fixed_overhead_us"):
            if field_name in result.values:
                value[field_name] = result.values[field_name]
        return value


class OperatorDataTransferLatencyProvider:
    def __init__(
        self,
        lookup: OperatorDataLookup,
        backend: str,
        link_group: str = "af-fabric",
        group_size: int = 2,
        metadata_bytes_per_token: int = 0,
    ) -> None:
        self.lookup = lookup
        self.backend = backend.strip().lower()
        self.link_group = link_group.strip().lower()
        self.group_size = group_size
        self.metadata_bytes_per_token = metadata_bytes_per_token

    def estimate(self, query: AFTransferQuery) -> AFServiceEstimate:
        group_size = query.group_size or self.group_size
        if type(group_size) is not int or group_size < 1:
            raise ValueError("transfer group_size must be a positive integer")
        collective_operations = {"all_reduce", "all_gather", "reduce_scatter"}
        supported = {"af_a_to_f", "af_f_to_a", "af_router_logits", "af_router_metadata", *collective_operations}
        if query.operation not in supported:
            raise ValueError(
                f"unsupported AF communication operation={query.operation!r}"
            )
        is_collective = query.operation in collective_operations
        if query.communication_kind is not None:
            expected_kind = "collective" if is_collective else "endpoint_leg"
            if query.communication_kind != expected_kind:
                raise ValueError(
                    f"operation={query.operation!r} requires communication_kind="
                    f"{expected_kind!r}, got {query.communication_kind!r}"
                )
        collective_details: dict[str, Any] = {}
        if is_collective:
            if not query.resource_lane:
                raise ValueError(
                    f"collective {query.operation!r} requires an explicit resource_lane"
                )
            payload_bytes, collective_details = _collective_wire_payload(
                query, group_size
            )
            # Direction describes the communication class; operation carries the
            # collective algorithm so records cannot conflate the two axes.
            key_direction = "collective"
        else:
            if query.payload_bytes is None:
                raise ValueError(
                    f"operation {query.operation!r} requires explicit payload_bytes"
                )
            if type(query.payload_bytes) is not int or query.payload_bytes < 0:
                raise ValueError(
                    "transfer payload_bytes must be a non-negative integer"
                )
            payload_bytes = query.payload_bytes
            if query.tensor_slices:
                slice_payload = sum(
                    item.payload_bytes or 0 for item in query.tensor_slices
                )
                if payload_bytes != slice_payload:
                    raise ValueError(
                        f"endpoint leg payload_bytes={payload_bytes} does not match "
                        f"tensor slice payload sum={slice_payload}"
                    )
            key_direction = query.direction
        transfer_dtype = _query_transfer_dtype(query)
        key = {
            "source_device": query.source_device,
            "target_device": query.target_device,
            "backend": self.backend,
            "network": query.network,
            "link_group": query.resource_group
            or query.link_group
            or self.link_group,
            "resource_lane": query.resource_lane or query.direction,
            "operation": query.operation,
            "direction": key_direction,
            "transfer_dtype": transfer_dtype,
            "source_endpoint": _communication_endpoint(query, source=True),
            "target_endpoint": _communication_endpoint(query, source=False),
            "group_size": group_size,
        }
        profile = self.lookup.link_profile(key)
        fixed_latency_us = float(profile.values["fixed_latency_us"])
        bandwidth = float(profile.values["effective_bandwidth_bytes_per_s"])
        transfer_us = fixed_latency_us + payload_bytes / bandwidth * 1e6
        return AFServiceEstimate(
            service_duration=transfer_us / 1e6,
            source_id=profile.source_id,
            payload_bytes=payload_bytes,
            dataset_version=self.lookup.package.dataset_version,
            match_type=profile.match_type,
            details={
                "dataset_version": self.lookup.package.dataset_version,
                "match_type": profile.match_type,
                "key": dict(profile.key),
                "fixed_latency_us": fixed_latency_us,
                "timing_scope": profile.values.get("timing_scope", "legacy_unspecified"),
                "effective_bandwidth_bytes_per_s": bandwidth,
                "payload_bytes": payload_bytes,
                **collective_details,
                "operation": query.operation,
                "direction": query.direction,
                "communication_kind": (
                    query.communication_kind
                    or ("collective" if is_collective else "endpoint_leg")
                ),
                "logical_stage_id": query.logical_stage_id,
                "leg_id": query.leg_id,
                "layer_type": query.layer_type,
                "original_token_count": query.original_token_count,
                "assignment_count": query.assignment_count,
                "transfer_dtype": transfer_dtype,
                "transfer_element_bytes": dtype_size_bytes(
                    transfer_dtype,
                    role=(
                        "a_to_f_dtype"
                        if query.direction == "a_to_f"
                        else "f_to_a_dtype"
                        if query.direction == "f_to_a"
                        else "activation_dtype"
                    ),
                ),
                "source_endpoint": _endpoint_details(query.source_endpoint),
                "target_endpoint": _endpoint_details(query.target_endpoint),
                "participant_endpoints": tuple(
                    _endpoint_details(item) for item in query.participant_endpoints
                ),
                "tensor_slices": _tensor_slice_details(query),
                "routing_edges": _routing_edge_details(query),
                "source_layout": query.source_layout,
                "target_layout": query.target_layout,
                "fabric": query.fabric,
                "resource_group": query.resource_group,
                "resource_lane": query.resource_lane,
                "logical_participants": query.logical_participants,
                "participant_ranks": query.participant_ranks,
                "physical_endpoints": query.physical_endpoints,
                "resource_identities": query.resource_identities,
                "topology": dict(query.topology),
                "total_latency_us": transfer_us,
            },
        )


def build_operator_data_providers(
    package_path: str | Path,
    model_manifest_path: str | Path,
    backend: str,
    link_group: str = "af-fabric",
    group_size: int = 2,
) -> tuple[OperatorDataComputeLatencyProvider, OperatorDataTransferLatencyProvider]:
    package = OperatorDataPackage.load(package_path)
    lookup = OperatorDataLookup(package)
    manifest = ModelOperatorManifest.load(model_manifest_path)
    return (
        OperatorDataComputeLatencyProvider(lookup, manifest, backend),
        OperatorDataTransferLatencyProvider(
            lookup,
            backend,
            link_group=link_group,
            group_size=group_size,
        ),
    )


def _collective_wire_payload(
    query: AFTransferQuery, group_size: int
) -> tuple[int, dict[str, Any]]:
    if group_size < 2:
        raise ValueError(
            f"collective {query.operation!r} requires group_size >= 2"
        )
    input_by_rank = _logical_bytes_by_rank(
        query.logical_input_bytes_per_rank,
        "logical_input_bytes_per_rank",
    )
    output_by_rank = _logical_bytes_by_rank(
        query.logical_output_bytes_per_rank,
        "logical_output_bytes_per_rank",
    )
    if set(input_by_rank) != set(output_by_rank):
        raise ValueError(
            f"collective {query.operation!r} logical input/output rank sets differ: "
            f"input={sorted(input_by_rank)}, output={sorted(output_by_rank)}"
        )
    if len(input_by_rank) != group_size:
        raise ValueError(
            f"collective {query.operation!r} group_size={group_size} does not "
            f"match logical rank count={len(input_by_rank)}"
        )
    if query.participant_ranks and set(query.participant_ranks) != set(input_by_rank):
        raise ValueError(
            f"collective {query.operation!r} participant ranks do not match "
            "logical byte ranks"
        )

    inputs = tuple(input_by_rank[rank] for rank in sorted(input_by_rank))
    outputs = tuple(output_by_rank[rank] for rank in sorted(output_by_rank))
    if query.operation == "all_reduce":
        if len(set(inputs)) != 1 or outputs != inputs:
            raise ValueError(
                "all_reduce requires equal per-rank logical input bytes and "
                "unchanged per-rank logical output bytes"
            )
        logical_reference_bytes = inputs[0]
        wire_bytes_float = (
            2.0 * (group_size - 1) / group_size * logical_reference_bytes
        )
        formula = "ring_all_reduce=2*(p-1)/p*logical_input_bytes"
    elif query.operation == "all_gather":
        gathered_bytes = sum(inputs)
        if any(value != gathered_bytes for value in outputs):
            raise ValueError(
                "all_gather requires each logical output to equal the sum of "
                "all per-rank logical inputs"
            )
        logical_reference_bytes = gathered_bytes
        wire_bytes_float = max(
            gathered_bytes - input_by_rank[rank] for rank in input_by_rank
        )
        formula = "ring_all_gather=sum(logical_inputs)-local_input"
    else:
        scattered_bytes = sum(outputs)
        if any(value != scattered_bytes for value in inputs):
            raise ValueError(
                "reduce_scatter requires each logical input to equal the sum of "
                "all per-rank logical outputs"
            )
        logical_reference_bytes = max(inputs)
        wire_bytes_float = max(
            input_by_rank[rank] - output_by_rank[rank] for rank in input_by_rank
        )
        formula = "ring_reduce_scatter=logical_input-local_output"

    wire_bytes = math.ceil(wire_bytes_float)
    factor = (
        wire_bytes_float / logical_reference_bytes
        if logical_reference_bytes > 0
        else 0.0
    )
    return wire_bytes, {
        "logical_input_bytes_per_rank": tuple(sorted(input_by_rank.items())),
        "logical_output_bytes_per_rank": tuple(sorted(output_by_rank.items())),
        "logical_input_total_bytes": sum(inputs),
        "logical_output_total_bytes": sum(outputs),
        "planner_payload_bytes": query.payload_bytes,
        "wire_bytes": wire_bytes,
        "wire_volume_factor": factor,
        "wire_volume_formula": formula,
    }


def _logical_bytes_by_rank(
    entries: tuple[tuple[int, int], ...], field_name: str
) -> dict[int, int]:
    if not entries:
        raise ValueError(f"collective requires non-empty {field_name}")
    values: dict[int, int] = {}
    for rank, byte_count in entries:
        if type(rank) is not int or rank < 0:
            raise ValueError(f"{field_name} ranks must be non-negative integers")
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError(f"{field_name} values must be non-negative integers")
        if rank in values:
            raise ValueError(f"{field_name} contains duplicate rank={rank}")
        values[rank] = byte_count
    return values


def _query_compute_dtypes(
    query: AFComputeQuery, model: ModelShape
) -> tuple[str, str, str]:
    return (
        canonicalize_dtype(
            query.weight_dtype or query.dtype, role="weight_dtype"
        ),
        canonicalize_dtype(
            query.activation_dtype or query.dtype, role="activation_dtype"
        ),
        canonicalize_dtype(
            query.kv_cache_dtype or model.kv_cache_dtype, role="kv_cache_dtype"
        ),
    )


def _query_transfer_dtype(query: AFTransferQuery) -> str:
    if query.direction == "a_to_f":
        return canonicalize_dtype(
            query.a_to_f_dtype or query.dtype, role="a_to_f_dtype"
        )
    if query.direction == "f_to_a":
        return canonicalize_dtype(
            query.f_to_a_dtype or query.dtype, role="f_to_a_dtype"
        )
    return canonicalize_dtype(
        query.activation_dtype or query.dtype, role="activation_dtype"
    )


def _positive_shape_value(
    query: AFComputeQuery, field_name: str, operation: str
) -> int:
    value = query.local_shape.get(field_name)
    if type(value) is not int or value < 1:
        raise ValueError(
            f"{operation} requires positive local_shape[{field_name!r}], "
            f"got {value!r}"
        )
    return value


def _require_optional_count(
    query: AFComputeQuery, field_name: str, expected: int
) -> None:
    actual = getattr(query, field_name)
    if actual is None:
        raise ValueError(
            f"formal AF stage={query.stage!r} requires {field_name}={expected}"
        )
    if actual != expected:
        raise ValueError(
            f"AF stage={query.stage!r} {field_name}={actual} does not match "
            f"expected={expected}"
        )


def _routing_edge_details(
    query: AFComputeQuery | AFTransferQuery,
) -> tuple[dict[str, Any], ...]:
    fields = (
        "iteration_id",
        "phase",
        "decode_step",
        "layer_id",
        "request_id",
        "token_id",
        "top_k_index",
        "expert_id",
        "routing_weight",
        "a_dp_rank",
        "a_tp_rank",
    )
    return tuple(
        {field: getattr(edge, field) for field in fields}
        for edge in tuple(getattr(query, "routing_edges", ()))
    )


def _tensor_slice_details(
    query: AFComputeQuery | AFTransferQuery,
) -> tuple[dict[str, Any], ...]:
    fields = (
        "tensor_item_id",
        "interval",
        "layout",
        "owner_domain",
        "owner_domain_rank",
        "owner_dp_rank",
        "owner_tp_rank",
        "owner_physical_worker",
        "partial_id",
        "canonical_producer",
        "element_count",
        "payload_bytes",
        "routing_weight",
    )
    return tuple(
        {
            field: getattr(tensor_slice, field)
            for field in fields
            if hasattr(tensor_slice, field)
        }
        for tensor_slice in query.tensor_slices
    )


def _tensor_slice_width(query: AFComputeQuery) -> int | None:
    if not query.tensor_slices:
        return None
    widths = {
        tensor_slice.interval[1] - tensor_slice.interval[0]
        for tensor_slice in query.tensor_slices
    }
    if len(widths) != 1:
        raise ValueError(
            f"AF stage={query.stage!r} tensor slices have inconsistent hidden "
            f"widths={sorted(widths)}"
        )
    return next(iter(widths))


def _endpoint_details(endpoint: object | None) -> Mapping[str, object] | None:
    if endpoint is None:
        return None
    return {
        "domain": getattr(endpoint, "domain"),
        "domain_rank": getattr(endpoint, "domain_rank"),
        "physical_worker": getattr(endpoint, "physical_worker"),
        "device_type": getattr(endpoint, "device_type"),
        "dp_rank": getattr(endpoint, "dp_rank"),
        "tp_rank": getattr(endpoint, "tp_rank"),
        "ep_rank": getattr(endpoint, "ep_rank"),
    }


def _communication_endpoint(query: AFTransferQuery, *, source: bool) -> str:
    if query.direction == "a_to_f":
        return "a" if source else "f"
    if query.direction == "f_to_a":
        return "f" if source else "a"

    endpoint = query.source_endpoint if source else query.target_endpoint
    domain = query.source_domain if source else query.target_domain
    device = query.source_device if source else query.target_device
    if endpoint is not None:
        domain = endpoint.domain
    value = str(domain or device).strip().lower()
    if value.startswith("a") or ":a" in value or "attention" in value:
        return "a"
    if value.startswith("f") or ":f" in value or "ffn" in value:
        return "f"
    return value


def _exact_tp_shard(global_size: int, tp_size: int, axis: str) -> int:
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError(f"{axis} tp_size must be a positive integer")
    if global_size % tp_size:
        raise ValueError(
            f"{axis}={global_size} is not divisible by tp_size={tp_size}; "
            "implicit uneven TP shards are unsupported"
        )
    return global_size // tp_size


def _local_kv_heads(num_kv_heads: int, tp_size: int) -> int:
    if num_kv_heads >= tp_size:
        return _exact_tp_shard(num_kv_heads, tp_size, "Attention KV heads")
    if tp_size % num_kv_heads:
        raise ValueError(
            f"Attention KV heads={num_kv_heads} cannot be replicated across "
            f"tp_size={tp_size}; tp_size must be divisible by KV heads"
        )
    return 1


def _validate_model(model: ModelShape) -> None:
    integer_fields = (
        "hidden_size",
        "dense_intermediate_size",
        "routed_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "top_k",
        "tp_size",
        "ep_size",
        "window_size",
    )
    for name in integer_fields:
        value = getattr(model, name)
        minimum = 0 if name == "window_size" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"model {model.model_id!r} {name} must be >= {minimum}")
    if type(model.num_experts) is not int or model.num_experts < 0:
        raise ValueError(f"model {model.model_id!r} num_experts must be non-negative")
    if type(model.num_shared_experts) is not int or model.num_shared_experts < 0:
        raise ValueError(
            f"model {model.model_id!r} num_shared_experts must be non-negative"
        )
    if model.num_experts and model.top_k > model.num_experts:
        raise ValueError(f"model {model.model_id!r} top_k exceeds num_experts")
    for name in ("kv_lora_rank", "q_lora_rank", "rope_head_dim", "nope_head_dim", "v_head_dim"):
        value = getattr(model, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"model {model.model_id!r} {name} must be a non-negative integer")
    for role in ("weight_dtype", "activation_dtype", "kv_cache_dtype"):
        canonicalize_dtype(getattr(model, role), role=role)
    _ = model.head_dim
    attention_type = model.resolved_attention_type
    if attention_type not in {"mha", "gqa", "mqa", "mla"}:
        raise ValueError(
            f"model {model.model_id!r} attention_type must be auto, mha, gqa, "
            f"mqa, or mla; got {attention_type!r}"
        )
    if attention_type == "mla":
        _ = model.mla_dimensions
        if model.window_size:
            raise ValueError("MLA window_size must be zero; sliding MLA is unsupported")
    if attention_type == "mha" and (
        model.num_key_value_heads != model.num_attention_heads
    ):
        raise ValueError(
            f"model {model.model_id!r} MHA requires num_key_value_heads to equal "
            "num_attention_heads"
        )
    if attention_type == "mqa" and model.num_key_value_heads != 1:
        raise ValueError(
            f"model {model.model_id!r} MQA requires num_key_value_heads=1"
        )
    if attention_type == "gqa" and not (
        1 < model.num_key_value_heads < model.num_attention_heads
    ):
        raise ValueError(
            f"model {model.model_id!r} GQA requires 1 < num_key_value_heads "
            "< num_attention_heads"
        )


def _validate_component_coverage(
    components: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, tuple[str, ...]],
) -> None:
    covered: dict[str, str] = {}
    for component in components:
        name = str(component["name"])
        includes = overrides.get(name, DEFAULT_INCLUDES.get(name, (name,)))
        if not includes:
            raise ValueError(
                f"component {name!r} must include at least one logical operation"
            )
        for logical_operation in includes:
            if logical_operation in covered:
                raise ValueError(
                    f"logical operation {logical_operation!r} is covered by both "
                    f"{covered[logical_operation]!r} and {name!r}"
                )
            covered[logical_operation] = name


def _validate_split_composition(model: ModelShape) -> None:
    router_names = [
        "f_rmsnorm" for operation in model.f_elementwise if operation == "rmsnorm"
    ]
    if model.num_experts > 0:
        router_names.append("router")
        expert_names = ["moe_expert"]
    else:
        expert_names = ["dense_gate_up", "dense_swiglu", "dense_down"]
    expert_names.extend(
        "f_residual_add" if operation == "residual_add" else "f_data_reorder"
        for operation in model.f_elementwise
        if operation in {"residual_add", "data_reorder"}
    )
    _validate_component_coverage(
        tuple({"name": name} for name in router_names + expert_names),
        model.component_includes,
    )
