from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from TokenSim.af.latency import AFComputeQuery, AFServiceEstimate, AFTransferQuery
from TokenSim.operator_data.analytical import DTYPE_BYTES
from TokenSim.operator_data.lookup import LookupResult, OperatorDataLookup
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class ModelShape:
    model_id: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int = 0
    top_k: int = 1
    tp_size: int = 1
    ep_size: int = 1
    window_size: int = 0
    kv_cache_dtype: str = "bf16"
    distribution: str = "balanced"
    a_elementwise: tuple[str, ...] = (
        "rmsnorm",
        "rope",
        "residual_add",
    )
    f_elementwise: tuple[str, ...] = ("rmsnorm", "residual_add")
    component_includes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
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
        return self.num_key_value_heads * self.head_dim


class ModelOperatorManifest:
    def __init__(self, models: Mapping[str, ModelShape]) -> None:
        self._models = dict(models)
        if not self._models:
            raise ValueError("model operator manifest must contain at least one model")

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
                f"unsupported model operator schema_version="
                f"{document.get('schema_version')!r}"
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
            values = dict(raw)
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


DEFAULT_INCLUDES: Mapping[str, tuple[str, ...]] = {
    "a_rmsnorm": ("a_rmsnorm",),
    "qkv_projection": ("qkv_projection",),
    "rope": ("rope",),
    "attention_core": ("attention_core", "softmax"),
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

    def estimate(self, query: AFComputeQuery) -> AFServiceEstimate:
        model = self.manifest.model(query.model_id)
        if query.token_count < 1:
            raise ValueError("AF compute query token_count must be positive")
        if query.stage == "a":
            components = self._attention_components(query, model)
        else:
            components = self._ffn_components(query, model)
        _validate_component_coverage(components, model.component_includes)
        latency_us = sum(float(component["latency_us"]) for component in components)
        match_type = (
            "interpolated"
            if any(item["match_type"] == "interpolated" for item in components)
            else "exact"
        )
        source_ids = tuple(dict.fromkeys(item["source_id"] for item in components))
        source_id = (
            source_ids[0] if len(source_ids) == 1 else "compose:" + ":".join(source_ids)
        )
        return AFServiceEstimate(
            service_duration=latency_us / 1e6,
            source_id=source_id,
            dataset_version=self.lookup.package.dataset_version,
            match_type=match_type,
            details={
                "dataset_version": self.lookup.package.dataset_version,
                "match_type": match_type,
                "components": components,
                "total_latency_us": latency_us,
            },
        )

    def _attention_components(
        self, query: AFComputeQuery, model: ModelShape
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        tokens = query.token_count
        common = {"device": device, "backend": self.backend}
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
                    self._element_component(element_names[op], op, query, model)
                )
        results.append(
            self._component(
                "qkv_projection",
                self.lookup.lookup(
                    "gemm",
                    {
                        **common,
                        "gemm_dtype": query.dtype,
                        "m": tokens,
                        "n": model.q_dimension + 2 * model.kv_dimension,
                        "k": model.hidden_size,
                    },
                ),
            )
        )
        for op in model.a_elementwise:
            if op == "rope":
                results.append(
                    self._element_component(element_names[op], op, query, model)
                )
        batch_size = max(1, len(query.context_lengths))
        if query.phase == "prefill":
            seq_len = max(1, math.ceil(tokens / batch_size))
            table_name = "context_attention"
            length_key = {"input_seq_len": seq_len}
        else:
            context_len = max(query.context_lengths, default=1)
            table_name = "generation_attention"
            length_key = {"context_len": context_len}
        results.append(
            self._component(
                "attention_core",
                self.lookup.lookup(
                    table_name,
                    {
                        **common,
                        "attn_dtype": query.dtype,
                        "kv_cache_dtype": model.kv_cache_dtype,
                        "batch_size": batch_size,
                        **length_key,
                        "num_heads": model.num_attention_heads,
                        "num_kv_heads": model.num_key_value_heads,
                        "head_dim": model.head_dim,
                        "window_size": model.window_size,
                    },
                ),
            )
        )
        results.append(
            self._component(
                "output_projection",
                self.lookup.lookup(
                    "gemm",
                    {
                        **common,
                        "gemm_dtype": query.dtype,
                        "m": tokens,
                        "n": model.hidden_size,
                        "k": model.q_dimension,
                    },
                ),
            )
        )
        for op in model.a_elementwise:
            if op in {"residual_add", "data_reorder"}:
                results.append(
                    self._element_component(element_names[op], op, query, model)
                )
        return results

    def _ffn_components(
        self, query: AFComputeQuery, model: ModelShape
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        tokens = query.token_count
        common = {"device": device, "backend": self.backend}
        results: list[dict[str, Any]] = []
        element_names = {
            "rmsnorm": "f_rmsnorm",
            "residual_add": "f_residual_add",
            "data_reorder": "f_data_reorder",
        }
        for op in model.f_elementwise:
            if op == "rmsnorm":
                results.append(
                    self._element_component(element_names[op], op, query, model)
                )
        if model.num_experts > 0:
            results.append(
                self._component(
                    "router",
                    self.lookup.lookup(
                        "gemm",
                        {
                            **common,
                            "gemm_dtype": query.dtype,
                            "m": tokens,
                            "n": model.num_experts,
                            "k": model.hidden_size,
                        },
                    ),
                )
            )
            results.append(
                self._component(
                    "moe_expert",
                    self.lookup.lookup(
                        "moe",
                        {
                            **common,
                            "moe_dtype": query.dtype,
                            "phase": query.phase,
                            "distribution": model.distribution,
                            "num_tokens": tokens,
                            "hidden_size": model.hidden_size,
                            "inter_size": model.intermediate_size,
                            "top_k": model.top_k,
                            "num_experts": model.num_experts,
                            "tp_size": model.tp_size,
                            "ep_size": model.ep_size,
                        },
                    ),
                )
            )
        else:
            results.extend(
                (
                    self._component(
                        "dense_gate_up",
                        self.lookup.lookup(
                            "gemm",
                            {
                                **common,
                                "gemm_dtype": query.dtype,
                                "m": tokens,
                                "n": 2 * model.intermediate_size,
                                "k": model.hidden_size,
                            },
                        ),
                    ),
                    self._element_component("dense_swiglu", "swiglu", query, model),
                    self._component(
                        "dense_down",
                        self.lookup.lookup(
                            "gemm",
                            {
                                **common,
                                "gemm_dtype": query.dtype,
                                "m": tokens,
                                "n": model.hidden_size,
                                "k": model.intermediate_size,
                            },
                        ),
                    ),
                )
            )
        for op in model.f_elementwise:
            if op in {"residual_add", "data_reorder"}:
                results.append(
                    self._element_component(element_names[op], op, query, model)
                )
        return results

    def _element_component(
        self,
        component_name: str,
        op_name: str,
        query: AFComputeQuery,
        model: ModelShape,
    ) -> dict[str, Any]:
        return self._component(
            component_name,
            self.lookup.lookup(
                "elementwise",
                {
                    "device": query.device_type,
                    "backend": self.backend,
                    "op_name": op_name,
                    "dtype": query.dtype,
                    "num_tokens": query.token_count,
                    "hidden_size": model.hidden_size,
                },
            ),
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
        try:
            element_bytes = DTYPE_BYTES[query.dtype.strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported transfer dtype={query.dtype!r}") from exc
        group_size = query.group_size or self.group_size
        if type(group_size) is not int or group_size < 1:
            raise ValueError("transfer group_size must be a positive integer")
        if type(query.top_k) is not int or query.top_k < 1:
            raise ValueError("transfer top_k must be a positive integer")
        activation_bytes = query.token_count * query.hidden_size * element_bytes
        if query.operation == "af_transfer":
            expected_payload = activation_bytes + (
                query.token_count * self.metadata_bytes_per_token
            )
        elif query.operation == "all_reduce":
            expected_payload = math.ceil(
                2 * (group_size - 1) / group_size * activation_bytes
            )
        elif query.operation in {"moe_dispatch", "moe_combine"}:
            expected_payload = query.top_k * (
                activation_bytes + query.token_count * self.metadata_bytes_per_token
            )
        elif query.payload_bytes is None:
            raise ValueError(
                f"operation {query.operation!r} requires explicit payload_bytes"
            )
        else:
            expected_payload = query.payload_bytes
        if query.payload_bytes is not None and query.payload_bytes != expected_payload:
            raise ValueError(
                f"transfer payload_bytes={query.payload_bytes} does not match "
                f"computed payload={expected_payload}"
            )
        payload_bytes = expected_payload
        key = {
            "source_device": query.source_device,
            "target_device": query.target_device,
            "backend": self.backend,
            "network": query.network,
            "link_group": query.link_group or self.link_group,
            "operation": query.operation,
            "direction": query.direction,
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
                "effective_bandwidth_bytes_per_s": bandwidth,
                "payload_bytes": payload_bytes,
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


def _validate_model(model: ModelShape) -> None:
    integer_fields = (
        "hidden_size",
        "intermediate_size",
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
    if model.num_experts and model.top_k > model.num_experts:
        raise ValueError(f"model {model.model_id!r} top_k exceeds num_experts")
    _ = model.head_dim


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
