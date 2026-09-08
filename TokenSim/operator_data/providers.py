from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from TokenSim.af.latency import AFComputeQuery, AFServiceEstimate, AFTransferQuery
from TokenSim.operator_data.lookup import (
    LookupResult,
    MissingOperatorDataError,
    OperatorDataLookup,
)
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
        tokens, local_hidden, global_hidden, tp_size = self._validate_local_work(
            query, model
        )
        _validate_split_composition(model)
        try:
            if query.stage == "a":
                components = self._attention_components(
                    query, model, tokens, local_hidden, global_hidden, tp_size
                )
            elif query.stage == "router":
                components = self._router_components(
                    query, model, tokens, global_hidden
                )
            elif query.stage == "expert":
                components = self._expert_components(
                    query, model, tokens, global_hidden, tp_size
                )
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
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
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
                        "gemm_dtype": query.dtype,
                        "m": tokens,
                        "n": math.ceil(
                            (model.q_dimension + 2 * model.kv_dimension) / tp_size
                        ),
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
                        "num_heads": math.ceil(model.num_attention_heads / tp_size),
                        "num_kv_heads": math.ceil(model.num_key_value_heads / tp_size),
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
                        "n": global_hidden,
                        "k": math.ceil(model.q_dimension / tp_size),
                    },
                ),
            )
        )
        for op in model.a_elementwise:
            if op in {"residual_add", "data_reorder"}:
                results.append(
                    self._element_component(
                        element_names[op], op, query, tokens, local_hidden
                    )
                )
        return results

    def _router_components(
        self,
        query: AFComputeQuery,
        model: ModelShape,
        tokens: int,
        global_hidden: int,
    ) -> list[dict[str, Any]]:
        device = query.device_type.strip().lower()
        common = {"device": device, "backend": self.backend}
        results: list[dict[str, Any]] = []
        for op in model.f_elementwise:
            if op == "rmsnorm":
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
                            "gemm_dtype": query.dtype,
                            "m": tokens,
                            "n": model.num_experts,
                            "k": global_hidden,
                        },
                    ),
                )
            )
        return results

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
            local_intermediate = math.ceil(model.intermediate_size / tp_size)
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
                                "gemm_dtype": query.dtype,
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
    ) -> dict[str, Any]:
        per_expert = []
        for expert_id, count in workload:
            key = {
                **common,
                "moe_dtype": query.dtype,
                "phase": query.phase,
                "distribution": model.distribution,
                "num_tokens": count,
                "hidden_size": global_hidden,
                "inter_size": model.intermediate_size,
                # Top-K has already expanded source tokens into this workload.
                "top_k": 1,
                "num_experts": 1,
                "tp_size": tp_size,
                "ep_size": max(1, len(query.ep_group)),
            }
            result = self.lookup.lookup("moe", key)
            per_expert.append(
                {
                    "expert_id": expert_id,
                    "num_tokens": count,
                    "key": dict(result.key),
                    "latency_us": result.latency_us,
                    "source_id": result.source_id,
                    "match_type": result.match_type,
                    "interpolation": result.interpolation,
                }
            )
        match_type = (
            "interpolated"
            if any(item["match_type"] == "interpolated" for item in per_expert)
            else "exact"
        )
        return {
            "name": "moe_expert",
            "key": {
                **common,
                "moe_dtype": query.dtype,
                "phase": query.phase,
                "distribution": model.distribution,
                "num_tokens": sum(count for _, count in workload),
                "hidden_size": global_hidden,
                "inter_size": model.intermediate_size,
                "top_k": 1,
                "num_experts": len(workload),
                "tp_size": tp_size,
                "ep_size": max(1, len(query.ep_group)),
                "token_semantics": "routed_assignments",
            },
            "latency_us": sum(float(item["latency_us"]) for item in per_expert),
            "source_id": "+".join(str(item["source_id"]) for item in per_expert),
            "match_type": match_type,
            "per_expert": per_expert,
        }

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

    def _element_component(
        self,
        component_name: str,
        op_name: str,
        query: AFComputeQuery,
        token_count: int,
        hidden_size: int,
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
        required = {"tokens", "hidden_size", "global_hidden_size", "tp_size"}
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
        if values["global_hidden_size"] != model.hidden_size:
            raise ValueError(
                f"AF compute query global_hidden_size={values['global_hidden_size']} "
                f"does not match model hidden_size={model.hidden_size}; {context}"
            )
        expected_hidden = (
            model.hidden_size
            if query.stage == "router"
            else math.ceil(model.hidden_size / values["tp_size"])
        )
        if values["hidden_size"] != expected_hidden:
            raise ValueError(
                f"AF compute query local hidden_size={values['hidden_size']} does not "
                f"match expected={expected_hidden}; {context}"
            )
        return (
            query.local_token_count,
            values["hidden_size"],
            values["global_hidden_size"],
            values["tp_size"],
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
        group_size = query.group_size or self.group_size
        if type(group_size) is not int or group_size < 1:
            raise ValueError("transfer group_size must be a positive integer")
        supported = {
            "af_a_to_f",
            "af_f_to_a",
            "tp_collective",
            "ep_dispatch",
            "ep_combine",
        }
        if query.operation not in supported:
            raise ValueError(
                f"unsupported AF communication operation={query.operation!r}"
            )
        if query.payload_bytes is None:
            raise ValueError(
                f"operation {query.operation!r} requires explicit payload_bytes"
            )
        if type(query.payload_bytes) is not int or query.payload_bytes < 0:
            raise ValueError("transfer payload_bytes must be a non-negative integer")
        payload_bytes = query.payload_bytes
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
                "operation": query.operation,
                "direction": query.direction,
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
