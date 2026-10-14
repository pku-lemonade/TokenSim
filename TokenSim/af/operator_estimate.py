"""Operator estimate."""

from __future__ import annotations

import math

from TokenSim.af.estimate_cache import cached_estimate
from TokenSim.af.latency import AFComputeQuery, AFServiceEstimate
from TokenSim.af.operator_composition import (
    _compose_dataset_version,
    _compose_source_id,
    _worst_match,
)
from TokenSim.comm.traffic import validated_bytes
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype


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
