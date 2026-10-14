"""Af config."""

from __future__ import annotations

from dataclasses import InitVar
from typing import Any

from pydantic import field_validator, model_validator
from pydantic.dataclasses import Field, dataclass

from TokenSim.config.af.common import (
    AF_LATENCY_FALLBACKS,
    AF_LATENCY_PROVIDERS,
    _AF_CONFIG_DTYPE_FIELDS,
    _NEW_DTYPE_FIELDS,
    _canonical_config_dtype,
    _capture_explicit_fields,
    _normalize_phases,
)
from TokenSim.config.af.communication import (
    AFCommunicationFabricConfig,
    AFCommunicationTimingConfig,
)
from TokenSim.config.af.domains import ExecutionDomainConfig
from TokenSim.config.af.links import AFEffectiveDTypes, AFLinkConfig
from TokenSim.config.constants import AF_PHASE_ORDER
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig


@dataclass
class AFConfig:
    enabled: bool = False
    phases: tuple[str, ...] = AF_PHASE_ORDER
    af_block_size: int = 1
    links: list[AFLinkConfig] = Field(default_factory=list)
    communication_fabrics: list[AFCommunicationFabricConfig] = Field(
        default_factory=list
    )
    execution_domains: list[ExecutionDomainConfig] = Field(default_factory=list)
    trace_enabled: bool = True
    # ``injected``: the caller supplies compute/transfer providers (tests).
    # ``operator_table``: the engine prices AF stages from the device operator
    # tables in ``data/operator_data`` and AF transfers from the link classes
    # in ``data/topologies/links.yaml``.
    latency_provider: str = "injected"
    # Operator-data backend for AF devices (defaults to the run-wide choice).
    operator_backend: str | None = None
    # ``table_first`` | ``table_only`` | ``analytical_only`` (defaults to the
    # run-wide ``--latency_fallback``).
    latency_fallback: str | None = None
    router_location: str = "a"
    routed_expert_execution: str = "grouped"
    transport_router_logits: bool = False
    af_dtype: str | None = None
    weight_dtype: str | None = None
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    dtype_presence_input: InitVar[tuple[str, ...]] = ()
    communication_timing: AFCommunicationTimingConfig = Field(
        default_factory=AFCommunicationTimingConfig
    )
    transfer_topology_mode: str = "physical"
    communication_domains: CommunicationDomainConfig | None = None
    service_estimate_cache: bool = True
    dag_template_cache: bool = True
    execution_protocol: str = "partial_return"
    operation_service_profile: dict[str, Any] | None = None
    transport_service_profile: dict[str, Any] | None = None
    engine_overhead_profile: dict[str, Any] | None = None
    calibration_environment: dict[str, str] = Field(default_factory=dict)
    routing_replay: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _capture_dtype_presence(cls, value: Any) -> Any:
        return _capture_explicit_fields(
            value,
            field_names=_AF_CONFIG_DTYPE_FIELDS,
            positional_names=(
                "enabled",
                "phases",
                "af_block_size",
                "links",
                "communication_fabrics",
                "execution_domains",
                "trace_enabled",
                "latency_provider",
                "operator_backend",
                "latency_fallback",
                "router_location",
                "routed_expert_execution",
                "transport_router_logits",
                "af_dtype",
                "weight_dtype",
                "activation_dtype",
                "kv_cache_dtype",
            ),
            init_var_name="dtype_presence_input",
        )

    @field_validator(
        "enabled",
        "trace_enabled",
        "service_estimate_cache",
        "dag_template_cache",
        mode="before",
    )
    @classmethod
    def _validate_boolean_type(cls, value: Any, info) -> bool:
        if type(value) is not bool:
            raise ConfigurationError(f"af.{info.field_name} must be a boolean")
        return value

    @field_validator("af_block_size", mode="before")
    @classmethod
    def _validate_block_size_type(cls, value: Any) -> int:
        if type(value) is not int:
            raise ConfigurationError("af_block_size must be a positive integer")
        return value

    def __post_init__(self, dtype_presence_input: tuple[str, ...]) -> None:
        self._explicit_dtype_fields = frozenset(dtype_presence_input)
        if self.transfer_topology_mode not in {"physical", "flat_link"}:
            raise ConfigurationError(
                "af.transfer_topology_mode must be physical or flat_link"
            )
        self.dtype_presence_input = tuple(sorted(self._explicit_dtype_fields))
        if type(self.enabled) is not bool:
            raise ConfigurationError("af.enabled must be a boolean")
        if type(self.trace_enabled) is not bool:
            raise ConfigurationError("af.trace_enabled must be a boolean")
        self.phases = _normalize_phases(self.phases, "af.phases")
        if self.execution_protocol not in {"partial_return", "vllm_afd"}:
            raise ConfigurationError(
                "af.execution_protocol must be partial_return or vllm_afd"
            )
        timing = self.communication_timing
        if timing.native_call_order_enabled and not timing.native_control_enabled:
            raise ConfigurationError(
                "native call order requires native_control_enabled=true"
            )
        if (
            self.engine_overhead_profile is not None
            or self.routing_replay is not None
            or self.operation_service_profile is not None
            or self.transport_service_profile is not None
        ):
            if (
                self.execution_protocol != "vllm_afd"
                or not timing.enabled
                or not timing.native_control_enabled
            ):
                raise ConfigurationError(
                    "native profile/replay requires vllm_afd with separated timing and native control"
                )
        control_costs = (
            timing.control_encode_us,
            timing.control_h2d_us,
            timing.control_length_readback_us,
            timing.control_payload_readback_us,
            timing.control_decode_us,
        )
        if any(control_costs) and not timing.native_control_enabled:
            raise ConfigurationError(
                "control costs require native_control_enabled=true"
            )
        if any(
            (
                timing.a_step_prepare_us,
                timing.f_step_prepare_us,
                timing.compute_submit_us,
                timing.host_receive_submit_us,
                timing.native_control_enabled,
            )
        ) and (not timing.enabled or self.execution_protocol != "vllm_afd"):
            raise ConfigurationError(
                "fixed host preparation requires vllm_afd and separated timing"
            )
        if self.execution_protocol == "vllm_afd":
            if set(self.phases) != {"decode"}:
                raise ConfigurationError(
                    "vllm_afd supports decode-only AF; prefill is not calibrated"
                )
            # The measured Qwen3 plugin owns gate, experts and combine on F.
            # Router placement is a protocol property, not a sweep parameter.
            self.router_location = "f"
            if self.transport_router_logits:
                raise ConfigurationError(
                    "vllm_afd does not transport A-side router logits"
                )
            if self.routed_expert_execution != "grouped":
                raise ConfigurationError("vllm_afd requires grouped expert execution")
            if not self.communication_timing.enabled:
                raise ConfigurationError(
                    "vllm_afd requires communication_timing.enabled=true"
                )
        if self.af_block_size < 1:
            raise ConfigurationError("af_block_size must be a positive integer")
        if self.latency_provider not in AF_LATENCY_PROVIDERS:
            raise ConfigurationError(
                "af.latency_provider must be one of " + ", ".join(AF_LATENCY_PROVIDERS)
            )
        if self.operator_backend is not None and (
            not isinstance(self.operator_backend, str)
            or not self.operator_backend.strip()
        ):
            raise ConfigurationError("af.operator_backend must be a non-empty string")
        if (
            self.latency_fallback is not None
            and self.latency_fallback not in AF_LATENCY_FALLBACKS
        ):
            raise ConfigurationError(
                "af.latency_fallback must be one of " + ", ".join(AF_LATENCY_FALLBACKS)
            )
        if self.router_location not in {"a", "f"}:
            raise ConfigurationError("af.router_location must be 'a' or 'f'")
        if self.routed_expert_execution not in {"grouped", "per_expert"}:
            raise ConfigurationError(
                "af.routed_expert_execution must be 'grouped' or 'per_expert'"
            )
        if type(self.transport_router_logits) is not bool:
            raise ConfigurationError("af.transport_router_logits must be a boolean")
        for field_name in self._explicit_dtype_fields:
            setattr(
                self,
                field_name,
                _canonical_config_dtype(getattr(self, field_name), field_name),
            )
        self.validate_dtype_style()

    @property
    def explicit_dtype_fields(self) -> frozenset[str]:
        return self._explicit_dtype_fields

    def validate_dtype_style(self) -> None:
        explicit_new = set(self._explicit_dtype_fields).intersection(_NEW_DTYPE_FIELDS)
        for link in self.links:
            explicit_new.update(link.explicit_dtype_fields)
        if "af_dtype" in self._explicit_dtype_fields and explicit_new:
            raise ConfigurationError(
                "legacy af_dtype cannot be combined with explicit dtype role fields; "
                f"remove af_dtype or {sorted(explicit_new)}"
            )

    def resolve_dtypes(
        self,
        *,
        model_dtype: str,
        link: AFLinkConfig | None = None,
        model_kv_cache_dtype: str | None = None,
    ) -> AFEffectiveDTypes:
        self.validate_dtype_style()
        model_value = _canonical_config_dtype(model_dtype, "activation_dtype")
        if "af_dtype" in self._explicit_dtype_fields:
            legacy = self.af_dtype
            return AFEffectiveDTypes(
                weight_dtype=legacy,
                activation_dtype=legacy,
                kv_cache_dtype=legacy,
                a_to_f_dtype=legacy,
                f_to_a_dtype=legacy,
                sources=tuple(
                    (role, "legacy:af_dtype") for role in sorted(_NEW_DTYPE_FIELDS)
                ),
                legacy_mapped=True,
                deprecated_fields=("af_dtype",),
            )

        weight = self.weight_dtype or model_value
        activation = self.activation_dtype or model_value
        if self.kv_cache_dtype is not None:
            kv_cache = self.kv_cache_dtype
            kv_source = "af.kv_cache_dtype"
        elif model_kv_cache_dtype is not None:
            kv_cache = _canonical_config_dtype(model_kv_cache_dtype, "kv_cache_dtype")
            kv_source = "model.kv_cache_dtype"
        else:
            kv_cache = activation
            kv_source = "af.activation_dtype"
        a_to_f = (
            link.a_to_f_dtype
            if link is not None and link.a_to_f_dtype is not None
            else activation
        )
        f_to_a = (
            link.f_to_a_dtype
            if link is not None and link.f_to_a_dtype is not None
            else activation
        )
        sources = {
            "weight_dtype": (
                "af.weight_dtype"
                if "weight_dtype" in self._explicit_dtype_fields
                else "model.dtype"
            ),
            "activation_dtype": (
                "af.activation_dtype"
                if "activation_dtype" in self._explicit_dtype_fields
                else "model.dtype"
            ),
            "kv_cache_dtype": kv_source,
            "a_to_f_dtype": (
                "af.link.a_to_f_dtype"
                if link is not None and "a_to_f_dtype" in link.explicit_dtype_fields
                else "af.activation_dtype"
            ),
            "f_to_a_dtype": (
                "af.link.f_to_a_dtype"
                if link is not None and "f_to_a_dtype" in link.explicit_dtype_fields
                else "af.activation_dtype"
            ),
        }
        return AFEffectiveDTypes(
            weight_dtype=weight,
            activation_dtype=activation,
            kv_cache_dtype=kv_cache,
            a_to_f_dtype=a_to_f,
            f_to_a_dtype=f_to_a,
            sources=tuple(sorted(sources.items())),
        )
