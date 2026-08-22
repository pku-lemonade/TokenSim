from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic.dataclasses import dataclass

from TokenSim.errors import ConfigurationError


@dataclass
class RoutingConfig:
    distribution: str = "uniform"
    hot_experts: list[int] | None = None
    hot_expert_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.distribution not in {"uniform", "skew", "hot", "burst"}:
            raise ConfigurationError(
                "moe_routing_distribution must be one of "
                + "['uniform', 'skew', 'hot', 'burst']"
            )
        if not 0 <= self.hot_expert_fraction <= 1:
            raise ConfigurationError("moe_hot_expert_fraction must be in [0, 1]")
        if self.hot_experts is not None:
            self.hot_experts = [int(expert_id) for expert_id in self.hot_experts]

    @classmethod
    def from_value(cls, value: Any | None) -> "RoutingConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(distribution=value)
        if isinstance(value, dict):
            return cls(**value)
        raise ConfigurationError(f"invalid routing config {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MoEModelConfig:
    is_moe_model: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 1
    moe_intermediate_size: int | None = None
    num_shared_experts: int = 0
    num_moe_layers: int = 0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    interleave_moe_layer_step: int = 1
    routing: RoutingConfig | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None

    def __post_init__(self) -> None:
        self.routing = RoutingConfig.from_value(self.routing)
        self.is_moe_model = bool(self.is_moe_model)
        if not self.is_moe_model:
            self.num_experts = int(self.num_experts or 0)
            self.num_moe_layers = int(self.num_moe_layers or 0)
            self.num_experts_per_tok = int(self.num_experts_per_tok or 1)
            return

        self.num_experts = _positive_int(self.num_experts, "num_experts")
        self.num_experts_per_tok = _positive_int(
            self.num_experts_per_tok,
            "num_experts_per_tok",
        )
        if self.num_experts_per_tok > self.num_experts:
            raise ConfigurationError(
                "num_experts_per_tok cannot exceed num_experts"
            )
        self.num_moe_layers = _positive_int(self.num_moe_layers, "num_moe_layers")
        if self.moe_intermediate_size is not None:
            self.moe_intermediate_size = _positive_int(
                self.moe_intermediate_size,
                "moe_intermediate_size",
            )
        self.num_shared_experts = _non_negative_int(
            self.num_shared_experts,
            "num_shared_experts",
        )
        self.first_k_dense_replace = _non_negative_int(
            self.first_k_dense_replace,
            "first_k_dense_replace",
        )
        self.moe_layer_freq = _positive_int(self.moe_layer_freq, "moe_layer_freq")
        self.interleave_moe_layer_step = _positive_int(
            self.interleave_moe_layer_step,
            "interleave_moe_layer_step",
        )
        for field_name in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, _positive_int(value, field_name))
        self._validate_layer_layout()

    @property
    def enabled(self) -> bool:
        return self.is_moe_model and self.num_moe_layers > 0

    @classmethod
    def from_model_config(cls, model_config: Any | None) -> "MoEModelConfig":
        if model_config is None:
            return cls()
        existing = getattr(model_config, "moe_config", None)
        if isinstance(existing, cls):
            return existing
        return cls(
            is_moe_model=getattr(model_config, "is_moe_model", False),
            num_experts=getattr(model_config, "num_experts", 0),
            num_experts_per_tok=getattr(model_config, "num_experts_per_tok", 1),
            moe_intermediate_size=getattr(model_config, "moe_intermediate_size", None),
            num_shared_experts=getattr(model_config, "num_shared_experts", 0),
            num_moe_layers=getattr(model_config, "num_moe_layers", 0),
            first_k_dense_replace=getattr(model_config, "first_k_dense_replace", 0),
            moe_layer_freq=getattr(model_config, "moe_layer_freq", 1),
            interleave_moe_layer_step=getattr(
                model_config,
                "interleave_moe_layer_step",
                1,
            ),
            routing=getattr(model_config, "moe_routing", None),
            hidden_size=getattr(model_config, "hidden_size", None),
            intermediate_size=getattr(model_config, "intermediate_size", None),
            num_attention_heads=getattr(model_config, "num_attention_heads", None),
            num_key_value_heads=getattr(model_config, "num_key_value_heads", None),
        )

    def moe_layer_indices(self, total_layers: int | None = None) -> list[int]:
        if not self.enabled:
            return []
        start = self.first_k_dense_replace
        limit = total_layers if total_layers is not None else start + self.num_moe_layers
        indices = list(range(start, limit, self.moe_layer_freq))
        return indices[: self.num_moe_layers]

    def count_moe_layers_in_stage(self, stage_layer_indices: range) -> int:
        stage_set = set(stage_layer_indices)
        return sum(1 for layer_id in self.moe_layer_indices() if layer_id in stage_set)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["routing"] = self.routing.to_dict() if self.routing else RoutingConfig().to_dict()
        return data

    def _validate_layer_layout(self) -> None:
        if not self.enabled:
            return
        last_index = self.first_k_dense_replace + (
            self.num_moe_layers - 1
        ) * self.moe_layer_freq
        if last_index < 0:
            raise ConfigurationError("invalid MoE layer layout")


def _positive_int(value: Any, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be positive, got {value}")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    value = int(value)
    if value < 0:
        raise ConfigurationError(f"{field_name} must be non-negative, got {value}")
    return value
