"""MLA geometry and model overrides retained for AF configurations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.mla import MLADimensions
from TokenSim.moe.config import MoEModelConfig


class AFModelMixin:
    @property
    def mla_dimensions(self) -> MLADimensions:
        if self.attention_type != "mla":
            raise ConfigurationError(f"model {self.model_id!r} is not MLA")
        return MLADimensions.from_model(self)

    def validate_mla_tp(self, tp: int) -> None:
        if type(tp) is not int or tp < 1 or self.num_attention_heads % tp:
            raise ConfigurationError(
                f"model {self.model_id!r}: num_attention_heads={self.num_attention_heads} must divide TP={tp}"
            )

    def kv_elements_per_token(self) -> int:
        return self.kv_dim if self.attention_type == "mla" else 2 * self.kv_dim

    def with_moe_override(self, moe_config: MoEModelConfig | None) -> "ModelSpec":
        """Apply MoE metadata supplied by a PSLA file on top of the catalog entry."""
        if moe_config is None or not moe_config.enabled:
            return self
        updates: dict[str, Any] = {"moe": moe_config}
        if moe_config.hidden_size:
            updates["hidden_size"] = moe_config.hidden_size
        if moe_config.intermediate_size:
            updates["intermediate_size"] = moe_config.intermediate_size
        if moe_config.num_attention_heads:
            updates["num_attention_heads"] = moe_config.num_attention_heads
            heads = moe_config.num_attention_heads
            hidden = updates.get("hidden_size", self.hidden_size)
            if (
                self.attention_type != "mla"
                and hidden % heads == 0
                and self.head_dim * self.num_attention_heads != hidden
            ):
                updates["head_dim"] = hidden // heads
        if moe_config.num_key_value_heads:
            updates["num_key_value_heads"] = moe_config.num_key_value_heads
        return replace(self, **updates)

    @property
    def kv_dim(self) -> int:
        if self.attention_type == "mla":
            return self.mla_dimensions.cache_width
        return self.kv_cache_dim or self.num_key_value_heads * self.head_dim
