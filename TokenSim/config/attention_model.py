"""Model attention geometry for ordinary and latent KV caches."""

from __future__ import annotations

from TokenSim.errors import ConfigurationError
from TokenSim.mla import MLADimensions


class AttentionModelMixin:
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

    @property
    def kv_dim(self) -> int:
        if self.attention_type == "mla":
            return self.mla_dimensions.cache_width
        return self.kv_cache_dim or self.num_key_value_heads * self.head_dim
