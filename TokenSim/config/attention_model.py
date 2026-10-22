"""Model attention geometry for ordinary and latent KV caches."""

from __future__ import annotations

from TokenSim.errors import ConfigurationError
from TokenSim.mla import MLADimensions
from TokenSim.config.attention_geometry import AttentionGeometry
from TokenSim.config.validation import boolean


class AttentionModelMixin:
    def attention_geometry(self, layer_id):
        self.attention_window(layer_id)
        return (
            self.attention_geometries[layer_id]
            if self.attention_geometries
            else AttentionGeometry(self.head_dim, self.num_key_value_heads, self.k_eq_v)
        )

    def validate_attention_geometries(self):
        boolean(self.k_eq_v, "shared K/V projection")
        geometries = tuple(
            (
                value
                if isinstance(value, AttentionGeometry)
                else AttentionGeometry(**value)
            )
            for value in self.attention_geometries
        )
        if geometries and (
            len(geometries) != self.num_layers
            or self.attention_type == "mla"
            or self.components
        ):
            raise ConfigurationError(
                "attention_geometries require one ordinary geometry per layer"
            )
        if any(
            self.num_attention_heads % value.num_key_value_heads for value in geometries
        ):
            raise ConfigurationError("layer KV heads must divide query heads")
        object.__setattr__(self, "attention_geometries", geometries)

    def attention_params_per_layer(self, layer_id=None):
        if self.attention_type == "mla":
            return self.mla_dimensions.weight_elements(
                self.num_attention_heads, self.hidden_size
            )
        if self.components is not None:
            spec = self.components
            return (
                self.hidden_size * (spec.q_lora_rank + self.head_dim)
                + spec.q_lora_rank * self.q_dim
                + self.q_dim * spec.o_lora_rank
                + spec.o_groups * spec.o_lora_rank * self.hidden_size
            )
        geometry = (
            self.attention_geometry(layer_id)
            if layer_id is not None
            else AttentionGeometry(self.head_dim, self.num_key_value_heads, self.k_eq_v)
        )
        q = self.num_attention_heads * geometry.head_dim
        kv = self.kv_cache_dim or geometry.num_key_value_heads * geometry.head_dim
        return self.hidden_size * (2 * q + geometry.kv_projections * kv) + (
            2 * geometry.head_dim if self.qk_norm else 0
        )

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
