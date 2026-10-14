"""Shared MLA geometry; no latency policy or hardware assumptions."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MLADimensions:
    q_lora_rank: int
    kv_lora_rank: int
    rope_head_dim: int
    nope_head_dim: int
    v_head_dim: int

    @classmethod
    def from_model(cls, model):
        dims = cls(
            **{name: getattr(model, name, 0) for name in cls.__dataclass_fields__}
        )
        hidden = getattr(model, "hidden_size", getattr(model, "Dmodel", None))
        for name, value in asdict(dims).items():
            if type(value) is not int or value < 1 or (hidden and value > hidden):
                raise ValueError(
                    f"MLA {name} must be a positive integer <= hidden_size"
                )
        if dims.rope_head_dim % 2:
            raise ValueError("MLA rope_head_dim must be even")
        return dims

    @property
    def head_dim(self):
        return self.nope_head_dim + self.rope_head_dim

    @property
    def cache_width(self):
        return self.kv_lora_rank + self.rope_head_dim

    def local_shape(self, heads, tp):
        if heads % tp:
            raise ValueError("MLA query heads must divide TP evenly")
        return dict(
            asdict(self),
            tp_size=tp,
            num_heads=heads // tp,
            num_kv_heads=1,
            head_dim=self.head_dim,
            query_width=(heads // tp) * self.head_dim,
            kv_width=self.cache_width,
        )

    def weight_elements(self, local_heads, hidden):
        # Q/KV down projections and latent norm weights replicate over A TP.
        return (
            hidden * self.q_lora_rank
            + self.q_lora_rank * local_heads * self.head_dim
            + hidden * self.cache_width
            + self.kv_lora_rank * local_heads * (self.nope_head_dim + self.v_head_dim)
            + local_heads * self.v_head_dim * hidden
            + self.q_lora_rank
            + self.kv_lora_rank
        )
