"""Layer-specific attention projections and cache geometry."""

from dataclasses import dataclass

from TokenSim.config.validation import boolean, integer


@dataclass(frozen=True)
class AttentionGeometry:
    head_dim: int
    num_key_value_heads: int
    k_eq_v: bool = False

    def __post_init__(self):
        integer(self.head_dim, "attention head dimension", minimum=1)
        integer(self.num_key_value_heads, "attention KV heads", minimum=1)
        boolean(self.k_eq_v, "shared K/V projection")

    @property
    def kv_projections(self):
        # Sharing the projection does not share the transformed K/V caches.
        return 1 if self.k_eq_v else 2
