"""Exact output-channel ownership after context-parallel attention."""

from dataclasses import dataclass

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class AttentionOutputPartition:
    """Whole heads or equal channel fragments retained by one tensor rank."""

    heads: int
    head_dim: int
    full_head_dim: int

    @property
    def width(self) -> int:
        return self.heads * self.head_dim

    @property
    def fragmented(self) -> bool:
        return self.head_dim != self.full_head_dim

    def local_shape(self, tp_rank: int) -> dict[str, int]:
        integer(tp_rank, "attention output TP rank")
        shape = {"output_heads": self.heads}
        if self.fragmented:
            shape.update(
                output_head_dim=self.head_dim,
                output_channel_start=tp_rank * self.width,
                output_channel_stop=(tp_rank + 1) * self.width,
            )
        return shape


def attention_output_partition(
    heads: int, head_dim: int, tp: int, kvp: int
) -> AttentionOutputPartition:
    """Partition output channels while keeping Q/K dot products within heads.

    Context ranks compute complete heads over disjoint sequence positions.
    Their combine exchange can retain equal channel fragments of a head,
    duplicating that head's LSE scalar for each destination fragment.
    """
    for name, value in (
        ("query heads", heads),
        ("head dimension", head_dim),
        ("TP size", tp),
        ("KV-parallel size", kvp),
    ):
        integer(value, name, minimum=1)
    if tp % kvp:
        raise ConfigurationError("KV-parallel size must divide TP size")
    if heads % (tp // kvp):
        raise ConfigurationError(
            "attention query heads must divide evenly across attention TP ranks"
        )
    if heads % tp == 0:
        return AttentionOutputPartition(heads // tp, head_dim, head_dim)
    if kvp == 1 or tp % heads or head_dim % (tp // heads):
        raise ConfigurationError(
            "attention output needs whole heads or equal head-channel fragments "
            "with KV parallelism"
        )
    return AttentionOutputPartition(1, head_dim // (tp // heads), head_dim)
