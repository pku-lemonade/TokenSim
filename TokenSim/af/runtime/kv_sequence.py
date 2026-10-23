"""KV-parallel token ownership and per-layer attention partitions.

A KV-parallel (KVP) group splits one attention layer's sequence positions
across ranks of the attention TP group.  Position ``p`` belongs to rank
``(p // block) % kvp``, so prompt and generated tokens rotate between ranks
in fixed-size blocks.  See docs/lpx-context-parallel.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from TokenSim.config.af.domains import KV_PARALLEL_LAYERS
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


def _check_group(kvp_size: int, kvp_rank: int, block: int) -> None:
    integer(kvp_size, "KV-parallel size", minimum=1)
    integer(block, "KV append block", minimum=1)
    integer(kvp_rank, "KV-parallel rank")
    if kvp_rank >= kvp_size:
        raise ConfigurationError("KV-parallel rank must be smaller than its group")


def token_owner(position: int, kvp_size: int, block: int) -> int:
    """Return the KVP rank that stores the KV entry at ``position``."""
    _check_group(kvp_size, 0, block)
    integer(position, "KV token position")
    return (position // block) % kvp_size


def owned_tokens(stop: int, kvp_size: int, kvp_rank: int, block: int) -> int:
    """Count positions in ``[0, stop)`` owned by ``kvp_rank``."""
    _check_group(kvp_size, kvp_rank, block)
    integer(stop, "KV token count")
    period = block * kvp_size
    remainder = stop % period - kvp_rank * block
    return stop // period * block + min(block, max(0, remainder))


def visible_tokens(
    context: int, window: int | None, kvp_size: int, kvp_rank: int, block: int
) -> int:
    """Count keys read by ``kvp_rank`` for a forward at KV length ``context``."""
    integer(context, "KV context")
    start = max(0, context - window) if window else 0
    return owned_tokens(context, kvp_size, kvp_rank, block) - owned_tokens(
        start, kvp_size, kvp_rank, block
    )


def busiest_visible_tokens(
    context: int, window: int | None, kvp_size: int, block: int
) -> int:
    """Keys read by the slowest rank of a KVP group, which bounds the step."""
    return max(
        visible_tokens(context, window, kvp_size, rank, block)
        for rank in range(kvp_size)
    )


def peak_resident_tokens(
    maximum_context: int, window: int | None, kvp_size: int, kvp_rank: int, block: int
) -> int:
    """Largest owned KV residency for any context up to ``maximum_context``."""
    integer(maximum_context, "maximum KV context")
    if not window or maximum_context <= window:
        return owned_tokens(maximum_context, kvp_size, kvp_rank, block)
    # Once the window is full its owned count repeats with one ownership period.
    last = min(maximum_context, window + block * kvp_size - 1)
    return max(
        visible_tokens(context, window, kvp_size, kvp_rank, block)
        for context in range(window, last + 1)
    )


def kvp_coordinates(tp_rank: int, kvp_size: int) -> tuple[int, int]:
    """Map a stage-local TP rank to ``(tpa_rank, kvp_rank)``.

    Consecutive TP ranks form one KVP group so its all-to-all stays local.
    """
    integer(tp_rank, "TP rank")
    integer(kvp_size, "KV-parallel size", minimum=1)
    return tp_rank // kvp_size, tp_rank % kvp_size


@dataclass(frozen=True)
class AttentionPartition:
    """How one layer's attention divides ``N`` stage-local TP ranks."""

    layer_id: int
    tensor_parallel_size: int
    kv_parallel_size: int
    attention_tensor_parallel_size: int
    query_heads_per_rank: int
    output_heads_per_rank: int
    kv_heads_per_rank: int
    kv_replica_factor: int

    @property
    def enabled(self) -> bool:
        return self.kv_parallel_size > 1

    def as_dict(self) -> dict[str, int]:
        return {
            "layer_id": self.layer_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "kv_parallel_size": self.kv_parallel_size,
            "attention_tensor_parallel_size": self.attention_tensor_parallel_size,
            "query_heads_per_rank": self.query_heads_per_rank,
            "output_heads_per_rank": self.output_heads_per_rank,
            "kv_heads_per_rank": self.kv_heads_per_rank,
            "kv_replica_factor": self.kv_replica_factor,
        }


def layer_kv_parallel_size(
    window: int | None, kv_parallel_size: int, kv_parallel_layers: str
) -> int:
    """Return the KVP applied to a layer with attention ``window``."""
    if kv_parallel_layers not in KV_PARALLEL_LAYERS:
        raise ConfigurationError(
            f"kv_parallel_layers must be one of {KV_PARALLEL_LAYERS}"
        )
    selected = kv_parallel_layers == "all" or not window
    return kv_parallel_size if selected else 1


def attention_partition(
    model,
    layer_id: int,
    tensor_parallel_size: int,
    kv_parallel_size: int = 1,
    kv_parallel_layers: str = "full_attention",
) -> AttentionPartition:
    """Return the attention head and sequence split for one layer."""
    tp = integer(tensor_parallel_size, "TP size", minimum=1)
    integer(kv_parallel_size, "KV-parallel size", minimum=1)
    if tp % kv_parallel_size:
        raise ConfigurationError("KV-parallel size must divide TP size")
    kvp = layer_kv_parallel_size(
        model.attention_window(layer_id), kv_parallel_size, kv_parallel_layers
    )
    tpa = tp // kvp
    geometry = model.attention_geometry(layer_id)
    heads, kv_heads = model.num_attention_heads, geometry.num_key_value_heads
    if kvp > 1 and heads % tp:
        raise ConfigurationError(
            f"layer {layer_id} KV parallelism needs {heads} query heads divisible "
            f"by TP={tp} for the combine all-to-all"
        )
    return AttentionPartition(
        layer_id=layer_id,
        tensor_parallel_size=tp,
        kv_parallel_size=kvp,
        attention_tensor_parallel_size=tpa,
        query_heads_per_rank=-(-heads // tpa),
        output_heads_per_rank=-(-heads // tp),
        kv_heads_per_rank=local_kv_heads(kv_heads, tpa),
        kv_replica_factor=max(1, tpa // kv_heads),
    )


def validate_kv_parallel_domain(domain, model, *, colocated: bool, mla: bool) -> None:
    """Reject KV-parallel domains outside the specified decode scope."""
    parallel = domain.parallel
    if parallel.kv_parallel_size == 1:
        return
    if mla or not colocated:
        raise ConfigurationError(
            f"execution domain {domain.name!r} KV parallelism requires colocated "
            "attention/FFN execution without MLA"
        )
    for layer_id in range(model.num_layers):
        attention_partition(
            model,
            layer_id,
            parallel.tensor_parallel_size,
            parallel.kv_parallel_size,
            parallel.kv_parallel_layers,
        )
