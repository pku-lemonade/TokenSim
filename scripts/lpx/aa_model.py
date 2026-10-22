"""Gemma text shapes and per-rank capacity for an isolated AA exploration.

This adapter does not add heterogeneous head dimensions to ModelSpec or claim
that the native AF executor supports Gemma. Runtime costs use TokenSim's
existing analytical operators; placement is an explicitly separate prototype.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.validation import boolean, integer, number
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import dtype_bytes


@dataclass(frozen=True)
class GemmaText:
    model_id: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    sliding_window: int
    full_attention_interval: int
    vocab_size: int
    max_position_embeddings: int
    global_k_equals_v_projection: bool
    tie_word_embeddings: bool

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ConfigurationError("Gemma model ID must be nonempty")
        for name, value in vars(self).items():
            if name.endswith("projection") or name == "tie_word_embeddings":
                boolean(value, name)
            elif name != "model_id":
                integer(value, name, minimum=1)
        if self.num_layers % self.full_attention_interval:
            raise ConfigurationError("full attention interval must divide layer count")
        for heads in (self.num_key_value_heads, self.num_global_key_value_heads):
            if self.num_attention_heads % heads:
                raise ConfigurationError("query heads must divide KV head groups")

    def full(self, layer):
        return (layer + 1) % self.full_attention_interval == 0

    def dims(self, full, tp):
        integer(tp, "attention TP", minimum=1)
        if self.num_attention_heads % tp:
            raise ConfigurationError("attention TP must divide query heads")
        heads = self.num_global_key_value_heads if full else self.num_key_value_heads
        dim = self.global_head_dim if full else self.head_dim
        return self.num_attention_heads // tp, local_kv_heads(heads, tp), dim

    def attention_elements(self, full, tp):
        q, kv, dim = self.dims(full, tp)
        projections = 1 if full and self.global_k_equals_v_projection else 2
        return self.hidden_size * dim * (2 * q + projections * kv)

    def ffn_elements(self, tp):
        if self.intermediate_size % tp:
            raise ConfigurationError("FFN TP must divide intermediate width")
        return 3 * self.hidden_size * (self.intermediate_size // tp)

    def kv_bytes(self, full, tp, context, dtype):
        _, heads, dim = self.dims(full, tp)
        keys = context if full else min(context, self.sliding_window)
        # Shared global K/V projections still produce distinct normalized,
        # rotary-transformed K and normalized V. Store both cache tensors.
        return 2 * keys * heads * dim * dtype_bytes(dtype)

    def parameter_elements(self):
        return sum(
            self.attention_elements(self.full(i), 1) + self.ffn_elements(1)
            for i in range(self.num_layers)
        ) + self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)


@dataclass(frozen=True)
class Placement:
    device: str
    tp: int
    part: str
    stages: tuple[tuple[int, int], ...]
    peak_rank_bytes: int
    head_on_first_stage: bool = False

    @property
    def cards(self):
        return self.tp * len(self.stages)


def pipeline_partition(
    layer_bytes, embedding_bytes, capacity_bytes, *, share_endpoints=False
):
    """Minimum contiguous PP stages, with first/last embedding allocation.

    Conventional PP endpoints need separate embedding/head copies.
    A serial client can return final hidden states to the embedding rank for
    its tied head, paying an activation return instead of a token return.
    Dedicated embedding/head stages may contain zero transformer layers.
    Returns None when a layer or endpoint cannot fit on one rank.
    """
    count = len(layer_bytes)
    prefix = [0]
    for size in layer_bytes:
        prefix.append(prefix[-1] + size)
    if prefix[-1] + embedding_bytes <= capacity_bytes:
        return ((0, count),), math.ceil(prefix[-1] + embedding_bytes)
    if embedding_bytes > capacity_bytes:
        return None
    choices = []
    for separate_first in (False, True) if embedding_bytes else (False,):
        for separate_last in (
            (False, True) if embedding_bytes and not share_endpoints else (False,)
        ):
            best = {0: ()}
            for end in range(1, count + 1):
                for start in range(end):
                    if start not in best:
                        continue
                    endpoints = embedding_bytes * (
                        (start == 0 and not separate_first)
                        + (end == count and not separate_last and not share_endpoints)
                    )
                    if prefix[end] - prefix[start] + endpoints > capacity_bytes:
                        continue
                    candidate = best[start] + ((start, end),)
                    if end not in best or len(candidate) < len(best[end]):
                        best[end] = candidate
            if count not in best:
                continue
            body = best[count]
            peak = max(
                prefix[end]
                - prefix[start]
                + embedding_bytes
                * (
                    (start == 0 and not separate_first)
                    + (end == count and not separate_last and not share_endpoints)
                )
                for start, end in body
            )
            stages = (((0, 0),) if separate_first else ()) + body
            stages += ((count, count),) if separate_last else ()
            choices.append((stages, math.ceil(max(peak, embedding_bytes))))
    return (
        min(choices, key=lambda choice: (len(choice[0]), choice[1]))
        if choices
        else None
    )


class Capacity:
    def __init__(self, hardware, model, *, weight_dtype, kv_dtype, reserve=0.05):
        number(reserve, "memory reserve", minimum=0, maximum=0.99)
        self.hardware, self.model = hardware, model
        self.weight_dtype, self.kv_dtype = weight_dtype, kv_dtype
        self.reserve = reserve

    @lru_cache(maxsize=None)
    def place(self, device, tp, part, context, chunk=256):
        for value, name in ((tp, "TP"), (context, "context"), (chunk, "prefill chunk")):
            integer(value, name, minimum=1)
        if part not in {"full", "attention", "ffn", "global_attention", "local_ffn"}:
            raise ConfigurationError("unknown Gemma placement component")
        model = self.model
        spec = self.hardware.device(device)
        weight_bytes = dtype_bytes(self.weight_dtype)
        # Workspace is separate from reserved SRAM/HBM and persistent tensors.
        # BF16 activations, fused attention without an O(context**2) buffer.
        workspace = (
            chunk
            * 2
            * (8 * model.hidden_size + 6 * math.ceil(model.intermediate_size / tp))
        )
        reserve = max(
            spec.memory_reserved_bytes.value,
            spec.memory_capacity_bytes.value * self.reserve,
        )
        available = spec.memory_capacity_bytes.value - reserve - workspace
        layers = []
        for i in range(model.num_layers):
            full = model.full(i)
            elements = 0
            cache = 0
            attention = (
                part in {"full", "attention"}
                or (part == "global_attention" and full)
                or (part == "local_ffn" and not full)
            )
            if attention:
                elements += model.attention_elements(full, tp)
                cache = model.kv_bytes(full, tp, context, self.kv_dtype)
                elements += 4 * model.hidden_size + 2 * (
                    model.global_head_dim if full else model.head_dim
                )
            if part in {"full", "ffn", "local_ffn"}:
                elements += model.ffn_elements(tp)
            layers.append(elements * weight_bytes + cache)
        embedding = (
            math.ceil(model.vocab_size / tp) * model.hidden_size * weight_bytes
            if part in {"full", "attention", "global_attention"}
            else 0
        )
        shared_head = bool(embedding and model.tie_word_embeddings)
        result = pipeline_partition(
            layers, embedding, available, share_endpoints=shared_head
        )
        if result is None:
            return None
        stages, peak = result
        return Placement(
            device, tp, part, stages, math.ceil(peak + reserve + workspace), shared_head
        )
