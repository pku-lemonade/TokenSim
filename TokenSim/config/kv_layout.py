"""Native compressed attention pages and packed groups derived from vLLM 30f5c01d4ebb.

All groups alias one physical pool. Group token widths are independent of
the common physical stride; KV has one latent vector, not K and V copies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import lcm

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import dtype_bytes


def cdiv(value, divisor):
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class CacheLayerPage:
    name: str
    kind: str
    block_tokens: int
    state_bytes: int
    tokens_per_state: int = 1
    window_tokens: int = 0
    alignment: int = 512
    draft: bool = False

    @property
    def payload_bytes(self):
        return self.block_tokens // self.tokens_per_state * self.state_bytes

    @property
    def page_bytes(self):
        return cdiv(self.payload_bytes, self.alignment) * self.alignment

    @property
    def uniform_key(self):
        # UniformTypeKVCacheSpecs compares token-block width, the cache type,
        # and SWA window/replay policy. Page bytes/dtype may differ in a bucket.
        return self.kind, self.block_tokens, self.window_tokens


@dataclass(frozen=True)
class CacheGroup:
    group_id: int
    layers: tuple[CacheLayerPage, ...]
    eagle: bool = False

    @property
    def block_tokens(self):
        return self.layers[0].block_tokens

    @property
    def window_tokens(self):
        return self.layers[0].window_tokens

    @property
    def kind(self):
        return self.layers[0].kind

    @property
    def group_bytes(self):
        return sum(page.page_bytes for page in self.layers)

    def max_request_slots(self, max_context, in_flight_tokens):
        if not self.window_tokens:
            return cdiv(max_context, self.block_tokens)
        return (
            cdiv(
                min(max_context, self.window_tokens - 1 + in_flight_tokens),
                self.block_tokens,
            )
            + 1
        )

    @property
    def hit_tail_slots(self):
        return (
            cdiv(self.window_tokens - 1, self.block_tokens) + int(self.eagle)
            if self.window_tokens
            else 0
        )


@dataclass(frozen=True)
class HybridKVLayout:
    groups: tuple[CacheGroup, ...]
    logical_block_tokens: int
    physical_slot_bytes: int
    retention_interval: int
    speculative_tokens: int
    max_in_flight_tokens: int
    max_model_len: int

    @property
    def full_group(self):
        return next(group for group in self.groups if group.kind == "full")

    def max_request_slots(self, context_tokens=None):
        length = self.max_model_len if context_tokens is None else context_tokens
        return sum(
            group.max_request_slots(length, self.max_in_flight_tokens)
            for group in self.groups
        )

    def equivalent_tokens(self, pool_slots):
        # This reproduces the group-aware log equivalent, not arbitrary
        # tokens of capacity. Runtime uses group slots, never this number.
        return pool_slots * self.max_model_len // self.max_request_slots()

    def to_dict(self):
        return {
            "kind": "grouped-packed-shared-pool",
            "logical_block_tokens": self.logical_block_tokens,
            "physical_slot_bytes": self.physical_slot_bytes,
            "retention_interval": self.retention_interval,
            "speculative_tokens": self.speculative_tokens,
            "max_in_flight_tokens": self.max_in_flight_tokens,
            "max_model_len": self.max_model_len,
            "max_request_slots": self.max_request_slots(),
            "groups": [
                {
                    "group_id": group.group_id,
                    "kind": group.kind,
                    "block_tokens": group.block_tokens,
                    "window_tokens": group.window_tokens,
                    "eagle": group.eagle,
                    "group_bytes": group.group_bytes,
                    "slot_padding_bytes": self.physical_slot_bytes - group.group_bytes,
                    "max_request_slots": group.max_request_slots(
                        self.max_model_len, self.max_in_flight_tokens
                    ),
                    "layers": [
                        {
                            **asdict(layer),
                            "payload_bytes": layer.payload_bytes,
                            "page_bytes": layer.page_bytes,
                        }
                        for layer in group.layers
                    ],
                }
                for group in self.groups
            ],
        }


def approximate_group_repeats(values, minimum):
    return min(
        range(minimum, max(values) + 1),
        key=lambda size: (sum((size - value % size) % size for value in values), -size),
    )


def hybrid_kv_layout(
    model,
    *,
    block_size=256,
    max_num_batched_tokens=8192,
    max_concurrent_batches=2,
    retention_interval=32768,
    speculative_tokens=3,
    include_mtp=True,
):
    spec = model.components
    if spec is None:
        raise ConfigurationError(
            "native compressed attention layout needs complete structural metadata"
        )
    if (
        block_size <= 0
        or block_size % 128
        or max_num_batched_tokens <= 0
        or max_concurrent_batches <= 0
    ):
        raise ConfigurationError(
            "compressed attention logical blocks must be a positive multiple of 128; token budgets must be positive"
        )
    if retention_interval < 0 or retention_interval % block_size:
        raise ConfigurationError(
            "retention must be zero or a multiple of the scheduler block"
        )
    if speculative_tokens < 0 or speculative_tokens > block_size:
        raise ConfigurationError("invalid compressed attention speculative lookahead")
    kv_state = int(model.head_dim * dtype_bytes(model.kv_cache_dtype))
    index_state = (
        spec.index_head_dim // 2
        + cdiv(spec.index_head_dim, spec.expert_scale_group_size)
        if spec.indexer_cache_dtype == "mxfp4"
        else spec.index_head_dim + 4
    )
    pages = []
    for layer in model.layer_specs(include_mtp=include_mtp):
        prefix = f"{'mtp' if layer.is_mtp else 'layers'}.{layer.layer_id}"
        # Attention registration order: CSA indexer is constructed first,
        # then SWA, full compressed cache and its compressor state.
        if layer.compress_ratio == spec.indexed_compress_ratio:
            pages.append(
                CacheLayerPage(
                    prefix + ".indexer",
                    "full",
                    block_size,
                    index_state,
                    spec.indexed_compress_ratio,
                )
            )
            pages.append(
                CacheLayerPage(
                    prefix + ".indexer_state",
                    "window",
                    4,
                    2 * 2 * spec.index_head_dim * 4,
                    window_tokens=8,
                )
            )
        pages.append(
            CacheLayerPage(
                prefix + ".swa",
                "window",
                64,
                kv_state,
                window_tokens=model.sliding_window,
                draft=layer.is_mtp,
            )
        )
        if layer.compress_ratio > 1:
            pages.append(
                CacheLayerPage(
                    prefix + ".compressed",
                    "full",
                    block_size,
                    kv_state,
                    layer.compress_ratio,
                )
            )
            coff = 2 if layer.compress_ratio == spec.indexed_compress_ratio else 1
            pages.append(
                CacheLayerPage(
                    prefix + ".compressor_state",
                    "window",
                    4 if coff == 2 else 8,
                    2 * coff * model.head_dim * 4,
                    window_tokens=coff * layer.compress_ratio,
                )
            )
    buckets = defaultdict(list)
    for page in pages:
        buckets[page.uniform_key].append(page)
    details = []
    for bucket in buckets.values():
        by_page = defaultdict(list)
        for page in bucket:
            by_page[page.page_bytes].append(page)
        counts = [len(names) for names in by_page.values()]
        balanced = len(set(counts)) == 1
        details.append((bucket, by_page, balanced, max(counts)))
    minimum = max(
        (
            count
            for _, by_page, balanced, count in details
            if balanced and len(by_page) > 1
        ),
        default=0,
    )
    repeats = (
        approximate_group_repeats(
            [count for _, _, balanced, count in details if balanced], minimum
        )
        if minimum
        else None
    )

    def group_count(detail):
        _, _, balanced, count = detail
        return cdiv(count, repeats) if balanced and repeats else 1

    def state_bucket(detail):
        bucket, _, _, _ = detail
        return repeats is None and bucket[0].kind == "window"

    anchor = max(
        sum(
            cdiv(
                len(layers),
                len(bucket) if state_bucket(detail) else group_count(detail),
            )
            * page_bytes
            for page_bytes, layers in by_page.items()
        )
        for detail in details
        for bucket, by_page, _, _ in [detail]
    )
    grouped = []
    for detail in details:
        bucket, by_page, _, _ = detail
        count = group_count(detail)
        if state_bucket(detail):
            count = max(
                count, cdiv(len(bucket), max(anchor // bucket[0].page_bytes, 1))
            )
        if count == 1:
            grouped.append(tuple(bucket))
        else:
            patterns = list(zip(*by_page.values()))
            for index in range(count):
                grouped.append(
                    tuple(page for repeat in patterns[index::count] for page in repeat)
                )
    groups = tuple(
        CacheGroup(i, layers, bool(speculative_tokens and any(p.draft for p in layers)))
        for i, layers in enumerate(grouped)
    )
    stride = max(group.group_bytes for group in groups)
    return HybridKVLayout(
        groups,
        lcm(*(group.block_tokens for group in groups)),
        stride,
        retention_interval,
        speculative_tokens,
        max_concurrent_batches * max_num_batched_tokens,
        model.max_position_embeddings,
    )
