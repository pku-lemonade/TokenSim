"""Capacity-aware contiguous PP partitions using native rank memory components.

Dynamic programming minimizes the busiest rank for each fixed stage count.
It does not optimize speed or certify prefill activation workspace.
"""

from dataclasses import dataclass, replace
from itertools import accumulate
import json
import math

from TokenSim.af.runtime.admission import rank_layer_kv_bytes
from TokenSim.af.runtime.compiler import DEFAULT_KV_WATERMARK
from TokenSim.config.cache_config import stage_layer_range
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class PartitionFootprint:
    weights: tuple[tuple[int, ...], ...]
    kv: tuple[tuple[int, ...], ...]
    blocks: tuple[tuple[int, ...], ...]
    embedding: tuple[int, ...]
    buffers: tuple[int, ...]
    capacities: tuple[int, ...]
    tied_embeddings: bool = True
    kv_watermark: float = 0.0

    def __post_init__(self):
        number(self.kv_watermark, "partition KV watermark", minimum=0)
        if self.kv_watermark >= 1:
            raise ConfigurationError("partition KV watermark must be less than one")
        ranks, layers = len(self.capacities), len(self.weights)
        if not ranks or not layers:
            raise ConfigurationError("partition footprint requires ranks and layers")
        arrays = (self.weights, self.kv, self.blocks)
        if any(len(rows) != layers for rows in arrays) or any(
            len(row) != ranks
            or any(type(value) is not int or value < 0 for value in row)
            for rows in arrays
            for row in rows
        ):
            raise ConfigurationError(
                "partition layer footprints must be nonnegative rank vectors"
            )
        for name in ("embedding", "buffers", "capacities"):
            values = getattr(self, name)
            if len(values) != ranks or any(type(v) is not int or v < 0 for v in values):
                raise ConfigurationError(
                    "partition boundary footprints must match rank vectors"
                )
        if any(value <= 0 for value in self.capacities):
            raise ConfigurationError("partition rank capacities must be positive")

    def stage_bytes(self, start, end):
        layers = len(self.weights)
        integer(start, "partition start")
        integer(end, "partition end", minimum=start + 1)
        if end > layers:
            raise ConfigurationError("partition exceeds the model layer range")
        boundary = int(start == 0) + int(end == layers)
        if self.tied_embeddings:
            boundary = min(1, boundary)
        result = []
        for rank in range(len(self.capacities)):
            weights = sum(self.weights[i][rank] for i in range(start, end))
            kv = sum(self.kv[i][rank] for i in range(start, end))
            block = sum(self.blocks[i][rank] for i in range(start, end))
            # Invert the compiler's B - int(watermark * B) available-block rule.
            required = (kv + block - 1) // block if block else 0
            slots = required
            if required and self.kv_watermark:
                slots = max(required, int((required - 1) / (1 - self.kv_watermark)) + 1)
                while slots - int(self.kv_watermark * slots) < required:
                    slots += 1
                while (
                    slots > 0
                    and slots - 1 - int(self.kv_watermark * (slots - 1)) >= required
                ):
                    slots -= 1
            rounded_kv = slots * block if block else kv
            result.append(
                weights
                + rounded_kv
                + self.buffers[rank]
                + boundary * self.embedding[rank]
            )
        return tuple(result)


def partition_solutions(footprint, max_stages):
    """Find minimum peak SRAM occupancy for every allowed number of stages."""
    integer(max_stages, "partition maximum stages", minimum=1)
    layers = len(footprint.weights)
    edges = {}
    for start in range(layers):
        for end in range(start + 1, layers + 1):
            loads = footprint.stage_bytes(start, end)
            if all(n <= cap for n, cap in zip(loads, footprint.capacities)):
                edges[start, end] = max(
                    n / cap for n, cap in zip(loads, footprint.capacities)
                )
    previous, solutions = {0: (0.0, ())}, {}
    for stages in range(1, min(max_stages, layers) + 1):
        current = {}
        for (start, end), load in edges.items():
            if start not in previous:
                continue
            peak, counts = previous[start]
            candidate = (max(peak, load), counts + (end - start,))
            if end not in current or candidate < current[end]:
                current[end] = candidate
        if layers in current:
            peak, counts = current[layers]
            boundaries = tuple(accumulate((0, *counts)))
            solutions[stages] = dict(
                layer_counts=counts,
                kv_watermark=footprint.kv_watermark,
                peak_capacity_fraction=peak,
                stage_rank_bytes=[
                    footprint.stage_bytes(a, b)
                    for a, b in zip(boundaries, boundaries[1:])
                ],
            )
        previous = current
    return solutions


def native_footprint(runtime, maximum_context):
    """Extract each layer once per TP rank, including colocated FFN weights."""
    plan, model = runtime.plan, runtime.model
    if (
        model.is_moe
        or model.components
        or any(w.hardware != "groq_lp30" for w in runtime.workers)
    ):
        raise ConfigurationError(
            "capacity partition search requires pure LP30 dense execution"
        )
    attention = next(d for d in plan.domains if d.capability == "af_attention")
    if attention.data_parallel_size != 1:
        raise ConfigurationError("capacity partition search requires one replica")
    tp, layers = attention.tensor_parallel_size, plan.total_layers
    weights = [[None] * tp for _ in range(layers)]
    kv, blocks = [[0] * tp for _ in range(layers)], [[0] * tp for _ in range(layers)]
    embedding, buffers, capacities = [0] * tp, [0] * tp, [math.inf] * tp
    for binding in plan.rank_bindings:
        if binding.domain != attention.name:
            continue
        rank = binding.tp_rank
        state = plan.physical_state(binding.physical_identity)
        local_kv = rank_layer_kv_bytes(plan, binding, maximum_context)
        memory_policy = runtime.hardware.device(binding.hardware).analytical.get(
            "lpx_memory"
        )

        def auxiliary(components):
            if memory_policy is None:
                return 0
            from TokenSim.af.runtime.lpx_memory import workspace_components

            return dict(workspace_components(components, model, memory_policy))[
                "lp30_auxiliary_weights"
            ]

        owned = stage_layer_range(
            layers,
            attention.pipeline_parallel_size,
            binding.pp_rank,
            attention.pipeline_layer_counts,
        )
        partition = binding.kv_partition
        for layer in owned:
            if weights[layer][rank] is not None:
                raise ConfigurationError("capacity footprint duplicates a layer shard")
            weights[layer][rank] = sum(
                c.bytes for c in state.components if c.layer_id == layer
            ) + auxiliary(tuple(c for c in state.components if c.layer_id == layer))
            kv[layer][rank] = local_kv[layer]
            blocks[layer][rank] = (
                partition.for_layer(layer).bytes_per_block
                if partition.layer_partitions
                else partition.bytes_per_block // len(owned)
            )
        boundary_components = tuple(
            c for c in state.components if c.category == "embedding_lm_head"
        )
        embedding[rank] = max(
            embedding[rank],
            sum(c.bytes for c in boundary_components) + auxiliary(boundary_components),
        )
        buffers[rank] = max(
            buffers[rank],
            state.protocol_buffer_bytes
            - dict(state.protocol_buffers).get("lp30_auxiliary_weights", 0),
        )
        capacities[rank] = min(capacities[rank], state.capacity_bytes)
    if any(value is None for row in weights for value in row):
        raise ConfigurationError("capacity footprint misses a layer shard")
    return PartitionFootprint(
        tuple(map(tuple, weights)),
        tuple(map(tuple, kv)),
        tuple(map(tuple, blocks)),
        tuple(embedding),
        tuple(buffers),
        tuple(capacities),
        model.tie_word_embeddings,
        DEFAULT_KV_WATERMARK,
    )


def capacity_partition(hardware, model, layout, maximum_context, cache, *, reuse=True):
    """Reuse compact memory vectors across PP points; retain no runtime graphs."""
    from scripts.lpx.native_search import build_runtime

    if not layout.colocated or layout.a_device != "groq_lp30" or layout.f_ep != 1:
        raise ConfigurationError(
            "capacity partition search requires colocated dense LP30 ranks"
        )
    tp = layout.a_tp
    key = (
        json.dumps(model.to_dict(), sort_keys=True, allow_nan=False),
        json.dumps(
            hardware.device("groq_lp30").to_dict(), sort_keys=True, allow_nan=False
        ),
        maximum_context,
        reuse,
        tp,
    )
    if key not in cache:
        stages = min(model.num_layers, 256 // tp)
        seed = replace(layout, a_pp=stages, f_pp=stages)
        try:
            runtime = build_runtime(hardware, model, seed, reuse=reuse)
            footprint = native_footprint(runtime, maximum_context)
            cache[key] = partition_solutions(footprint, stages)
        except ConfigurationError as exc:
            cache[key] = str(exc)
    if isinstance(cache[key], str):
        raise ConfigurationError(cache[key])
    result = cache[key].get(layout.a_pp)
    if result is None:
        raise ConfigurationError(
            "no contiguous pipeline partition fits native rank memory budgets"
        )
    return result
