"""Compose mixed attention layers using the same primitive latency backend."""

from collections import defaultdict
from copy import copy
from dataclasses import replace

from TokenSim.config.cache_config import stage_layer_range
from TokenSim.errors import ConfigurationError


def layer_backends(backend):
    model = backend.model
    if not model.attention_windows and not model.attention_geometries:
        return ()
    if model.components or backend.native_moe:
        raise ConfigurationError(
            "mixed ordinary attention cannot use compressed module estimators"
        )
    groups = defaultdict(list)
    for i in stage_layer_range(
        model.num_layers,
        backend.parallel_config.pipeline_parallel_size,
        backend.rank_info.pp_rank,
        backend.pipeline_layer_counts,
    ):
        groups[(model.attention_window(i), model.attention_geometry(i))].append(i)
    children = []
    moe = set(model.moe_layer_indices())
    for (window, geometry), indices in groups.items():
        child = copy(backend)
        child._layer_backends = ()
        child.model = replace(
            model,
            attention_windows=(),
            attention_geometries=(),
            sliding_window=window,
            head_dim=geometry.head_dim,
            num_key_value_heads=geometry.num_key_value_heads,
            k_eq_v=geometry.k_eq_v,
        )
        child._init_dimensions()
        child.layers_local = len(indices)
        child.moe_layers_local = len(moe.intersection(indices))
        child.dense_layers_local = len(indices) - child.moe_layers_local
        child.is_last_stage = False
        child.include_pp_transfer = False
        child.include_step_overhead = False
        children.append(child)
    children[-1].is_last_stage = backend.is_last_stage
    children[-1].include_pp_transfer = True
    children[-1].include_step_overhead = True
    return tuple(children)
