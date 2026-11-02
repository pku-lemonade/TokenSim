"""Resident transformer stages with a separate shared embedding and logits pool.

The native transformer proxy omits explicitly replaced boundary operators.
The external pool prices the actual model embedding, normalization, LM head and
greedy selection. Useful-work credit always comes from the complete target.
"""

from copy import deepcopy
from dataclasses import replace

from TokenSim.config.validation import integer
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from scripts.lpx.mfu_capacity import ResidentCapacity
from scripts.lpx.mfu_metrics import useful_matrix_flops
from scripts.lpx.mfu_probe import MFUCase, runtime_for
from scripts.lpx.native_search import memory_report, streaming_settings
from scripts.lpx.partition import native_footprint, partition_solutions
from scripts.lpx.program import LP30Program
from scripts.lpx.pool_ops import column_shards, embedding, logits
from scripts.lpx.replay_profiles import cached_profile, NativeProfile, save_profile
from scripts.lpx.streaming import duration
from scripts.run_lpx_context_parallel import environment, layout_for


def transformer_setup(
    config, *, tp, queries, global_kv_layout="separate", batch=1, logits_rows=None
):
    """Keep the target communication budget while relocating boundary weights."""
    integer(tp, "transformer tensor width", minimum=1)
    integer(queries, "transformer verification queries", minimum=1)
    integer(batch, "transformer independent request batch", minimum=1)
    rows = batch * queries
    if logits_rows is not None:
        integer(logits_rows, "transformer output logits rows")
        if logits_rows > rows:
            raise ConfigurationError("transformer logits rows exceed hidden rows")
    config = deepcopy(config)
    _, full_model, _, _ = environment(config)
    full_model = replace(full_model, global_kv_layout=global_kv_layout)
    memory = config["memory"]
    memory["prefill_chunk"] = rows
    # Keep the full target's conservative communication arena on every stage.
    original_workspace = 4 * rows * max(full_model.hidden_size, full_model.vocab_size)
    if logits_rows is not None:
        original_workspace = 4 * max(
            rows * full_model.hidden_size, logits_rows * full_model.vocab_size
        )
    proxy_workspace = 4 * rows * full_model.hidden_size
    memory["reserve_bytes"] += original_workspace - proxy_workspace
    hardware, _, _, _ = environment(config)
    proxy = replace(
        full_model,
        model_id=full_model.model_id + "-transformer-io-proxy",
        vocab_size=tp,
        aliases=(),
    )
    return config, hardware, full_model, proxy


def transformer_runtime(
    config,
    *,
    tp,
    kvp,
    pp,
    queries,
    resident=None,
    kv_parallel_layers="full_attention",
    global_kv_layout="separate",
    batch=1,
    logits_rows=None,
):
    base_reserve = config["memory"]["reserve_bytes"]
    config, hardware, full_model, proxy = transformer_setup(
        config,
        tp=tp,
        queries=queries,
        global_kv_layout=global_kv_layout,
        batch=batch,
        logits_rows=logits_rows,
    )
    if pp > full_model.num_layers:
        raise ConfigurationError("transformer stages exceed model layers")
    case = MFUCase(
        tp, kvp, pp, batch=batch * queries, kv_parallel_layers=kv_parallel_layers
    )
    seed = runtime_for(hardware, proxy, case, proxy.num_layers)
    footprint = native_footprint(seed, 102000)
    screen = ResidentCapacity(footprint).frontier(256, pp)
    maximum = screen["maximum_resident_by_pp"].get(pp, 0)
    resident = maximum // batch * batch if resident is None else resident
    if not resident or resident > maximum or resident % batch:
        raise ConfigurationError("transformer proxy has insufficient resident KV")
    scaled = replace(
        footprint,
        kv=tuple(tuple(value * resident for value in row) for row in footprint.kv),
    )
    counts = partition_solutions(scaled, pp)[pp]["layer_counts"]
    runtime = runtime_for(hardware, proxy, case, pp, counts)
    admission = memory_report(runtime, 102000, resident)
    return (
        hardware,
        full_model,
        runtime,
        dict(
            config=config,
            proxy_model=proxy.to_dict(),
            resident_requests=resident,
            layer_counts=counts,
            memory=admission,
            extra_communication_arena_per_chip=config["memory"]["reserve_bytes"]
            - base_reserve,
            scratch_query_rows=batch * queries,
            **(
                dict(
                    independent_requests_per_forward=batch, queries_per_request=queries
                )
                if batch != 1
                else {}
            ),
        ),
    )


def transformer_profile(
    config,
    *,
    tp,
    kvp,
    pp,
    queries,
    cache,
    resident=None,
    kv_parallel_layers="full_attention",
    global_kv_layout="separate",
    batch=1,
):
    hardware, model, runtime, info = transformer_runtime(
        config,
        tp=tp,
        kvp=kvp,
        pp=pp,
        queries=queries,
        resident=resident,
        kv_parallel_layers=kv_parallel_layers,
        global_kv_layout=global_kv_layout,
        batch=batch,
    )
    profile = cached_profile(
        runtime,
        "mfu-search",
        cache,
        batch=batch,
        queries=queries,
        phase="decode",
        emit_logits=False,
        record_node_ids=True,
        context=101000,
        contexts=(101000,) * batch,
        maximum_contexts=(102000,) * batch,
        dp_rank=0,
        seed=42,
        sample=0,
        streaming=streaming_settings(layout_for(kvp, pp, tp), launch_us=0),
    )
    profile = transformer_component(profile, tp=tp, cache=cache)
    info["native_profile"] = profile.metadata
    return hardware, model, profile, info


def transformer_component(profile, *, tp, cache):
    """Omit native embedding work whose true replacement is explicitly composed."""
    names = profile.metadata["native_node_ids"]
    removed = {
        i
        for i, name in enumerate(names)
        if ":boundary:embedding_lookup:" in name
        or name.endswith(":boundary:embedding_reduce")
    }
    if len(removed) != tp + int(tp > 1) or any(
        ":boundary:lm_head:" in name or ":boundary:final_norm:" in name
        for name in names
    ):
        raise ConfigurationError("transformer-only native boundary inventory differs")
    pending = list(removed)
    while pending:
        index = pending.pop()
        for parent in profile.nodes[index].predecessors:
            if parent >= len(names) and parent not in removed:
                removed.add(parent)
                pending.append(parent)
    nodes = tuple(
        (
            replace(node, resources=(), primary=None, delay=0, duration=0, occupancy=0)
            if i in removed
            else node
        )
        for i, node in enumerate(profile.nodes)
    )
    metadata = dict(profile.metadata)
    parent = dict(
        input_sha256=metadata.pop("input_sha256"),
        node_body_sha256=metadata.pop("node_body_sha256"),
    )
    metadata.update(
        parent_profile=parent,
        elided_boundary_nodes=sorted(removed),
        modeled_duration_s=duration(nodes),
        component_scope="Transformer layers with external embedded input and no final normalization or logits; dummy boundary weights remain conservatively charged.",
    )
    identity = dict(parent_profile=parent, kind="external-embedding-transformer-v1")
    import hashlib
    import json
    from pathlib import Path

    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    result = save_profile(
        Path(cache) / (key + ".json.gz"), NativeProfile(nodes, metadata), identity
    )
    result.verify()
    return result


def pool_memory(model, ranks, queries, reserve=5_000_000, *, logits_rows=None):
    integer(queries, "pool query rows", minimum=1)
    integer(reserve, "pool reserve bytes")
    if logits_rows is not None:
        integer(logits_rows, "pool output logits rows")
        if logits_rows > queries:
            raise ConfigurationError("pool logits rows exceed hidden rows")
    if model.dtype not in {"fp8", "bf16", "fp16"} or model.activation_dtype != "bf16":
        raise ConfigurationError("I/O pool needs unpacked weights and BF16 activations")
    rows = []
    for rank, columns in enumerate(column_shards(model.vocab_size, ranks)):
        weights = (
            int(columns * model.hidden_size * dtype_size_bytes(model.dtype))
            + 2 * model.hidden_size
        )
        output = queries if logits_rows is None else logits_rows
        live = output * 6 * columns + queries * (8 * model.hidden_size + 256)
        communication = (
            queries * 4 * model.vocab_size
            if logits_rows is None
            else (4 * max(output * model.vocab_size, queries * model.hidden_size))
        )
        total = weights + live + communication + reserve
        rows.append(
            dict(
                rank=rank,
                weights_bytes=weights,
                live_bytes=live,
                communication_bytes=communication,
                reserve_bytes=reserve,
                total_bytes=total,
                headroom_bytes=500_000_000 - total,
            )
        )
    if any(row["headroom_bytes"] < 0 for row in rows):
        raise ConfigurationError("embedding/LM-head pool exceeds SRAM")
    return rows


def pooled_program(
    hardware, model, transformer, *, tp, pp, pool_chips, queries=1, batch=1
):
    integer(batch, "pool independent request batch", minimum=1)
    integer(queries, "pool queries per request", minimum=1)
    queries *= batch
    transformer_chips = tp * pp
    chips = tuple(range(transformer_chips, transformer_chips + pool_chips))
    admission = pool_memory(model, pool_chips, queries)
    p = LP30Program(hardware, transformer_chips + pool_chips)
    hosts = [
        node
        for node in transformer.nodes
        if any("/host:" in key for key in node.resources)
    ]
    if len(hosts) != 1 or hosts[0].predecessors:
        raise ConfigurationError("I/O composition requires one native host launch")
    host = hosts[0]
    launched = p.append((), host.resources, delay=host.delay, duration=host.duration)
    arena = "embedding-logits-pool/arena"
    acquired = p.acquire((launched,), arena)
    h = model.hidden_size
    ready = embedding(p, (acquired,), chips, model, queries)
    entered = p.transfer(
        (ready,),
        [(chips[rank % pool_chips], rank, queries * h * 2) for rank in range(tp)],
    )
    p.release_at(acquired, entered)
    transformed = p.include(transformer, (entered,), suppress_hosts=True)
    output_acquired = p.acquire((transformed,), arena)
    arrived = p.transfer(
        (output_acquired,),
        [
            ((pp - 1) * tp + rank % tp, chip, queries * h * 2)
            for rank, chip in enumerate(chips)
        ],
    )
    completed = logits(p, (arrived,), chips, model, queries)
    p.release_at(output_acquired, completed)
    return p.profile(), dict(
        pool_chips=pool_chips,
        pool_memory=admission,
        target_useful_work=useful_matrix_flops(model, 101000),
        protocol="Shared tied embedding/LM-head weights with greedy distributed selection; native boundary costs are replaced explicitly and dummy boundary weights remain charged.",
    )
