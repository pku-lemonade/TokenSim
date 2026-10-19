"""Explicit full or incremental KV handoffs with destination replication."""

from collections import defaultdict
from dataclasses import dataclass
import simpy

from TokenSim.config.cache_config import stage_layer_range
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class KVCopy:
    source: str
    target: str
    layer_id: int
    head_id: int
    first_token: int
    end_token: int
    bytes: int
    network: str
    replication: bool = False


def _owners(service, session):
    lease = service.memory.leases[session.request_id]
    result = defaultdict(list)
    plan = service.runtime.plan
    for binding in plan.bindings_for_instance(lease.domain, lease.instance):
        if binding.kv_partition is None:
            continue
        stages = plan.domain_runtime(binding.domain).pipeline_parallel_size
        for layer in stage_layer_range(plan.total_layers, stages, binding.pp_rank):
            for head in binding.kv_partition.global_head_ids:
                result[(layer, head)].append(binding)
    return result


def plan_handoff(
    source,
    source_session,
    target,
    target_session,
    *,
    context,
    mode,
    cross_network,
    replication_network,
):
    if mode not in {"full", "incremental"}:
        raise ConfigurationError("KV handoff mode must be full or incremental")
    if source_session.request_id != target_session.request_id:
        raise ConfigurationError("KV handoff requires matching request identity")
    if (
        source.runtime.plan.effective_dtypes.kv_cache_dtype
        != target.runtime.plan.effective_dtypes.kv_cache_dtype
    ):
        raise ConfigurationError(
            "KV precision conversion requires an explicit conversion plan"
        )
    if source.runtime.model.cache_signature() != target.runtime.model.cache_signature():
        raise ConfigurationError(
            "KV handoff requires matching model and cache precision"
        )
    if source_session.cached_context < context:
        raise ConfigurationError("source KV does not cover the handoff context")
    if target_session.cached_context > context:
        raise ConfigurationError("destination KV cannot be ahead of the source")
    if context > target.memory.leases[target_session.request_id].maximum_context:
        raise ConfigurationError("destination KV reservation is too small")
    sources, targets = _owners(source, source_session), _owners(target, target_session)
    if set(sources) != set(targets):
        raise ConfigurationError("KV handoff layer/head partitions must match")
    model = source.runtime.model
    if model.attention_type == "mla":
        element_bytes = (
            source.runtime.plan.kv_dtype_size_bytes * model.mla_dimensions.cache_width
        )
    else:
        element_bytes = source.runtime.plan.kv_dtype_size_bytes * model.head_dim * 2
    existing = target_session.cached_context if mode == "incremental" else 0
    copies = []
    for (layer, head), destination in sorted(targets.items()):
        window = model.attention_window(layer)
        first = max(existing, context - window) if window else existing
        if first == context:
            continue
        producer = min(sources[(layer, head)], key=lambda binding: binding.domain_rank)
        consumers = sorted(destination, key=lambda binding: binding.domain_rank)
        canonical = consumers[0]
        size = (context - first) * element_bytes
        if producer.physical_identity != canonical.physical_identity:
            copies.append(
                KVCopy(
                    producer.physical_identity,
                    canonical.physical_identity,
                    layer,
                    head,
                    first,
                    context,
                    size,
                    cross_network,
                )
            )
        for consumer in consumers[1:]:
            copies.append(
                KVCopy(
                    canonical.physical_identity,
                    consumer.physical_identity,
                    layer,
                    head,
                    first,
                    context,
                    size,
                    replication_network,
                    True,
                )
            )
    return tuple(copies)


def execute_handoff(
    env,
    copies,
    *,
    hardware,
    resources,
    identity,
    target,
    target_session,
    context,
    events,
):
    ready = {}
    for copy in copies:
        ready.setdefault((copy.layer_id, copy.head_id), env.event())
    for key, event in ready.items():
        if not any(
            not copy.replication and (copy.layer_id, copy.head_id) == key
            for copy in copies
        ):
            event.succeed()

    def move(index, copy):
        key = (copy.layer_id, copy.head_id)
        if copy.replication:
            yield ready[key]
        endpoint_keys = tuple(
            resources.format_key(key)
            for key, _ in resources.for_step(
                copy.network, "a_to_f", (copy.source, copy.target)
            )
        )
        link = hardware.link(resources.networks[copy.network])
        duration = link.analytical_transfer_us(copy.bytes) / 1e6
        times = yield from resources.run_transfer(
            endpoint_keys,
            duration,
            link.latency_us / 1e6 if duration else 0.0,
            (*identity, index),
        )
        events.append(
            {
                "source": copy.source,
                "target": copy.target,
                "layer": copy.layer_id,
                "head": copy.head_id,
                "bytes": copy.bytes,
                "network": copy.network,
                "ready_s": times[0],
                "start_s": times[1],
                "end_s": times[2],
                "injection_end_s": times[3],
            }
        )
        if not copy.replication and not ready[key].triggered:
            ready[key].succeed()

    processes = [env.process(move(index, copy)) for index, copy in enumerate(copies)]
    for process in processes:
        process.defused = True
    try:
        if processes:
            joined = env.all_of(processes)
            joined.defused = True
            yield joined
    except BaseException:
        for process in processes:
            if process.is_alive:
                process.interrupt("KV handoff cancelled")
        for process in processes:
            if not process.triggered:
                try:
                    yield process
                except BaseException:
                    pass
        raise
    target.memory.commit(target_session.request_id, context)
    target_session.cached_context = context
