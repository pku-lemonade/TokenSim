"""Validate attention-service reservations at all contexts of a finite workload."""

from dataclasses import replace
import math

from TokenSim.af.providers.operator_table import analytical_execution_device
from TokenSim.af.runtime.kv_sequence import visible_tokens
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.analytical.lp30 import estimate_lp30
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.streaming import duration


def reserve_contexts(profile, runtime, contexts, queries):
    """Increase captured attention durations when any actual context costs more."""
    records = profile.metadata.get("attention_keys", ())
    if not records:
        raise ConfigurationError("context envelopes require recorded attention keys")
    contexts = tuple(contexts)
    if not contexts:
        raise ConfigurationError("context envelope has no covered positions")
    integer(queries, "context envelope query rows", minimum=1)
    for context in contexts:
        integer(context, "context envelope position", minimum=queries)
    original = runtime.hardware.device("groq_lp30")
    device = analytical_execution_device(
        original, runtime.cluster.af.graph_device_kernel_us
    )
    nodes = list(profile.nodes)
    groups = {}
    report = []
    for record in records:
        part = record["component"]
        if part["match_type"] != "analytical":
            raise ConfigurationError("context reservation needs analytical attention")
        table, key = part["table"], dict(part["key"])
        shape = record["local_shape"]
        kvp = shape.get("kv_parallel_size", 1)
        rank = shape.get("kv_parallel_rank", 0)
        block = shape.get("kv_append_block", 16)
        window = runtime.model.attention_window(record["layer_id"]) or 0
        identity = (
            table,
            tuple(sorted((k, v) for k, v in key.items() if k != "context_len")),
            kvp,
            rank,
            block,
            window,
        )
        if identity not in groups:
            base = estimate_lp30(table, key, device).latency_us
            if not math.isclose(base, part["latency_us"], rel_tol=1e-10, abs_tol=1e-9):
                raise ConfigurationError(
                    "context envelope differs from native component pricing"
                )
            maximum = base
            worst = None
            for context in contexts:
                local = context
                if kvp > 1:
                    local = (
                        min(
                            context,
                            queries
                            + visible_tokens(
                                context - queries, window or None, kvp, rank, block
                            ),
                        )
                        if queries > 1
                        else max(
                            1, visible_tokens(context, window or None, kvp, rank, block)
                        )
                    )
                actual = {**key, "context_len": local}
                cost = estimate_lp30(table, actual, device).latency_us
                if cost > maximum:
                    maximum, worst = cost, context
            groups[identity] = (
                maximum - base,
                dict(
                    table=table,
                    key=key,
                    kvp=kvp,
                    rank=rank,
                    covered_contexts=len(contexts),
                    captured_latency_us=base,
                    reserved_latency_us=maximum,
                    extension_context=worst,
                ),
            )
            report.append(groups[identity][1])
        extra = groups[identity][0] * 1e-6
        index = record["index"]
        nodes[index] = replace(
            nodes[index],
            duration=nodes[index].duration + extra,
            occupancy=nodes[index].occupancy + extra,
        )
    result = NativeProfile(
        tuple(nodes),
        {
            **profile.metadata,
            "modeled_duration_s": duration(nodes),
            "context_reservations": report,
            "reservation_scope": "Every requested context is priced analytically; captured per-node attention budgets are extended if needed. Whole-buffer, stage-gated execution remains an uncalibrated hypothesis.",
        },
    )
    return result
