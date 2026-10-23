"""Experimental latency accounting for one fixed, colocated LPX decode layout.

Instrumentation and estimator patches remain local to the calling audit.
No estimator, layout, or hardware default is changed by importing this module.
"""

from collections import Counter
from dataclasses import replace
from functools import lru_cache
from math import isclose
from unittest.mock import patch

import simpy

from TokenSim.af.providers.traffic import TransferPlanMixin
from TokenSim.af.providers.transfer import TransferEstimateMixin
from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.af.serving.service import AFService
from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.comm.collectives import CollectiveModel
from TokenSim.latency.operator_table.queries import OperatorQueries
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.analytical import analytical_model_for
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration


def close(left, right):
    if not isclose(left, right, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError(f"latency conservation failed: {left} != {right}")


def build(hardware, *, colocated_dense_reuse=False, micro_batch=1):
    model = replace(
        hardware.models.resolve("gemma-4-31b-text"),
        dtype="fp8",
        kv_cache_dtype="fp8",
    )
    layout = Layout(
        a_device="groq_lp30",
        f_device="groq_lp30",
        a_tp=16,
        a_pp=16,
        f_tp=16,
        f_pp=16,
        f_ep=1,
    )
    runtime = configuration(
        hardware,
        model,
        layout,
        namespace="ledger",
        dag_template_cache=True,
        colocated_dense_reuse=colocated_dense_reuse,
        micro_batch=micro_batch,
    )
    memory = ResidentKV(runtime.plan)
    owner = next(iter(runtime.plan.request_owners))
    memory.admit(
        0, owner.domain, owner.instance, maximum_context=102000, current_context=0
    )
    return runtime


class WeightedLatency(float):
    """Carry linear query multiplicities through grouped-layer arithmetic."""

    def __new__(cls, value, weights):
        result = super().__new__(cls, value)
        result.weights = weights
        return result

    def __add__(self, other):
        weights = self.weights.copy()
        weights.update(getattr(other, "weights", {}))
        return WeightedLatency(float(self) + float(other), weights)

    __radd__ = __add__

    def __mul__(self, scalar):
        if isinstance(scalar, WeightedLatency):
            raise TypeError("operator ledger requires linear latency arithmetic")
        return WeightedLatency(
            float(self) * scalar,
            Counter({key: value * scalar for key, value in self.weights.items()}),
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1 / scalar)


def identity(table, key):
    return table, tuple(sorted(key.items()))


def priced_operators(runtime, weights):
    device = runtime.hardware.device("groq_lp30")
    analytical = analytical_model_for(device.family)
    rows, totals = [], Counter()
    for (table, fields), count in sorted(weights.items()):
        close(count, round(count))
        count = round(count)
        if not count:
            continue
        key = dict(fields)
        estimate = analytical.estimate(table, key, device)
        dominant = (
            "compute_roofline_us"
            if estimate.compute_time_us >= estimate.memory_time_us
            else "sram_roofline_us"
        )
        totals[dominant] += count * max(
            estimate.compute_time_us, estimate.memory_time_us
        )
        totals["startup_us"] += count * estimate.overhead_us
        totals["service_us"] += count * estimate.latency_us
        totals["operators"] += count
        rows.append(
            dict(
                table=table,
                key=key,
                count=count,
                weighted_latency_us=count * estimate.latency_us,
                **estimate.analysis_row(),
            )
        )
    close(
        totals["service_us"],
        totals["startup_us"]
        + totals["compute_roofline_us"]
        + totals["sram_roofline_us"],
    )
    return dict(totals), rows


def composed(runtime, context):
    service = AFService(simpy.Environment(), runtime, trace=False, retain_trace=False)
    owner = next(iter(service.owners.values()))
    bindings = runtime.plan.bindings_for_instance(owner.domain, owner.instance)
    backends = [
        service.prefill._backend(binding)
        for binding in sorted(bindings, key=lambda b: b.pp_rank)
        if binding.tp_rank == 0
    ]
    for backend in backends:
        backend.decode_context_bucket = 1
        for child in backend._layer_backends:
            child.decode_context_bucket = 1
    request = Request(0, context - 1, 2000, 16)
    request.num_computed_tokens = context - 1
    request.generation_idx = 1
    request.scheduled_tokens = request.scheduled_query_tokens = 1
    request.emit_logits = True
    original = OperatorQueries.query

    def query(backend, table, key):
        result = original(backend, table, key)
        return replace(
            result,
            latency_us=WeightedLatency(
                result.latency_us, Counter({identity(table, key): 1})
            ),
        )

    communication, components = Counter(), Counter()
    with patch.object(OperatorQueries, "query", query):
        latency = sum(backend.estimate_step_latency([request]) for backend in backends)
    for backend in backends:
        for kind, group, payload, seconds in backend.communicator.operations:
            communication[kind] += seconds * 1e6
        for name, seconds in backend.stats.component_seconds.items():
            components[name] += float(seconds) * 1e6
    weights = Counter({key: value * 1e6 for key, value in latency.weights.items()})
    totals, operators = priced_operators(runtime, weights)
    close(float(latency) * 1e6, totals["service_us"] + sum(communication.values()))
    return dict(
        context=context,
        latency_us=float(latency) * 1e6,
        tps=1 / float(latency),
        operator_budget=totals,
        communication_us=dict(communication),
        backend_components_us=dict(components),
        operators=operators,
    )


def catalog_collective(runtime):
    original = TransferPlanMixin._collective
    hardware = runtime.hardware
    endpoints = {
        worker.physical_worker_ref.identity: worker.physical_worker_ref.worker_index
        for worker in runtime.workers
    }
    device = hardware.device("groq_lp30")
    model = CollectiveModel(
        hardware.topology("groq_lpx256"),
        hardware.links,
        family=device.family,
        launch_us=device.analytical.get("collective_launch_us"),
        fallback="analytical_only",
    )

    @lru_cache(maxsize=None)
    def estimate(operation, payload, physical, dtype):
        full_bytes = payload * len(physical) if operation == "all_gather" else payload
        return model.estimate(
            CollectiveQuery(
                operation,
                full_bytes,
                model.topology.group_layout([endpoints[e] for e in physical]),
                dtype,
            )
        )

    def collective(provider, query):
        previous = original(provider, query)
        operation = (
            "all_reduce" if query.operation == "tp_collective" else query.operation
        )
        result = estimate(
            operation, query.payload_bytes, tuple(query.physical_endpoints), query.dtype
        )
        return replace(
            previous,
            service_duration=result.latency_us / 1e6,
            source_id=result.source_id,
            dataset_version="experimental:catalog-alignment",
            details={
                **previous.details,
                "total_latency_us": result.latency_us,
                "algorithm": "catalog_hierarchical",
                "launch_us": result.launch_us,
                "levels": [vars(level) for level in result.levels],
            },
        )

    return collective


def free_services(*, handoffs=False, residual_gathers=False):
    """Counterfactual attribution; zero time does not validate a tensor layout."""
    original = TransferEstimateMixin.estimate

    def estimate(provider, query):
        result = original(provider, query)
        skip = handoffs and query.operation in {"af_a_to_f", "af_f_to_a"}
        skip |= (
            residual_gathers and query.operation == "all_gather" and query.layer_id >= 0
        )
        if not skip:
            return result
        return replace(
            result,
            service_duration=0.0,
            source_id="diagnostic:free-service",
            details={
                **result.details,
                "fixed_latency_us": 0.0,
                "total_latency_us": 0.0,
            },
        )

    return estimate


def native(runtime, context, *, steps=1, details=True):
    env = simpy.Environment()
    service = AFService(env, runtime, seed=42, retain_trace=True)
    session = service.admit(0, 0, maximum_context=102000, current_context=context - 1)
    session.generated = 1
    elapsed = []

    def decode():
        for _ in range(steps):
            start = env.now
            yield env.process(service.forward([session], phase="decode"))
            elapsed.append((env.now - start) * 1e6)

    env.process(decode())
    env.run()
    events = service.trace.events
    result = dict(
        context=context,
        latency_us=env.now * 1e6 / steps,
        tps=steps / env.now,
        step_latencies_us=elapsed,
        events=len(events),
        event_counts=dict(Counter(event.operation for event in events)),
    )
    if details:
        if steps != 1:
            raise ValueError("detailed ledger requires one decode step")
        result.update(path_ledger(runtime, events))
    service.trace.clear()
    return result


def path_ledger(runtime, events):
    by_id = {event.node_id: event for event in events}
    node = max(events, key=lambda event: event.end_at)
    chain, seen = [], set()
    while True:
        if node.node_id in seen:
            raise AssertionError("dependency cycle")
        seen.add(node.node_id)
        chain.append(node)
        if not node.predecessor_ids:
            break
        node = max(
            (by_id[parent] for parent in node.predecessor_ids),
            key=lambda event: event.end_at,
        )
    times, queues, gaps, counts, weights, named = (Counter() for _ in range(6))
    rows = []
    for event in reversed(chain):
        parent_end = max(
            (by_id[parent].end_at for parent in event.predecessor_ids), default=0.0
        )
        gap = (event.ready_at - parent_end) * 1e6
        if gap < -1e-8:
            raise AssertionError("event readiness precedes its latest predecessor")
        times[event.operation] += event.service_duration * 1e6
        queues[event.operation] += event.queue_duration * 1e6
        gaps[event.operation] += gap
        counts[event.operation] += 1
        close(
            event.end_at * 1e6 - parent_end * 1e6,
            event.service_duration * 1e6 + event.queue_duration * 1e6 + gap,
        )
        components = event.details.get("components", ())
        if components:
            close(
                sum(part["latency_us"] for part in components),
                event.service_duration * 1e6,
            )
        for part in components:
            weights[identity(part["table"], part["key"])] += 1
            named[part["name"]] += part["latency_us"]
        rows.append(
            dict(
                node_id=event.node_id,
                layer=event.layer_id,
                operation=event.operation,
                service_us=event.service_duration * 1e6,
                queue_us=event.queue_duration * 1e6,
                readiness_gap_us=gap,
            )
        )
    total = sum(times.values()) + sum(queues.values()) + sum(gaps.values())
    close(total, max(event.end_at for event in events) * 1e6)
    budget, operators = priced_operators(runtime, weights)
    close(budget["service_us"], sum(named.values()))
    return dict(
        path_service_us=dict(times),
        path_queue_us=dict(queues),
        path_readiness_gap_us=dict(gaps),
        path_counts=dict(counts),
        accounted_us=total,
        operator_budget=budget,
        named_operators_us=dict(named),
        operators=operators,
        path=rows,
    )
