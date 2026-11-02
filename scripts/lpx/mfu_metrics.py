"""Useful decode FLOPs and necessary throughput bounds for native DAGs."""

from collections import Counter, deque

from TokenSim.config.validation import boolean, integer
from TokenSim.errors import ConfigurationError


def useful_matrix_flops(model, context):
    integer(context, "MFU decode context", minimum=1)
    if model.is_moe or model.components:
        raise ConfigurationError(
            "MFU search currently requires an ordinary dense model"
        )
    linear = 2 * (
        sum(
            model.attention_params_per_layer(i)
            - (2 * model.attention_geometry(i).head_dim if model.qk_norm else 0)
            for i in range(model.num_layers)
        )
        + model.num_layers * model.dense_ffn_params_per_layer()
        + model.hidden_size * model.vocab_size
    )
    attention = sum(
        4
        * model.num_attention_heads
        * model.attention_geometry(i).head_dim
        * min(context, model.attention_window(i) or context)
        for i in range(model.num_layers)
    )
    return dict(linear=linear, attention=attention, total=linear + attention)


def dependency_order(nodes):
    """Validate dependencies and return one deterministic topological order."""
    if not nodes:
        raise ConfigurationError("MFU bound requires a nonempty DAG")
    children = [[] for _ in nodes]
    incoming = [len(node.predecessors) for node in nodes]
    ready = deque(i for i, degree in enumerate(incoming) if not degree)
    for index, node in enumerate(nodes):
        for parent in node.predecessors:
            if parent < 0 or parent >= len(nodes):
                raise ConfigurationError("MFU bound has an invalid predecessor")
            children[parent].append(index)
    order = []
    while ready:
        index = ready.popleft()
        order.append(index)
        for child in children[index]:
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
    if len(order) != len(nodes):
        raise ConfigurationError("MFU bound requires an acyclic DAG")
    return order


def critical_path_seconds(nodes):
    """Relax resource contention while retaining every declared dependency."""
    ends = [0.0] * len(nodes)
    for index in dependency_order(nodes):
        node = nodes[index]
        ready = max((ends[p] for p in node.predecessors), default=0.0)
        ends[index] = ready + node.delay + node.duration
    return max(ends)


def resource_demands(nodes, *, include_lifetimes=False):
    """Lower bounds on exclusive resource time per profile invocation.

    A lifetime acquisition holds resources through its release descendant.
    Its dependency path supplies a minimum hold time, excluding its own
    readiness delay and external ancestors that may finish before acquisition.
    Contention may extend the hold. Different program mixes need their own
    resource-demand aggregation before converting demand into throughput.
    """
    boolean(include_lifetimes, "MFU bound lifetime accounting")
    order = dependency_order(nodes) if include_lifetimes else None
    positions = {index: i for i, index in enumerate(order or ())}
    demand = Counter()
    for index, node in enumerate(nodes):
        if len(set(node.resources)) != len(node.resources):
            raise ConfigurationError("MFU demand has duplicate resources")
        service = node.occupancy
        if include_lifetimes and node.release_after is not None:
            target = node.release_after
            if target >= len(nodes) or positions[target] <= positions[index]:
                raise ConfigurationError("MFU lifetime release must descend")
            ends = {index: node.duration}
            for child in order[positions[index] + 1 : positions[target] + 1]:
                parents = [ends[p] for p in nodes[child].predecessors if p in ends]
                if parents:
                    ends[child] = (
                        max(parents) + nodes[child].delay + nodes[child].duration
                    )
            if target not in ends:
                raise ConfigurationError("MFU lifetime release must descend")
            service = ends[target]
        for resource in node.resources:
            demand[resource] += service
    return demand


def throughput_bounds(nodes, *, resident_requests, batch, include_lifetimes=False):
    """Necessary steady-throughput limits for identical fixed-width cohorts.

    These limits constrain the specified DAG and physical resource model.
    They do not certify a legal compiler schedule or bound different kernels.
    """
    integer(resident_requests, "bound resident requests", minimum=1)
    integer(batch, "bound batch size", minimum=1)
    if resident_requests % batch:
        raise ConfigurationError("MFU bound requires complete identical cohorts")
    demand = resource_demands(nodes, include_lifetimes=include_lifetimes)
    critical_path = critical_path_seconds(nodes)
    if critical_path <= 0:
        raise ConfigurationError("MFU bound requires positive DAG latency")
    busiest = max(demand, key=demand.get) if demand else None
    service = demand[busiest] if busiest else 0
    latency_limit = resident_requests / critical_path
    resource_limit = batch / service if service else None
    result = dict(
        aggregate_tps=min(latency_limit, resource_limit or latency_limit),
        latency_limited_tps=latency_limit,
        resource_limited_tps=resource_limit,
        critical_path_s=critical_path,
        busiest_resource=busiest,
        resource_seconds_per_forward=service,
        scope="Necessary steady-throughput bound for this fixed DAG; not an attainable schedule or global hardware bound.",
    )
    if include_lifetimes:
        result["includes_minimum_lifetime_holds"] = True
    return result
