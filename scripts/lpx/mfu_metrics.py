"""Useful decode FLOPs and necessary throughput bounds for native DAGs."""

from collections import Counter, deque

from TokenSim.config.validation import integer
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


def critical_path_seconds(nodes):
    """Relax resource contention while retaining every declared dependency."""
    if not nodes:
        raise ConfigurationError("MFU bound requires a nonempty DAG")
    children = [[] for _ in nodes]
    incoming = [len(node.predecessors) for node in nodes]
    ready = deque(i for i, degree in enumerate(incoming) if not degree)
    starts = [node.delay for node in nodes]
    ends = [0.0] * len(nodes)
    for index, node in enumerate(nodes):
        for parent in node.predecessors:
            if parent < 0 or parent >= len(nodes):
                raise ConfigurationError("MFU bound has an invalid predecessor")
            children[parent].append(index)
    visited = 0
    while ready:
        index = ready.popleft()
        visited += 1
        ends[index] = starts[index] + nodes[index].duration
        for child in children[index]:
            starts[child] = max(starts[child], ends[index] + nodes[child].delay)
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
    if visited != len(nodes):
        raise ConfigurationError("MFU bound requires an acyclic DAG")
    return max(ends)


def throughput_bounds(nodes, *, resident_requests, batch):
    """Necessary steady-throughput limits for identical fixed-width cohorts.

    These limits constrain the specified DAG and physical resource model.
    They do not certify a legal compiler schedule or bound different kernels.
    """
    integer(resident_requests, "bound resident requests", minimum=1)
    integer(batch, "bound batch size", minimum=1)
    if resident_requests % batch:
        raise ConfigurationError("MFU bound requires complete identical cohorts")
    demand = Counter()
    for node in nodes:
        for resource in node.resources:
            demand[resource] += node.occupancy
    critical_path = critical_path_seconds(nodes)
    if critical_path <= 0:
        raise ConfigurationError("MFU bound requires positive DAG latency")
    busiest = max(demand, key=demand.get) if demand else None
    service = demand[busiest] if busiest else 0
    latency_limit = resident_requests / critical_path
    resource_limit = batch / service if service else None
    return dict(
        aggregate_tps=min(latency_limit, resource_limit or latency_limit),
        latency_limited_tps=latency_limit,
        resource_limited_tps=resource_limit,
        critical_path_s=critical_path,
        busiest_resource=busiest,
        resource_seconds_per_forward=service,
        scope="Necessary steady-throughput bound for this fixed DAG; not an attainable schedule or global hardware bound.",
    )
