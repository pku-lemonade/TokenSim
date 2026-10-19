"""Build collective, point-to-point and statistical traffic plans."""

from __future__ import annotations

import math

from TokenSim.comm.traffic_types import (
    CollectiveVolume,
    TrafficPhase,
    TrafficPlan,
    Transmission,
    ring_volume,
    validated_bytes,
)
from TokenSim.errors import ConfigurationError


def _ring_phase(phase_id, operation, ranks, size, nodes, predecessors=(), dtype=None):
    volume = ring_volume(operation, len(ranks), size, dtype)
    count = len(ranks)
    rounds = []
    if count > 1 and size:
        chunk = size if operation == "all_gather" else size // count
        for step in range(volume.rounds):
            round_messages = []
            for index, source in enumerate(ranks):
                target = ranks[(index + 1) % count]
                # RS sends index-step; AG starts with the shard owned after RS.
                if operation == "reduce_scatter" or (
                    operation == "all_reduce" and step < count - 1
                ):
                    shard_index = (index - step - 1) % count
                elif operation == "all_reduce":
                    shard_index = (index - (step - (count - 1))) % count
                else:
                    shard_index = (index - step) % count
                round_messages.append(
                    Transmission(
                        source,
                        target,
                        chunk,
                        nodes[source],
                        nodes[target],
                        (shard_index * chunk, (shard_index + 1) * chunk),
                    )
                )
            rounds.append(tuple(round_messages))
    return TrafficPhase(
        phase_id,
        operation,
        tuple(ranks),
        count,
        size,
        volume.output_bytes,
        tuple(rounds),
        tuple(predecessors),
        dtype or "bf16",
    )


def collective_plan(operation, ranks, message_bytes, node_by_rank, dtype="bf16"):
    ranks = tuple(ranks)
    if (
        not ranks
        or len(set(ranks)) != len(ranks)
        or any(rank not in node_by_rank for rank in ranks)
    ):
        raise ConfigurationError(
            "collective participants must be nonempty, unique and mapped"
        )
    size = validated_bytes(message_bytes, dtype)
    volume = ring_volume(operation, len(ranks), size, dtype)
    by_node = {}
    for rank in ranks:
        by_node.setdefault(node_by_rank[rank], []).append(rank)
    groups = tuple(tuple(group) for _, group in sorted(by_node.items()))
    if len({len(group) for group in groups}) != 1:
        raise ConfigurationError(
            "hierarchical collective requires equal participants per node"
        )
    local_count = len(groups[0])
    if len(groups) == 1 or local_count == 1:
        phase = _ring_phase("ring", operation, ranks, size, node_by_rank, dtype=dtype)
        return TrafficPlan(operation, ranks, size, volume.output_bytes, (phase,))
    phases = []
    if operation == "all_reduce":
        for index, group in enumerate(groups):
            phases.append(
                _ring_phase(
                    f"local_rs:{index}",
                    "reduce_scatter",
                    group,
                    size,
                    node_by_rank,
                    dtype=dtype,
                )
            )
        for slot in range(local_count):
            subgroup = tuple(group[slot] for group in groups)
            phases.append(
                _ring_phase(
                    f"inter_ar:{slot}",
                    "all_reduce",
                    subgroup,
                    size // local_count,
                    node_by_rank,
                    tuple(f"local_rs:{i}" for i in range(len(groups))),
                    dtype,
                )
            )
        for index, group in enumerate(groups):
            phases.append(
                _ring_phase(
                    f"local_ag:{index}",
                    "all_gather",
                    group,
                    size // local_count,
                    node_by_rank,
                    tuple(f"inter_ar:{slot}" for slot in range(local_count)),
                    dtype,
                )
            )
    elif operation == "all_gather":
        for slot in range(local_count):
            phases.append(
                _ring_phase(
                    f"inter_ag:{slot}",
                    "all_gather",
                    tuple(group[slot] for group in groups),
                    size,
                    node_by_rank,
                    dtype=dtype,
                )
            )
        for index, group in enumerate(groups):
            phases.append(
                _ring_phase(
                    f"local_ag:{index}",
                    "all_gather",
                    group,
                    size * len(groups),
                    node_by_rank,
                    tuple(f"inter_ag:{slot}" for slot in range(local_count)),
                    dtype,
                )
            )
        # Slot-major gathered blocks need a local permutation into logical
        # participant order. This is device work, never extra network traffic.
        if size:
            gathered_order = tuple(
                group[slot] for slot in range(local_count) for group in groups
            )
            permutation = tuple(gathered_order.index(rank) for rank in ranks)
            if permutation != tuple(range(len(ranks))):
                for index, group in enumerate(groups):
                    phases.append(
                        TrafficPhase(
                            f"layout:{index}",
                            "data_reorder",
                            group,
                            local_count,
                            volume.output_bytes,
                            volume.output_bytes,
                            (),
                            (f"local_ag:{index}",),
                            dtype,
                            volume.output_bytes,
                            permutation,
                        )
                    )
    else:
        grouped_order = tuple(rank for group in groups for rank in group)
        reorder_input = grouped_order != ranks and size > 0
        if reorder_input:
            permutation = tuple(ranks.index(rank) for rank in grouped_order)
            for index, group in enumerate(groups):
                phases.append(
                    TrafficPhase(
                        f"input_layout:{index}",
                        "data_reorder",
                        group,
                        local_count,
                        size,
                        size,
                        (),
                        (),
                        dtype,
                        size,
                        permutation,
                    )
                )
        for index, group in enumerate(groups):
            for destination in range(len(groups)):
                phases.append(
                    _ring_phase(
                        f"local_rs:{index}:{destination}",
                        "reduce_scatter",
                        group,
                        size // len(groups),
                        node_by_rank,
                        (f"input_layout:{index}",) if reorder_input else (),
                        dtype=dtype,
                    )
                )
        for slot in range(local_count):
            # Each slot has one reduced shard for every destination node.
            phases.append(
                _ring_phase(
                    f"inter_rs:{slot}",
                    "reduce_scatter",
                    tuple(group[slot] for group in groups),
                    size // local_count,
                    node_by_rank,
                    tuple(
                        f"local_rs:{i}:{j}"
                        for i in range(len(groups))
                        for j in range(len(groups))
                    ),
                    dtype,
                )
            )
    return TrafficPlan(operation, ranks, size, volume.output_bytes, tuple(phases))


def point_to_point_plan(
    source, target, message_bytes, node_by_rank, dtype=None, wire_dtype=None
):
    size = validated_bytes(message_bytes, dtype)
    if source == target:
        return TrafficPlan("send_recv", (source,), size, size, ())
    message = Transmission(
        source, target, size, node_by_rank[source], node_by_rank[target]
    )
    phase = TrafficPhase(
        "transfer",
        "send_recv",
        (source, target),
        2,
        size,
        size,
        ((message,),) if size else (),
        dtype=wire_dtype or dtype or "bf16",
    )
    return TrafficPlan("send_recv", (source, target), size, size, (phase,))


def validate_all_to_all_parameters(
    local_fraction=None,
    remote_fraction=None,
    load_imbalance=1.0,
    destination_redistribution=False,
):
    """Topology-independent validation shared by flat and physical pricing."""
    for fraction in (local_fraction, remote_fraction):
        if fraction is not None and (
            isinstance(fraction, bool)
            or not isinstance(fraction, (float, int))
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ConfigurationError(
                "local/remote fractions must be finite values in [0,1]"
            )
    if (
        local_fraction is not None
        and remote_fraction is not None
        and not math.isclose(local_fraction + remote_fraction, 1, abs_tol=1e-12)
    ):
        raise ConfigurationError("local_fraction + remote_fraction must equal one")
    if (
        isinstance(load_imbalance, bool)
        or not isinstance(load_imbalance, (int, float))
        or not math.isfinite(load_imbalance)
        or load_imbalance < 1
    ):
        raise ConfigurationError("load_imbalance must be finite and at least one")
    if type(destination_redistribution) is not bool:
        raise ConfigurationError("destination_redistribution must be boolean")


def all_to_all_plan(
    ranks,
    message_bytes,
    node_by_rank,
    *,
    local_fraction=None,
    remote_fraction=None,
    load_imbalance=1.0,
    destination_redistribution=False,
    dtype="bf16",
):
    """Expected rank-pair volumes; never invent per-token expert assignments."""
    ranks = tuple(ranks)
    size = validated_bytes(message_bytes, dtype)
    if (
        not ranks
        or len(set(ranks)) != len(ranks)
        or any(rank not in node_by_rank for rank in ranks)
    ):
        raise ConfigurationError(
            "all-to-all participants must be nonempty, unique and mapped"
        )
    validate_all_to_all_parameters(
        local_fraction, remote_fraction, load_imbalance, destination_redistribution
    )
    groups = {}
    for rank in ranks:
        groups.setdefault(node_by_rank[rank], []).append(rank)
    if len({len(group) for group in groups.values()}) != 1:
        raise ConfigurationError("all-to-all needs equal participants per node")
    nodes = tuple(sorted(groups))
    local_count = len(groups[nodes[0]])
    provenance = (
        "uniform_fallback"
        if local_fraction is None and remote_fraction is None
        else "statistical_locality"
    )
    fraction = (
        local_fraction
        if local_fraction is not None
        else (
            1 - remote_fraction
            if remote_fraction is not None
            else local_count / len(ranks)
        )
    )
    if len(nodes) == 1 and fraction != 1:
        raise ConfigurationError("single-node all-to-all cannot have remote payload")
    local_messages = {node: [] for node in nodes}
    remote_messages = {}
    target_messages = {node: [] for node in nodes}
    received = {rank: size * fraction for rank in ranks}
    for source in ranks:
        source_node = node_by_rank[source]
        for target in groups[source_node]:
            if source != target:
                local_messages[source_node].append(
                    Transmission(
                        source,
                        target,
                        size * fraction / local_count,
                        source_node,
                        source_node,
                    )
                )
        remote_nodes = [node for node in nodes if node != source_node]
        weights = [
            load_imbalance if index == 0 else 1.0 for index in range(len(remote_nodes))
        ]
        for target_node, weight in zip(remote_nodes, weights):
            per_target = size * (1 - fraction) * weight / sum(weights) / local_count
            messages = remote_messages.setdefault((source_node, target_node), [])
            for slot, target in enumerate(groups[target_node]):
                received[target] += per_target
                relay = groups[source_node][slot]
                if relay != source:
                    local_messages[source_node].append(
                        Transmission(
                            source, relay, per_target, source_node, source_node
                        )
                    )
                receiver = (
                    groups[target_node][0] if destination_redistribution else target
                )
                messages.append(
                    Transmission(relay, receiver, per_target, source_node, target_node)
                )
                if receiver != target:
                    target_messages[target_node].append(
                        Transmission(
                            receiver, target, per_target, target_node, target_node
                        )
                    )
    phases = []
    for node in nodes:
        phases.append(
            TrafficPhase(
                f"source_local:{node}",
                "all_to_all",
                tuple(groups[node]),
                local_count,
                size,
                size,
                (tuple(local_messages[node]),),
                dtype=dtype,
            )
        )
    for (source_node, target_node), messages in sorted(remote_messages.items()):
        phases.append(
            TrafficPhase(
                f"inter:{source_node}:{target_node}",
                "all_to_all",
                tuple(groups[source_node] + groups[target_node]),
                len(nodes),
                size,
                size,
                (tuple(messages),),
                (f"source_local:{source_node}",),
                dtype=dtype,
            )
        )
    if destination_redistribution:
        for node, messages in sorted(target_messages.items()):
            phases.append(
                TrafficPhase(
                    f"destination_local:{node}",
                    "all_to_all",
                    tuple(groups[node]),
                    local_count,
                    size,
                    size,
                    (tuple(messages),),
                    tuple(
                        f"inter:{source}:{node}" for source in nodes if source != node
                    ),
                    dtype=dtype,
                )
            )
    output = next(iter(received.values()))
    uniform_output = all(
        math.isclose(value, output, rel_tol=1e-12, abs_tol=1e-9)
        for value in received.values()
    )
    return TrafficPlan(
        "all_to_all",
        ranks,
        size,
        output if uniform_output else None,
        tuple(phases),
        provenance,
        expected_output_bytes=tuple(received.items()),
    )
