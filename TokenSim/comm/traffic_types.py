"""Immutable traffic volumes, phases and service schedules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math

from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import (
    MODEL_VERSION,
    transfer_resources,
    transmission_service,
)


def validated_bytes(value, dtype=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise ConfigurationError("message bytes must be a nonnegative integer")
    value = int(value)
    if dtype is not None and value % dtype_size_bytes(dtype, role="activation_dtype"):
        raise ConfigurationError("message bytes must align to dtype elements")
    return value


@dataclass(frozen=True)
class CollectiveVolume:
    operation: str
    participants: int
    input_bytes: int
    output_bytes: int
    rounds: int
    send_bytes_per_rank: float


def ring_volume(operation, participants, message_bytes, dtype=None):
    if type(participants) is not int or participants < 1:
        raise ConfigurationError("collective needs at least one actual participant")
    size = validated_bytes(message_bytes, dtype)
    if operation not in {"all_reduce", "all_gather", "reduce_scatter"}:
        raise ConfigurationError(f"unsupported ring volume operation {operation!r}")
    if operation in {"all_reduce", "reduce_scatter"} and participants > 1:
        width = dtype_size_bytes(dtype, role="activation_dtype") if dtype else 1
        if size % (participants * width):
            raise ConfigurationError(
                "ring shards must contain an integer number of dtype elements"
            )
    output = (
        size * participants
        if operation == "all_gather"
        else size // participants if operation == "reduce_scatter" else size
    )
    rounds = (participants - 1) * (2 if operation == "all_reduce" else 1)
    send = (
        (participants - 1) * size
        if operation == "all_gather"
        else rounds * size / participants
    )
    return CollectiveVolume(operation, participants, size, output, rounds, send)


@dataclass(frozen=True)
class Transmission:
    source: int
    target: int
    wire_bytes: float
    source_node: int
    target_node: int
    # Half-open tensor byte range; used by independent data oracles.
    shard: tuple[int, int] | None = None


@dataclass(frozen=True)
class TrafficPhase:
    phase_id: str
    operation: str
    participants: tuple[int, ...]
    fan: int
    logical_input_bytes: int
    logical_output_bytes: int
    rounds: tuple[tuple[Transmission, ...], ...]
    predecessor_ids: tuple[str, ...] = ()
    dtype: str = "bf16"
    local_work_bytes: int = 0
    layout_permutation: tuple[int, ...] = ()
    transport_backend: str = "unspecified"

    @property
    def wire_bytes(self):
        return sum(
            message.wire_bytes for messages in self.rounds for message in messages
        )

    @cached_property
    def resource_ids(self):
        return tuple(
            sorted(
                {
                    resource.identity
                    for messages in self.rounds
                    for message in messages
                    if message.wire_bytes
                    for resource in transfer_resources(
                        message.source_node, message.target_node
                    )
                }
            )
        )

    @property
    def resource_wire_bytes(self):
        totals = {}
        for messages in self.rounds:
            for message in messages:
                for resource in transfer_resources(
                    message.source_node, message.target_node
                ):
                    totals[resource.identity] = (
                        totals.get(resource.identity, 0) + message.wire_bytes
                    )
        return tuple(sorted(totals.items()))

    def relative_timetable(self, config):
        # Stable FIFO internal schedule, all inputs ready, no external queue.
        available = {}
        barrier = 0.0
        timetable = []
        for round_index, messages in enumerate(self.rounds):
            ends = [barrier]
            for message in messages:
                if not message.wire_bytes:
                    continue
                keys = tuple(
                    resource.identity
                    for resource in transfer_resources(
                        message.source_node, message.target_node
                    )
                )
                start = max(barrier, *(available.get(key, 0.0) for key in keys))
                end = start + transmission_service(
                    config, message.source_node, message.target_node, message.wire_bytes
                )
                timetable.append(
                    (
                        round_index,
                        message.source,
                        message.target,
                        keys,
                        start,
                        end,
                        message.wire_bytes,
                    )
                )
                for key in keys:
                    available[key] = end
                ends.append(end)
            barrier = max(ends)
        return tuple(timetable)

    def service(self, config):
        return max((row[5] for row in self.relative_timetable(config)), default=0.0)


@dataclass(frozen=True)
class TrafficPlan:
    operation: str
    participants: tuple[int, ...]
    input_bytes_per_rank: int
    output_bytes_per_rank: int | float | None
    phases: tuple[TrafficPhase, ...]
    provenance: str = "exact_routing"
    model_version: str = MODEL_VERSION
    expected_output_bytes: tuple[tuple[int, float], ...] = ()

    @property
    def wire_bytes(self):
        return sum(phase.wire_bytes for phase in self.phases)

    def service(self, config, service_for=None):
        ends, available = {}, {}
        pending = list(self.phases)
        while pending:
            candidates = [
                phase
                for phase in pending
                if all(key in ends for key in phase.predecessor_ids)
            ]
            if not candidates:
                raise ConfigurationError("cyclic or missing traffic phase dependency")
            phase = min(
                candidates,
                key=lambda item: (
                    max((ends[key] for key in item.predecessor_ids), default=0.0),
                    item.phase_id,
                ),
            )
            ready = max((ends[key] for key in phase.predecessor_ids), default=0.0)
            start = max(
                [ready] + [available.get(key, 0.0) for key in phase.resource_ids]
            )
            ends[phase.phase_id] = start + (
                service_for(phase) if service_for else phase.service(config)
            )
            for key in phase.resource_ids:
                available[key] = ends[phase.phase_id]
            pending.remove(phase)
        return max(ends.values(), default=0.0)
