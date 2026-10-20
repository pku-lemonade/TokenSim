"""Algorithm-aware alpha-beta costs with per-rank input byte semantics.

This flat fabric approximation does not model topology routes, software
launch, reduction arithmetic or compute overlap. Direct collectives require
an explicitly assumed nonblocking fabric; auto compares ring and tree only.
"""

from TokenSim.comm.collective_types import ALGORITHMS, CollectiveEstimate, _rounds
from TokenSim.comm.traffic_types import ring_volume
from TokenSim.errors import ConfigurationError


def flat_collective(
    operation, participants, input_bytes, link, *, algorithm="ring", dtype=None
):
    if algorithm not in ALGORITHMS:
        raise ConfigurationError(f"unsupported flat collective algorithm {algorithm!r}")
    volume = ring_volume(operation, participants, input_bytes, dtype)
    # The ordinary round model takes a complete buffer for AllGather.
    # AF queries instead provide the local input shard, as NCCL does.
    full_bytes = (
        volume.output_bytes if operation == "all_gather" else volume.input_bytes
    )
    candidates = ("ring", "tree") if algorithm == "auto" else (algorithm,)
    estimates = []
    for candidate in candidates:
        if participants <= 1 or not input_bytes:
            rounds, wire_bytes = 0, 0.0
        elif candidate == "ring":
            # Preserve the established byte arithmetic and default results.
            rounds, wire_bytes = volume.rounds, volume.send_bytes_per_rank
        else:
            rounds, wire_bytes = _rounds(operation, candidate, participants, full_bytes)
        service_us = (
            rounds * link.latency_us
            + wire_bytes / link.bandwidth_per_direction_bytes_per_s * 1e6
        )
        estimates.append((service_us, candidate, rounds, wire_bytes))
    service_us, selected, rounds, wire_bytes = min(estimates, key=lambda row: row[0])
    details = dict(
        algorithm=selected,
        operation=operation,
        logical_bytes=input_bytes,
        input_bytes_per_rank=volume.input_bytes,
        output_bytes_per_rank=volume.output_bytes,
        rounds=rounds,
        wire_bytes=wire_bytes,
        wire_bytes_scope="per_rank",
        group_size=participants,
        link=link.link_id,
    )
    if algorithm != "ring":
        details.update(requested_algorithm=algorithm, algorithm_candidates=candidates)
    return CollectiveEstimate(
        service_us,
        operation,
        participants,
        input_bytes,
        "analytical",
        f"link:{link.link_id}:{link.source_id}",
        detail=details,
    )
