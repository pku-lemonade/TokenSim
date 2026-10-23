"""Attribute proven peer-arrival waits while preserving raw device residence."""

from collections import defaultdict

from TokenSim.profiler.records import union_duration


def common_clock(inputs):
    if len(inputs) <= 1:
        return dict(available=True, evidence="single_input_clock")
    witnesses = []
    for entry in inputs:
        manifest = entry["manifest"]
        metadata = entry.get("metadata", {})
        identities = manifest.get("worker_identity", [])
        if (
            manifest.get("measurement_protocol") != "paired_structure_gpu_timing_v1"
            or not identities
        ):
            return dict(available=False, reason="no_shared_clock_collection_evidence")
        for identity in identities:
            witness = tuple(
                identity.get(key) for key in ("hostname", "boot_id", "time_namespace")
            )
            if not all(isinstance(value, str) and value for value in witness):
                return dict(available=False, reason="incomplete_worker_clock_identity")
            if metadata.get("host_name") != witness[0]:
                return dict(available=False, reason="profiler_host_differs_from_worker")
            base = metadata.get("baseTimeNanoseconds")
            if type(base) is not int or base <= 0:
                return dict(available=False, reason="missing_profiler_clock_basis")
            witnesses.append((*witness, base))
    if len(set(witnesses)) != 1:
        return dict(available=False, reason="worker_or_profiler_clock_domains_differ")
    return dict(
        available=True, evidence="same_host_boot_time_namespace_and_kineto_basis"
    )


def analyze_collectives(pairs, observed, rank_map=None):
    clock = common_clock(observed.inputs)
    groups = defaultdict(dict)
    unresolved = []
    rank_map = rank_map or {}
    for left, right in pairs:
        calls = right.details.get("collective_calls", [])
        right.details["collective_wait_status"] = (
            "not_applicable" if not calls else "unresolved"
        )
        if not calls:
            continue
        specifications = left.details.get("collective_groups", [])
        if left.details.get("participants"):
            specifications = [
                dict(
                    operation=left.operation, participants=left.details["participants"]
                )
            ]
        for ordinal, call in enumerate(calls):
            if call.get("wait_model") != "peer_start_barrier":
                unresolved.append(
                    dict(id=right.id, reason="unproven_collective_start_barrier")
                )
                continue
            candidates = [
                spec
                for spec in specifications
                if spec["operation"] == call["operation"]
            ]
            if not clock["available"] or len(candidates) != 1:
                unresolved.append(
                    dict(id=right.id, reason="unproven_clock_or_collective_group")
                )
                continue
            participants = tuple(
                sorted(str(rank) for rank in candidates[0]["participants"])
            )
            rank = str(rank_map.get(right.rank, right.rank))
            key = (
                participants,
                left.forward,
                left.layer,
                left.operation,
                call["operation"],
                ordinal,
            )
            if rank not in participants or rank in groups[key]:
                unresolved.append(
                    dict(id=right.id, reason="duplicate_or_unexpected_collective_rank")
                )
                continue
            groups[key][rank] = right, call
    analyzed = 0
    for key, members in groups.items():
        participants = key[0]
        if set(members) != set(participants):
            unresolved.append(
                dict(participants=participants, reason="missing_collective_rank")
            )
            continue
        arrival = max(call["start_us"] for _, call in members.values())
        if any(call["end_us"] < arrival for _, call in members.values()):
            unresolved.append(
                dict(
                    participants=participants,
                    reason="collective_intervals_do_not_overlap",
                )
            )
            continue
        for span, call in members.values():
            call.update(
                peer_arrival_us=arrival,
                peer_wait_us=arrival - call["start_us"],
                after_peer_arrival_us=call["end_us"] - arrival,
            )
            analyzed += 1
    for span in observed.spans:
        calls = span.details.get("collective_calls", [])
        if not calls:
            span.details["device_work_intervals"] = span.details.get(
                "service_intervals", [(span.start_us, span.end_us)]
            )
            continue
        if any("peer_arrival_us" not in call for call in calls):
            span.details["collective_wait_status"] = "unresolved"
            continue
        span.details["collective_wait_status"] = "attributed_peer_arrival"
        span.details["collective_wait_intervals"] = [
            (call["start_us"], call["peer_arrival_us"])
            for call in calls
            if call["peer_wait_us"] > 0
        ]
        arrivals = {
            (call["start_us"], call["end_us"]): call["peer_arrival_us"]
            for call in calls
        }
        span.details["device_work_intervals"] = [
            (arrivals.get((start, end), start), end)
            for start, end in span.details["service_intervals"]
        ]
    return dict(
        shared_clock=clock,
        analyzed_calls=analyzed,
        unresolved=unresolved,
        method="observed_time_until_last_matching_peer_kernel_start",
        limitation="After-arrival residence can still include barrier and execution overhead; it is not pure network payload time.",
    )


def service_metrics(left, right):
    details = right.details
    calls = details.get("collective_calls", [])
    work = details.get("device_work_intervals")
    available = work is not None
    measured = union_duration(work) if available else None
    simulated = left.busy_us
    error = simulated - measured if available else None
    return dict(
        simulated_device_service_us=simulated,
        profiled_device_service_us=measured,
        device_service_error_us=error,
        relative_device_service_error=error / measured if measured else None,
        profiled_internal_gap_us=max(0, right.duration_us - right.busy_us),
        profiled_collective_residence_us=union_duration(
            [(call["start_us"], call["end_us"]) for call in calls]
        ),
        profiled_peer_wait_us=(
            union_duration(details.get("collective_wait_intervals", []))
            if available
            else None
        ),
        device_service_unavailable_reason=(
            None if available else "collective_peer_wait_unresolved"
        ),
    )
