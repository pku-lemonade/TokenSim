"""Finite-window closed-loop replay with explicit request phasing and pacing."""

from collections import Counter

import numpy as np

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.replay_engine import ReplayKernel


def replay_cohorts(
    profile,
    *,
    resident_requests,
    batch=1,
    warmup_rounds=8,
    measured_rounds=32,
    initial_spacing_s=0.0,
    issue_interval_s=0.0,
    additional_profiles=(),
    program_assignment=None,
    initial_offsets_s=None,
):
    integer(resident_requests, "replay resident requests", minimum=1)
    integer(batch, "replay batch size", minimum=1)
    integer(warmup_rounds, "replay warm-up rounds", minimum=1)
    integer(measured_rounds, "replay measured rounds", minimum=2)
    number(initial_spacing_s, "replay initial spacing")
    number(issue_interval_s, "replay issue interval")
    if resident_requests % batch:
        raise ConfigurationError("MFU replay requires complete fixed-width cohorts")
    cohorts = resident_requests // batch
    profiles = (profile, *additional_profiles)
    try:
        assignment = (
            tuple(client % len(profiles) for client in range(cohorts))
            if program_assignment is None
            else tuple(program_assignment)
        )
    except TypeError as exc:
        raise ConfigurationError("MFU assignment requires an iterable") from exc
    if len(assignment) != cohorts:
        raise ConfigurationError("MFU assignment must cover every cohort")
    for index in assignment:
        integer(index, "MFU assigned program")
        if index >= len(profiles):
            raise ConfigurationError("MFU assigned program is unavailable")
    try:
        offsets = (
            tuple(client * initial_spacing_s for client in range(cohorts))
            if initial_offsets_s is None
            else tuple(initial_offsets_s)
        )
    except TypeError as exc:
        raise ConfigurationError("MFU offsets require an iterable") from exc
    if len(offsets) != cohorts or (initial_offsets_s is not None and initial_spacing_s):
        raise ConfigurationError(
            "MFU offsets require one value per cohort and zero spacing"
        )
    for offset in offsets:
        number(offset, "MFU initial cohort offset")
    kernel = ReplayKernel()
    try:
        identifiers = [kernel.add(item.nodes) for item in profiles]
        pending = {
            kernel.submit(identifiers[assignment[client]], offsets[client]): client
            for client in range(cohorts)
        }
        next_issue = (
            cohorts * initial_spacing_s
            if initial_offsets_s is None
            else max(offsets) + issue_interval_s
        )
        previous = dict.fromkeys(range(cohorts), 0.0)
        count = Counter()
        intervals = []
        warm_count = warmup_rounds * cohorts
        measured_count = measured_rounds * cohorts
        half_count = measured_count // 2
        started = middle = None
        for step in range(warm_count + measured_count):
            done = kernel.advance()
            if done is None:
                raise SimulationStateError("MFU replay exhausted before measurement")
            client = pending.pop(done.id)
            interval = done.end - previous[client]
            previous[client] = done.end
            if step == warm_count - 1:
                started = done.end
            if step >= warm_count:
                count[client] += 1
                intervals.append(interval)
            if step == warm_count + half_count - 1:
                middle = done.end
            issue_at = max(done.end, next_issue) if issue_interval_s else done.end
            next_issue = issue_at + issue_interval_s
            pending[kernel.submit(identifiers[assignment[client]], issue_at)] = client
        elapsed = done.end - started
        tokens = measured_count * batch
        earlier = half_count * batch / (middle - started)
        later = (measured_count - half_count) * batch / (done.end - middle)
        result = dict(
            resident_requests=resident_requests,
            batch=batch,
            cohorts=cohorts,
            warmup_completions=warm_count,
            measured_completions=measured_count,
            measured_tokens=tokens,
            measured_seconds=elapsed,
            aggregate_tps=tokens / elapsed,
            mean_per_request_tps=tokens / elapsed / resident_requests,
            per_cohort_completions=dict(count),
            per_cohort_per_request_tps={
                client: count[client] / elapsed for client in range(cohorts)
            },
            mean_tpot_s=float(np.mean(intervals)),
            p50_tpot_s=float(np.median(intervals)),
            p99_tpot_s=float(np.percentile(intervals, 99)),
            initial_spacing_s=initial_spacing_s,
            issue_interval_s=issue_interval_s,
            convergence=dict(
                earlier_half_tps=earlier,
                later_half_tps=later,
                relative_change=later / earlier - 1,
            ),
            scope="Fixed-context operator-DAG replay; compiler scheduling and buffer lifetime safety unverified.",
        )
        if len(identifiers) > 1 or program_assignment is not None:
            result["programs"] = len(identifiers)
            result["program_assignment"] = {
                client: assignment[client] for client in range(cohorts)
            }
        if initial_offsets_s is not None:
            result["initial_offsets_s"] = list(offsets)
        return result
    finally:
        kernel.close()
