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
    kernel = ReplayKernel()
    try:
        identifiers = [
            kernel.add(item.nodes) for item in (profile, *additional_profiles)
        ]
        pending = {
            kernel.submit(
                identifiers[client % len(identifiers)], client * initial_spacing_s
            ): client
            for client in range(cohorts)
        }
        next_issue = cohorts * initial_spacing_s
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
            pending[kernel.submit(identifiers[client % len(identifiers)], issue_at)] = (
                client
            )
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
        if len(identifiers) > 1:
            result["programs"] = len(identifiers)
            result["program_assignment"] = {
                client: client % len(identifiers) for client in range(cohorts)
            }
        return result
    finally:
        kernel.close()
