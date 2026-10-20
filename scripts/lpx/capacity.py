"""Observed near-zero admission-wait capacity; every probe remains reviewable."""

from TokenSim.config.validation import integer, number


def classify(row, isolated_latency, *, wait_fraction=0.01):
    if row["status"] != "ok":
        return dict(passed=False, reason=row.get("reason", "infeasible"))
    arrivals, requests = row["arrivals"], row["request_statistics"]
    budget = max(50e-6, wait_fraction * isolated_latency)
    wait = arrivals["admission_wait_s"]["p95"]
    # Count unfinished requests at the boundary explicitly. Arrival and
    # completion rates otherwise differ even in a stable short window.
    unfinished = arrivals["admissions"] - requests["completed_requests"]
    balance = (
        arrivals["arrivals"]
        - requests["completed_requests"]
        - unfinished
        - arrivals["pending_at_end"]
    )
    checks = dict(
        requests_completed=requests["completed_requests"] > 0,
        boundary_balanced=balance == 0,
        wait_budget=wait is not None and wait <= budget,
        no_pending_arrivals=arrivals["pending_at_end"] == 0,
        no_backlog_growth=arrivals["backlog_growth_per_s"] <= 0,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        admission_wait_budget_s=budget,
        incomplete_at_boundary=unfinished,
    )


def search(
    evaluate,
    isolated_latency,
    *,
    resolution=0.02,
    max_probes=24,
    wait_fraction=0.01,
    maximum_multiple=64,
):
    number(isolated_latency, "isolated request latency", positive=True)
    number(resolution, "capacity resolution", positive=True, maximum=0.5)
    integer(max_probes, "capacity probes", minimum=2)
    number(maximum_multiple, "maximum QPS multiple", positive=True)
    probes = []

    def probe(qps, multiplier=1):
        result = evaluate(qps, multiplier)
        verdict = classify(result, isolated_latency, wait_fraction=wait_fraction)
        probes.append(
            dict(qps=qps, window_multiplier=multiplier, verdict=verdict, result=result)
        )
        return verdict["passed"]

    qps, lower, upper = 0.05 / isolated_latency, None, None
    ceiling = maximum_multiple / isolated_latency
    while len(probes) < max_probes - 1 and qps <= ceiling:
        if probe(qps):
            lower = qps
        else:
            upper = qps
            break
        qps *= 2
    # This refinement assumes local monotonicity. The output states the
    # searched interval and does not claim a global optimum over batching.
    while lower is not None and upper is not None and len(probes) < max_probes - 1:
        if upper / lower - 1 <= resolution:
            break
        middle = (lower + upper) / 2
        if probe(middle):
            lower = middle
        else:
            upper = middle
    confirmed = lower is not None and probe(lower, 2)
    return dict(
        isolated_latency_s=isolated_latency,
        maximum_passing_qps=lower,
        recommended_qps=lower * 0.95 if confirmed else None,
        doubled_window_passed=confirmed,
        failed_upper_qps=upper,
        relative_bracket=upper / lower - 1 if lower and upper else None,
        searched_ceiling_qps=ceiling,
        probes=probes,
        scope="observed interval; local monotonicity assumed; no global batching optimum claimed",
    )
