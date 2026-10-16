"""Validate persisted request/token accounting and compute finite-run metrics."""

from collections import Counter
import math
from TokenSim.errors import SimulationStateError


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def validate_and_summarize(args, qps, arrivals, requests, tokens, batches, queues):
    if not queues or any(
        (queues[-1][key] for key in ("active_users", "pending", "runnable"))
    ):
        raise SimulationStateError("the final user queues must be empty")
    complete = [r for r in requests if r["status"] == "complete"]
    dropped = [r for r in requests if r["status"] == "dropped"]
    if not len(requests) == len(arrivals) == len(complete) + len(dropped):
        raise SimulationStateError(
            "record validation failed: len(requests) == len(arrivals) == len(complete) + len(dropped)"
        )
    counts = Counter((t["request_id"] for t in tokens))
    if not len(tokens) == len(complete) * args.output_tokens:
        raise SimulationStateError(
            "record validation failed: len(tokens) == len(complete) * args.output_tokens"
        )
    by_request = {r["request_id"]: r for r in complete}
    for r in complete:
        if not counts[r["request_id"]] == args.output_tokens:
            raise SimulationStateError(
                "record validation failed: counts[r['request_id']] == args.output_tokens"
            )
        if (
            not r["arrival_s"]
            <= r["eligible_s"]
            <= r["first_admitted_s"]
            <= r["first_token_s"]
            <= r["completion_s"]
        ):
            raise SimulationStateError(
                "record validation failed: r['arrival_s'] <= r['eligible_s'] <= r['first_admitted_s'] <= r['first_token_s'] <= r['completion_s']"
            )
    previous = {}
    for token in tokens:
        r = by_request[token["request_id"]]
        if not token["context_len"] == args.context + token["token_index"]:
            raise SimulationStateError(
                "record validation failed: token['context_len'] == args.context + token['token_index']"
            )
        if (
            not r["arrival_s"]
            <= token["ready_s"]
            <= token["admitted_s"]
            <= token["completion_s"]
        ):
            raise SimulationStateError(
                "record validation failed: r['arrival_s'] <= token['ready_s'] <= token['admitted_s'] <= token['completion_s']"
            )
        key = token["request_id"]
        if not token["token_index"] == previous.get(key, (-1, 0))[0] + 1:
            raise SimulationStateError(
                "record validation failed: token['token_index'] == previous.get(key, (-1, 0))[0] + 1"
            )
        if not token["ready_s"] >= previous.get(key, (-1, 0))[1] - 1e-09:
            raise SimulationStateError(
                "record validation failed: token['ready_s'] >= previous.get(key, (-1, 0))[1] - 1e-09"
            )
        previous[key] = (token["token_index"], token["completion_s"])
    for user in range(args.users):
        rows = sorted(
            (r for r in complete if r["user"] == user), key=lambda r: r["arrival_s"]
        )
        for left, right in zip(rows, rows[1:]):
            if not right["first_admitted_s"] >= left["completion_s"] - 1e-09:
                raise SimulationStateError(
                    "record validation failed: right['first_admitted_s'] >= left['completion_s'] - 1e-09"
                )
    if not all(
        (
            b["batch_size"] <= args.max_batch
            and len(set(b["users"])) == b["batch_size"]
            for b in batches
        )
    ):
        raise SimulationStateError(
            "record validation failed: all((b['batch_size'] <= args.max_batch and len(set(b['users'])) == b['batch_size'] for b in batches))"
        )
    if not max((q["active_users"] for q in queues)) <= args.users:
        raise SimulationStateError(
            "record validation failed: max((q['active_users'] for q in queues)) <= args.users"
        )
    end = max((r["completion_s"] for r in complete))
    last_arrival = arrivals[-1]["arrival_s"]
    arrival_tokens = sum((t["completion_s"] <= last_arrival for t in tokens))
    last_complete = sum((r["completion_s"] <= last_arrival for r in complete))
    latencies = [(r["completion_s"] - r["arrival_s"]) * 1000 for r in complete]
    ttfts = [(r["first_token_s"] - r["arrival_s"]) * 1000 for r in complete]
    service_ttfts = [
        (r["first_token_s"] - r["first_admitted_s"]) * 1000 for r in complete
    ]
    tbts = [t["itl_ms"] for t in tokens if t["itl_ms"] is not None]
    waits = [t["scheduler_wait_ms"] for t in tokens]
    for left, right in zip(batches, batches[1:]):
        if right["start_s"] < left["end_s"] - 1e-12:
            raise SimulationStateError("decode iterations must be serialized")
    if getattr(args, "unique_users", False) and len(
        {r["user"] for r in requests}
    ) != len(requests):
        raise SimulationStateError("unique-user requests must use distinct user IDs")
    for request in complete:
        rows = [
            token for token in tokens if token["request_id"] == request["request_id"]
        ]
        if (
            rows[0]["completion_s"] != request["first_token_s"]
            or rows[-1]["completion_s"] != request["completion_s"]
        ):
            raise SimulationStateError(
                "request endpoints do not match token completions"
            )
        for left, right in zip(rows, rows[1:]):
            expected = (right["completion_s"] - left["completion_s"]) * 1000
            if right["itl_ms"] is None or not math.isclose(
                right["itl_ms"], expected, abs_tol=1e-08
            ):
                raise SimulationStateError(
                    "TBT must be computed from adjacent output tokens"
                )
    return dict(
        qps=qps,
        users=args.users,
        observed_users=len({r["user"] for r in requests}),
        a_tp=args.a_tp,
        f_ep=args.f_ep,
        max_batch=args.max_batch,
        af_batch_size=args.af_batch_size,
        requests=len(requests),
        completed_requests=len(complete),
        dropped_requests=len(dropped),
        output_tokens_per_request=args.output_tokens,
        total_output_tokens=len(tokens),
        realized_arrival_qps=len(arrivals) / last_arrival,
        offered_token_rate=qps * args.output_tokens,
        arrival_window_s=last_arrival,
        drained_end_s=end,
        drain_s=end - last_arrival,
        total_tps=len(tokens) / end,
        tps_during_arrivals=arrival_tokens / last_arrival,
        completed_qps=len(complete) / end,
        mean_user_tps=len(tokens) / end / args.users,
        unfinished_at_last_arrival=len(complete) - last_complete,
        max_pending=max((q["pending"] for q in queues)),
        max_runnable=max((q["runnable"] for q in queues)),
        mean_tbt_ms=sum(tbts) / len(tbts) if tbts else None,
        p95_tbt_ms=percentile(tbts, 0.95) if tbts else None,
        max_tbt_ms=max(tbts) if tbts else None,
        mean_scheduler_wait_ms=sum(waits) / len(waits),
        p95_scheduler_wait_ms=percentile(waits, 0.95),
        max_active_users=max((q["active_users"] for q in queues)),
        mean_batch=len(tokens) / len(batches),
        max_actual_batch=max((b["batch_size"] for b in batches)),
        mean_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=percentile(latencies, 0.95),
        max_latency_ms=max(latencies),
        mean_ttft_ms=sum(ttfts) / len(ttfts),
        p95_ttft_ms=percentile(ttfts, 0.95),
        max_ttft_ms=max(ttfts),
        mean_service_ttft_ms=sum(service_ttfts) / len(service_ttfts),
        p95_service_ttft_ms=percentile(service_ttfts, 0.95),
    )
