"""Steady-state token counts and resource utilization from native AF events."""

from collections import defaultdict

from TokenSim.config.validation import number


def steady_metrics(events, traces, *, start, end, capacities=None):
    number(start, "measurement start")
    number(end, "measurement end", minimum=start, positive=True)
    if end == start:
        raise ValueError("steady-state measurement must have positive duration")
    output = sum(row["tokens"] for row in events if start <= row["time"] < end)
    decode = sum(
        row["tokens"]
        for row in events
        if start <= row["time"] < end and row["phase"] == "decode"
    )
    busy, queue, counts = defaultdict(float), defaultdict(float), defaultdict(int)
    for event in traces:
        queued = max(0.0, min(end, event.start_at) - max(start, event.ready_at))
        queue[event.resource_id] += queued
        overlap = max(0.0, min(end, event.end_at) - max(start, event.start_at))
        if not overlap:
            continue
        for resource in event.resource_ids or (event.resource_id,):
            busy[resource] += overlap
            counts[resource] += 1
    seconds = end - start
    return {
        "start_s": start,
        "end_s": end,
        "duration_s": seconds,
        "output_tokens": output,
        "decode_tokens": decode,
        "output_tokens_per_s": output / seconds,
        "decode_tokens_per_s": decode / seconds,
        "resource_utilization": {
            resource: value / (seconds * (capacities or {}).get(resource, 1))
            for resource, value in busy.items()
        },
        "resource_queue_s": dict(queue),
        "resource_operation_counts": dict(counts),
    }
