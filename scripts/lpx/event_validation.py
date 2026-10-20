"""Reconstruct token totals, per-user time and the closing boundary independently."""

from collections import Counter, defaultdict
import math

from TokenSim.config.validation import boolean, integer, number
from scripts.lpx.validation_checks import equal, require


def validate_accounting_events(row, workload):
    names = ("token_events", "decode_intervals")
    if not any(name in row for name in names):
        return False
    require(all(name in row for name in names), "incomplete token-accounting evidence")
    metric, timing = row["steady_state"], row["timings"]
    start, end = metric["start_s"], metric["end_s"]
    finish = end + timing["drain_s"]
    clients, micro = row["clients"], row["micro_batch"]

    def cohort_size(event):
        integer(event["cohort"], "accounting cohort")
        integer(event["round"], "accounting round")
        size = min(micro, clients - event["cohort"] * micro)
        require(size > 0, "accounting cohort range")
        boolean(event["warmup"], "warm-up label")
        return size

    output, decode, token_keys = 0, 0, Counter()
    turn_output, turn_decode = Counter(), Counter()
    previous = 0.0
    for event in row["token_events"]:
        size = cohort_size(event)
        integer(event["tokens"], "completed token count", minimum=1)
        number(event["time"], "token completion", minimum=previous)
        previous = event["time"]
        require(event["time"] <= finish + 1e-10, "token after drain completion")
        require(event["phase"] in {"prefill", "decode"}, "token phase")
        require(event["tokens"] % size == 0, "cohort token multiplicity")
        turn_key = (event["cohort"], event["round"])
        turn_output[turn_key] += event["tokens"]
        if event["phase"] == "prefill":
            require(event["tokens"] == size, "prefill produces one output per request")
        else:
            turn_decode[turn_key] += event["tokens"]
        if event["phase"] == "decode":
            token_keys[
                (
                    event["time"],
                    event["cohort"],
                    event["round"],
                    event["tokens"],
                    event["warmup"],
                )
            ] += 1
        if not event["warmup"] and start <= event["time"] <= end:
            output += event["tokens"]
            if event["phase"] == "decode":
                decode += event["tokens"]
    equal(metric["output_tokens"], output, "event-derived output count")
    equal(metric["decode_tokens"], decode, "event-derived decode count")
    for turn in row["turns"]:
        if turn["completed"]:
            size = min(micro, clients - turn["cohort"] * micro)
            key = (turn["cohort"], turn["round"])
            equal(
                turn_output[key],
                workload.output_tokens * size,
                "complete-turn event count",
            )
            equal(
                turn_decode[key],
                (workload.output_tokens - 1) * size,
                "complete-turn decode event count",
            )
    interval_keys, user_seconds = Counter(), 0.0
    for interval in row["decode_intervals"]:
        size = cohort_size(interval)
        require(interval["clients"] == size, "decode interval cohort size")
        integer(interval["tokens"], "decode interval tokens", minimum=1)
        number(interval["start"], "decode start")
        number(interval["end"], "decode end", minimum=interval["start"])
        interval_keys[
            (
                interval["end"],
                interval["cohort"],
                interval["round"],
                interval["tokens"] * size,
                interval["warmup"],
            )
        ] += 1
        if not interval["warmup"] and start <= interval["end"] <= end:
            user_seconds += (interval["end"] - interval["start"]) * size
    require(token_keys == interval_keys, "decode tokens disagree with pass intervals")
    number(user_seconds, "measured per-user decode time", positive=True)
    equal(
        row.get("decode_weighted_tokens_per_s_per_user", row["tokens_per_s_per_user"]),
        decode / user_seconds,
        "event-derived interactivity",
    )
    validate_closing_boundary(row)
    return True


def validate_closing_boundary(row):
    metric, timing = row["steady_state"], row["timings"]
    minimum = timing["minimum_complete_turns_per_cohort"]
    integer(minimum, "minimum complete turns")
    ends = defaultdict(list)
    for turn in row["turns"]:
        if not turn["warmup"] and turn["completed"]:
            ends[turn["cohort"]].append(turn["end"])
    required = [metric["start_s"] + timing["minimum_measurement_s"]]
    if minimum:
        for cohort in range(math.ceil(row["clients"] / row["micro_batch"])):
            require(len(ends[cohort]) >= minimum, "missing closing turn")
            required.append(sorted(ends[cohort])[minimum - 1])
    equal(metric["end_s"], max(required), "measurement end changed after closure")
