"""SimPy execution and output-token metrics for four LPX serving paths."""

from dataclasses import replace
import math

import simpy

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.operators import Operators, Task


def expected_committed_tokens(draft_tokens, acceptance_probability):
    integer(draft_tokens, "draft_tokens", minimum=1)
    number(acceptance_probability, "acceptance_probability", maximum=1)
    # An autoregressive rejection stops the accepted prefix; verification
    # commits one correction/bonus token in addition to that prefix.
    return sum(acceptance_probability**i for i in range(draft_tokens + 1))


def draft_model(target):
    return replace(
        target,
        model_id="gpt-oss-draft-proxy",
        num_layers=4,
        attention_windows=target.attention_windows[:4],
        moe=replace(target.moe, num_experts=128, num_moe_layers=4),
        sources={
            "assumption-draft": {
                "grade": "D",
                "notes": "Four-layer GPT-OSS geometry with 128 experts; assumed acceptance independent of quality.",
            }
        },
    )


def averaged_forward(operators, layout, pool, workload, batch, query_len=1):
    remaining = max(1, workload.output_tokens - 1)
    bins = min(remaining, workload.context_samples)
    samples = []
    weights = []
    for i in range(bins):
        start = remaining * i // bins
        end = remaining * (i + 1) // bins
        position = (start + end - 1) // 2
        context = workload.cached_tokens + workload.new_tokens + position + query_len
        samples.append(
            operators.forward(
                layout,
                pool,
                batch,
                query_len,
                context,
                workload.seed + position,
                logits_rows=batch * query_len,
            )
        )
        weights.append(end - start)
    tasks = []
    if any(len(sample) != len(samples[0]) for sample in samples):
        raise ConfigurationError("sampled forwards have inconsistent task structure")
    for columns in zip(*samples):
        tasks.append(
            Task(
                columns[0].name,
                columns[0].pool,
                sum(task.seconds * weight for task, weight in zip(columns, weights))
                / sum(weights),
            )
        )
    return tasks


def point(
    hardware,
    target,
    workload,
    case,
    batch,
    *,
    power_overrides=None,
    facility_watts=0.0,
    trace=False,
):
    integer(batch, "batch", minimum=1)
    number(facility_watts, "facility_watts")
    peak_context = (
        workload.cached_tokens
        + workload.new_tokens
        + workload.output_tokens
        + workload.draft_tokens
    )
    if peak_context > target.max_position_embeddings:
        raise ConfigurationError("workload exceeds target model context limit")
    operators = Operators(hardware, target)
    prefill_layout = case.prefill or case.decode
    memory = {
        "decode": operators.resident_bytes(
            case.decode,
            batch,
            peak_context,
            workload.draft_tokens + 1 if case.mode == "draft" else 1,
        ),
        "prefill": operators.resident_bytes(
            prefill_layout,
            batch,
            peak_context,
            min(workload.prefill_chunk, workload.new_tokens),
        ),
    }
    draft = None
    if case.draft:
        draft = Operators(hardware, draft_model(target))
        memory["draft"] = draft.resident_bytes(case.draft, batch, peak_context, 1)
    inventory = case.inventory()
    power = facility_watts
    for device, count in inventory.items():
        watts = (power_overrides or {}).get(device, hardware.device(device).tdp_w)
        number(watts, f"{device} power", positive=True)
        power += count * watts
    env = simpy.Environment()
    resources = {}
    events = []
    service_totals = {}
    timings = {}

    def run_tasks(tasks, capture=False):
        for task in tasks:
            resource = resources.setdefault(task.pool, simpy.Resource(env, capacity=1))
            with resource.request() as request:
                queued_at = env.now
                yield request
                started = env.now
                yield env.timeout(task.seconds)
                service_totals[task.pool] = (
                    service_totals.get(task.pool, 0.0) + task.seconds
                )
                if capture:
                    events.append(
                        {
                            "name": task.name,
                            "pool": task.pool,
                            "start_s": started,
                            "end_s": env.now,
                            "queue_s": started - queued_at,
                        }
                    )

    def execute():
        prefix = workload.cached_tokens
        left = workload.new_tokens
        pool = "prefill" if case.prefill else "target"
        while left:
            query = min(workload.prefill_chunk, left)
            prefix += query
            left -= query
            tasks = operators.forward(
                prefill_layout,
                pool,
                batch,
                query,
                prefix,
                workload.seed,
                logits=left == 0,
            )
            yield from run_tasks(tasks, trace and left == 0)
        if case.prefill:
            # Copy the full retained cache once, including the warm prefix.
            kv_bytes = (
                batch * prefix * target.kv_elements_per_token() * target.num_layers * 2
            )
            if case.mode == "pd":
                copy = (
                    hardware.link("ethernet1600Gb").analytical_transfer_us(kv_bytes)
                    / 1e6
                )
            else:
                copy = hardware.link("nvlink6").analytical_transfer_us(kv_bytes) / 1e6
            timings["kv_handoff_bytes"] = kv_bytes
            yield from run_tasks([Task("kv_handoff", "handoff", copy)], trace)
        if draft:
            # Each model owns its own KV; draft prefill is paid separately.
            # Warm cached prefix creation is excluded for both models.
            prefix = workload.cached_tokens
            left = workload.new_tokens
            while left:
                query = min(workload.prefill_chunk, left)
                prefix += query
                left -= query
                # Draft prefill also needs a workspace budget on each rank.
                draft.resident_bytes(case.draft, batch, peak_context, query)
                yield from run_tasks(
                    draft.forward(
                        case.draft,
                        "draft",
                        batch,
                        query,
                        prefix,
                        workload.seed,
                        logits=left == 0,
                    )
                )
        timings["ttft_s"] = env.now
        remaining = workload.output_tokens - 1
        decode_started = env.now
        if case.mode != "draft":
            tasks = averaged_forward(
                operators,
                case.decode,
                "decode" if case.prefill else "target",
                workload,
                batch,
            )
            for token in range(remaining):
                yield from run_tasks(tasks, trace and token == 0)
            timings["verification_width"] = 1
            timings["expected_tokens_per_cycle"] = 1.0
        else:
            width = workload.draft_tokens + 1
            committed = expected_committed_tokens(
                workload.draft_tokens, workload.acceptance_probability
            )
            verify = averaged_forward(
                operators, case.decode, "target", workload, batch, width
            )
            drafting = averaged_forward(draft, case.draft, "draft", workload, batch)
            control_bytes = batch * (workload.draft_tokens * 4 + 4)
            link = hardware.link("ethernet1600Gb")
            control = link.analytical_transfer_us(control_bytes) / 1e6
            timings["verification_width"] = width
            timings["expected_tokens_per_cycle"] = committed
            timings["draft_control_bytes_per_cycle"] = control_bytes * 2
            cycles = remaining / committed
            for cycle in range(math.ceil(cycles)):
                # Expected-value projection uses a fractional terminal cycle.
                fraction = min(1.0, cycles - cycle)
                tasks = (
                    drafting * workload.draft_tokens
                    + [Task("draft_tokens", "draft_control_tx", control)]
                    + verify
                    + [Task("reject_positions", "draft_control_rx", control)]
                )
                scaled = [
                    replace(task, seconds=task.seconds * fraction) for task in tasks
                ]
                yield from run_tasks(scaled, trace and cycle == 0)
        timings["decode_s"] = env.now - decode_started
        timings["turn_s"] = env.now

    env.process(execute())
    env.run()
    output_speed = (
        (workload.output_tokens - 1) / timings["decode_s"]
        if workload.output_tokens > 1
        else 0.0
    )
    throughput = batch * workload.output_tokens / timings["turn_s"]
    return {
        "case": case.name,
        "mode": case.mode,
        "batch": batch,
        "status": "ok",
        "tokens_per_s_per_user": output_speed,
        "output_tokens_per_s": throughput,
        "power_watts": power,
        "tokens_per_s_per_mw": throughput * 1e6 / power,
        "inventory": inventory,
        "resident_memory_bytes_per_rank": memory,
        "timings": timings,
        "pool_service_s": service_totals,
        "trace": events,
        "operator_provenance": operators.provider.describe(),
        "precision": {
            "weight": target.dtype,
            "activation": target.activation_dtype,
            "kv": target.kv_cache_dtype,
        },
    }


def pareto(rows):
    points = sorted(
        (row for row in rows if row["status"] == "ok"),
        key=lambda row: (-row["tokens_per_s_per_user"], -row["tokens_per_s_per_mw"]),
    )
    frontier, best = [], -1.0
    for row in points:
        if row["tokens_per_s_per_mw"] > best:
            frontier.append(row)
            best = row["tokens_per_s_per_mw"]
    return list(reversed(frontier))
