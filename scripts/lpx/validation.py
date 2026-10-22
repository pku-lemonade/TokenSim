"""Audit physical inventories, serving units and exported native frontiers."""

from collections import Counter
import math

from TokenSim.config.validation import integer, number
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.af.serving.timing import namespace_resource
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.event_validation import validate_accounting_events
from scripts.lpx.requests import request_metrics
from scripts.lpx.replay_validation import validate_replay_profiles
from scripts.lpx.validation_checks import equal as _equal, require as _require


def power_snapshot(hardware, overrides, devices=("rubin", "groq_lp30")):
    result = {}
    for name in devices:
        device = hardware.device(name)
        source = device.extra.get("power_source", "unspecified")
        result[name] = dict(
            watts=overrides.get(name, device.tdp_w),
            source_id="user_override" if name in overrides else source,
            grade=(
                "D"
                if name in overrides
                else device.sources.get(source, {}).get("grade", "D")
            ),
        )
    return result


def _case(raw):
    value = dict(raw)
    for key in ("decode", "prefill", "draft"):
        if value.get(key) is not None:
            value[key] = Layout(**value[key])
    return Case(**value)


def _handoff_unit(model, workload, case, mode):
    windows = (
        model.get("attention_windows")
        or [model.get("sliding_window", 0)] * model["num_layers"]
    )
    _require(len(windows) == model["num_layers"], "attention window count")
    positions = sum(
        (
            min(
                (
                    workload.new_tokens
                    if mode == "incremental"
                    else workload.cached_tokens + workload.new_tokens
                ),
                window,
            )
            if window
            else (
                workload.new_tokens
                if mode == "incremental"
                else workload.cached_tokens + workload.new_tokens
            )
        )
        for window in windows
    )
    heads = model["num_key_value_heads"]
    replicas = max(1, case.decode.a_tp // heads)
    return (
        positions
        * heads
        * model["head_dim"]
        * 2
        * dtype_size_bytes(model["kv_cache_dtype"], role="kv_cache_dtype")
        * replicas
    )


def validate_point(
    row,
    case,
    workload,
    *,
    a_replicas,
    watts,
    facility_watts,
    model,
    executor="ModelAFExecutor",
):
    _require(row["mode"] == case.mode, "case mode")
    integer(row["batch"], "result batch", minimum=1)
    if row["status"] == "infeasible":
        _require(bool(row.get("reason")), "capacity rejection needs a reason")
        return
    _require(row["status"] == "ok", "unknown point status")
    _require(row.get("native_executor") == executor, "point executor")
    clients, micro = row["clients"], row["micro_batch"]
    integer(clients, "clients", minimum=1)
    integer(micro, "micro-batch", minimum=1)
    _require(clients == row["batch"], "client/batch agreement")
    _require(
        all(
            layout.a_cp == 1
            for layout in (case.decode, case.prefill, case.draft)
            if layout
        ),
        "unsupported CP in feasible point",
    )
    inventory = case.inventory()
    inventory[case.decode.a_device] += case.decode.a_ranks * (a_replicas - 1)
    _require(row["inventory"] == inventory, "physical inventory")
    expected_power = facility_watts + sum(
        count * watts[device]["watts"] for device, count in inventory.items()
    )
    _equal(row["power_watts"], expected_power, "allocated processor/facility power")
    metric, timings = row["steady_state"], row["timings"]
    seconds = metric["end_s"] - metric["start_s"]
    number(seconds, "measurement duration", positive=True)
    _equal(metric["duration_s"], seconds, "wall-clock duration")
    _equal(timings["warmup_s"], metric["start_s"], "warm-up boundary")
    _require(
        timings["warmup_s"] + 1e-10 >= row.get("minimum_warmup_seconds", 0),
        "warm-up time minimum",
    )
    _equal(timings["measurement_s"], seconds, "measurement boundary")
    number(timings["drain_s"], "drain duration")
    _require(seconds + 1e-10 >= timings["minimum_measurement_s"], "measurement minimum")
    for key in ("output_tokens", "decode_tokens"):
        integer(metric[key], key)
        _equal(metric[f"{key}_per_s"], metric[key] / seconds, f"{key} rate")
    _require(
        0 < metric["decode_tokens"] <= metric["output_tokens"], "committed token counts"
    )
    _equal(
        row["output_tokens_per_s"], metric["output_tokens_per_s"], "output throughput"
    )
    _equal(
        row["tokens_per_s_per_mw"],
        metric["output_tokens_per_s"] * 1e6 / expected_power,
        "TPS/MW units",
    )
    number(row["tokens_per_s_per_user"], "interactivity", positive=True)
    _require(
        row.get("decode_weighted_tokens_per_s_per_user", row["tokens_per_s_per_user"])
        + 1e-9
        >= metric["decode_tokens_per_s"] / clients,
        "per-user decode-time bound",
    )
    completed = Counter()
    complete_decode_tokens, complete_user_seconds = 0, 0.0
    for turn in row["turns"]:
        if (
            not turn["warmup"]
            and turn["completed"]
            and turn["end"] <= metric["end_s"] + 1e-10
        ):
            _require(
                metric["start_s"]
                <= turn["start"]
                <= turn["end"]
                <= metric["end_s"] + 1e-10,
                "complete-turn interval",
            )
            completed[turn["cohort"]] += 1
            cohort_size = min(micro, clients - turn["cohort"] * micro)
            elapsed = turn["end"] - turn["start"]
            number(turn["ttft_s"], "time to first token", maximum=elapsed)
            complete_decode_tokens += cohort_size * (workload.output_tokens - 1)
            complete_user_seconds += cohort_size * (elapsed - turn["ttft_s"])
    if complete_decode_tokens == metric["decode_tokens"]:
        _equal(
            row.get(
                "decode_weighted_tokens_per_s_per_user", row["tokens_per_s_per_user"]
            ),
            complete_decode_tokens / complete_user_seconds,
            "complete-turn interactivity",
        )
    if "request_statistics" in row:
        expected = request_metrics(
            row["turns"],
            workload,
            clients,
            micro,
            {"start": metric["start_s"], "end": metric["end_s"]},
        )
        actual = row["request_statistics"]
        _require(actual == expected, "request-level statistics and timestamps")
        if expected["completed_requests"]:
            _equal(
                row["tokens_per_s_per_user"],
                expected["generation_tps"]["p50"],
                "request P50 interactivity",
            )
    minimum = timings["minimum_complete_turns_per_cohort"]
    for cohort in range(math.ceil(clients / micro)):
        _require(completed[cohort] >= minimum, "complete turns per cohort")
    _require(
        metric["output_tokens"] >= minimum * clients * workload.output_tokens,
        "complete-turn output count",
    )
    validate_accounting_events(row, workload)
    capacities = {}
    physical_compute_slots = 0
    for name, service in row["services"].items():
        physical_compute_slots += sum(
            key.startswith("worker:") for key in service["resource_capacities"]
        )
        capacities.update(
            {
                namespace_resource(name, key): value
                for key, value in service["resource_capacities"].items()
            }
        )
        memory = service["memory"]
        _require(memory["requests"] == clients, "resident request count")
        for worker, current in memory["resident_current_kv_bytes"].items():
            _require(
                0 <= current <= memory["reserved_peak_kv_bytes"][worker],
                "resident KV reservation",
            )
    _require(
        physical_compute_slots == sum(inventory.values()), "allocated compute slots"
    )
    for resource, utilization in metric["resource_utilization"].items():
        _require(resource in capacities, "undeclared utilized resource")
        number(utilization, "resource utilization")
        _require(utilization <= 1 + 1e-9, "resource capacity exceeded")
    for queued in metric["resource_queue_s"].values():
        number(queued, "resource queue delay")
    for event in (*row["handoff_events"], *row["control_events"]):
        integer(event["bytes"], "transfer bytes", minimum=1)
        _require(
            0
            <= event["ready_s"]
            <= event["start_s"]
            <= event["injection_end_s"]
            <= event["end_s"],
            "transport interval",
        )
    if case.prefill:
        total = sum(event["bytes"] for event in row["handoff_events"])
        unit = _handoff_unit(model, workload, case, row["handoff_mode"])
        copies = total / unit
        _equal(copies, round(copies), "complete window-aware KV handoff bytes")
        _require(copies >= minimum * clients, "missing KV handoffs")
    if case.mode == "draft":
        _require(bool(row["verification_events"]), "missing verification cycles")
        measured_commits = 0
        for cycle in row["verification_events"]:
            _require(
                cycle["verification_width"] == workload.draft_tokens + 1,
                "full verification width",
            )
            _require(
                0 <= cycle["accepted_prefix"] <= workload.draft_tokens,
                "accepted prefix",
            )
            _require(
                1 <= cycle["committed_tokens"] <= cycle["accepted_prefix"] + 1,
                "committed prefix",
            )
            _require(cycle["draft_catchup_tokens"] in (0, 1), "draft KV catch-up")
            integer(cycle["cohort"], "verification cohort")
            _require(cycle["cohort"] * micro < clients, "verification cohort range")
            number(cycle["time"], "verification completion")
            if (
                not cycle["warmup"]
                and metric["start_s"] <= cycle["time"] <= metric["end_s"]
            ):
                cohort_size = min(micro, clients - cycle["cohort"] * micro)
                measured_commits += cycle["committed_tokens"] * cohort_size
        _require(
            measured_commits == metric["decode_tokens"],
            "measured verification commits disagree with decode tokens",
        )


def _validate_grid(payload, cases, identities, expected_cases, expected_batches):
    if expected_cases is not None:
        expected = {case.name: case for case in expected_cases}
        _require(
            len(expected) == len(expected_cases) and cases == expected,
            "saved cases differ from requested configurations",
        )
    recorded = payload["run_settings"].get("client_counts")
    batches = expected_batches if expected_batches is not None else recorded
    _require({name for name, _ in identities} == set(cases), "missing declared case")
    if batches is None:
        return "not_recorded"
    for counts in (recorded, batches):
        if counts is None:
            continue
        _require(bool(counts), "empty client-count grid")
        for batch in counts:
            integer(batch, "expected client count", minimum=1)
        _require(len(set(counts)) == len(counts), "duplicate client counts")
    if recorded is not None and expected_batches is not None:
        _require(set(recorded) == set(batches), "requested client counts differ")
    expected = {(name, batch) for name in cases for batch in batches}
    _require(set(identities) == expected, "incomplete or unexpected experiment grid")
    return "external" if expected_batches is not None else "declared"


def validate_summary(
    payload,
    *,
    hardware=None,
    original_workload=False,
    expected_cases=None,
    expected_batches=None,
):
    replay = payload.get("version") == 3
    _require(
        payload.get("schema") == "tokensim.lpx_projection"
        and payload.get("version") in {2, 3},
        "summary schema",
    )
    _require(
        payload.get("executor")
        == ("sampled_native_dag_replay" if replay else "production_ModelAFExecutor"),
        "summary executor",
    )
    if replay:
        _require(
            payload["run_settings"]["execution_backend"] == "sampled-replay",
            "replay backend declaration",
        )
    workload = Workload(**payload["workload"])
    if original_workload:
        _require(
            (workload.cached_tokens, workload.new_tokens, workload.output_tokens)
            == (400000, 4000, 400),
            "original figure workload",
        )
        _require(
            payload["run_settings"].get("measurement_turns", 0) >= 1,
            "original-workload audit requires complete measured turns",
        )
    powers = payload.get("power_model")
    if powers is None:
        _require(
            hardware is not None,
            "saved power snapshot or explicit catalog audit required",
        )
        for device, saved in payload["hardware"].items():
            _require(
                hardware.device(device).to_dict() == saved,
                "catalog/saved hardware mismatch",
            )
        powers = power_snapshot(
            hardware, payload["power_overrides"], payload["hardware"]
        )
    for spec in powers.values():
        number(spec["watts"], "processor watts", positive=True)
    cases = {case.name: case for case in map(_case, payload["cases"])}
    _require(
        bool(cases) and len(cases) == len(payload["cases"]),
        "empty or duplicate declared cases",
    )
    rows = payload["results"]
    identities = [(row["case"], row["batch"]) for row in rows]
    _require(len(set(identities)) == len(identities), "duplicate experiment points")
    grid_coverage = _validate_grid(
        payload, cases, identities, expected_cases, expected_batches
    )
    for row in rows:
        _require(row["case"] in cases, "undeclared case")
        if row["status"] == "ok" and payload["run_settings"].get(
            "event_accounting_required", False
        ):
            _require(
                "token_events" in row and "decode_intervals" in row,
                "missing required token-accounting events",
            )
        if replay and row["status"] == "ok":
            validate_replay_profiles(row)
        validate_point(
            row,
            cases[row["case"]],
            workload,
            a_replicas=payload["run_settings"]["a_replicas"],
            watts=powers,
            facility_watts=payload["facility_watts"],
            model=payload["model"],
            executor="SampledNativeReplay" if replay else "ModelAFExecutor",
        )
    feasible = [row for row in rows if row["status"] == "ok"]
    rubin = [row for row in feasible if row["mode"] == "rubin"]
    _require(bool(rubin), "missing normalization baseline")
    baseline = max(row["tokens_per_s_per_mw"] for row in rubin)
    _equal(
        payload["normalization_tokens_per_s_per_mw"],
        baseline,
        "common Rubin normalization",
    )
    for row in feasible:
        _equal(
            row["normalized_throughput"],
            row["tokens_per_s_per_mw"] / baseline,
            "normalized efficiency",
        )
    for mode in ("rubin", "draft", "af", "pd"):
        candidates = [row for row in feasible if row["mode"] == mode]
        frontier = payload["frontiers"][mode]
        for point in frontier:
            matching = [
                r
                for r in candidates
                if (r["case"], r["batch"]) == (point["case"], point["batch"])
            ]
            _require(
                len(matching) == 1, "frontier includes missing or infeasible point"
            )
            for key in (
                "tokens_per_s_per_user",
                "tokens_per_s_per_mw",
                "normalized_throughput",
            ):
                _equal(point[key], matching[0][key], "frontier point values")
            _require(
                not any(_dominates(row, point) for row in candidates),
                "dominated frontier point",
            )
        for row in candidates:
            _require(
                any(_covers(point, row) for point in frontier),
                "frontier omits a nondominated point",
            )
    return dict(
        status="passed",
        points=len(rows),
        feasible=len(feasible),
        rejected=len(rows) - len(feasible),
        power_model=powers,
        original_workload=original_workload,
        execution_fidelity=(
            "modeled_native_operator_dags"
            if any(
                (row.get("replay_profiles") or {}).get("method")
                == "modeled_native_operator_dags"
                for row in feasible
            )
            else "sampled_native_operator_dags" if replay else "native_per_pass_dags"
        ),
        numerical_accuracy=(
            "requires_separate_native_control"
            if replay
            else "unvalidated_hardware_hypotheses"
        ),
        grid_coverage=grid_coverage,
        event_accounting=(
            "token_events_and_intervals"
            if all(
                "token_events" in row and "decode_intervals" in row for row in feasible
            )
            else "legacy_turn_totals"
        ),
    )


def _covers(a, b):
    return all(
        a[key] >= b[key] for key in ("tokens_per_s_per_user", "tokens_per_s_per_mw")
    )


def _dominates(a, b):
    return _covers(a, b) and any(
        a[key] > b[key] for key in ("tokens_per_s_per_user", "tokens_per_s_per_mw")
    )
