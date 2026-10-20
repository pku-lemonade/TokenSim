"""Run LPX requests through AF services in a finite measurement window."""

import numpy as np

from TokenSim.af.serving.handoff import plan_handoff, execute_handoff
from TokenSim.af.runtime.host import host_collection
from TokenSim.config.validation import boolean, integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.native_config import runtime_configurations, session_prefix_context
from scripts.lpx.native_metrics import collect_metrics
from scripts.lpx.native_setup import create_serving, allocated_power
from scripts.lpx.requests import Arrivals, request_metrics
from scripts.lpx.control import token_control as send_tokens


@host_collection()
def point(
    hardware,
    target,
    workload,
    case,
    clients,
    *,
    micro_batch=1,
    a_dp=1,
    warmup_steps=2,
    warmup_seconds=0.0,
    warmup_mode="resident_decode",
    measurement_seconds=0.05,
    measurement_turns=1,
    execution_mode="cuda_graph",
    graph_kernel_us=0.2,
    endpoint_transfer_occupancy="wire",
    handoff_mode="incremental",
    power_overrides=None,
    facility_watts=0.0,
    draft_spec=None,
    trace=False,
    stream_metrics=False,
    qps=None,
    arrival_pattern="uniform",
    dag_template_cache=True,
    execution_factory=None,
    ffn_batching=False,
    ffn_batch_wait_us=5.0,
):
    boolean(stream_metrics, "stream metrics")
    number(warmup_seconds, "warm-up seconds")
    if qps is not None and min(micro_batch, clients) > 1:
        raise ConfigurationError(
            "open arrivals require micro-batch 1; synchronized cohorts support fixed-client runs"
        )
    for value, name in (
        (clients, "clients"),
        (micro_batch, "micro-batch"),
        (a_dp, "A replicas"),
        (warmup_steps, "warm-up steps"),
    ):
        integer(value, name, minimum=1)
    integer(measurement_turns, "measurement turns")
    number(measurement_seconds, "measurement interval", positive=True)
    if warmup_mode not in {"resident_decode", "prefill_and_decode"}:
        raise ConfigurationError("invalid warm-up mode")
    number(facility_watts, "facility power")
    if workload.output_tokens < 2:
        raise ConfigurationError(
            "serving interactivity requires at least two output tokens"
        )
    peak_context = workload.maximum_context(case.mode == "draft")
    if peak_context > target.max_position_embeddings:
        raise ConfigurationError("workload exceeds the target context limit")
    runtimes = runtime_configurations(
        hardware,
        target,
        case,
        draft_spec=draft_spec,
        verification_width=workload.draft_tokens + 1 if case.draft else 1,
        a_dp=a_dp,
        micro_batch=micro_batch,
        execution_mode=execution_mode,
        graph_kernel_us=graph_kernel_us,
        endpoint_transfer_occupancy=endpoint_transfer_occupancy,
        dag_template_cache=dag_template_cache,
        ffn_batching=ffn_batching,
        ffn_batch_wait_us=ffn_batch_wait_us,
    )
    measurement = {}
    env, communication, services, windows, driver = create_serving(
        hardware,
        runtimes,
        case,
        workload,
        measurement,
        trace=trace,
        stream_metrics=stream_metrics,
        execution_factory=execution_factory,
        endpoint_transfer_occupancy=endpoint_transfer_occupancy,
    )
    inventory, power = allocated_power(
        hardware, runtimes, power_overrides, facility_watts
    )
    stop = env.event()
    begin_measurement = env.event()
    arrivals = Arrivals(
        env,
        begin_measurement,
        stop,
        measurement,
        qps=qps,
        pattern=arrival_pattern,
        seed=workload.seed,
    )
    token_events, handoff_events, control_events, decode_intervals, turns = (
        [],
        [],
        [],
        [],
        [],
    )
    verification_events = []
    active_turns = {}
    warm_events = []
    complete_events = []
    initial_sessions = []
    cohort_indices = [
        list(range(start, min(start + micro_batch, clients)))
        for start in range(0, clients, micro_batch)
    ]
    for cohort, indices in enumerate(cohort_indices):
        bundle = {}
        for name, service in services.items():
            replicas = service.runtime.plan.domain_runtime(
                next(iter(service.owners.values())).domain
            ).data_parallel_size
            dp = cohort % replicas
            prefix = session_prefix_context(
                workload,
                case,
                name,
                handoff_mode=handoff_mode,
                warmup_mode=warmup_mode,
                warming=True,
            )
            bundle[name] = [
                service.admit(
                    i, dp, maximum_context=peak_context, current_context=prefix
                )
                for i in indices
            ]
        initial_sessions.append(bundle)
        warm_events.append(env.event())
        complete_events.append(env.event())

    def token_control(bundle, cohort, round_index, step, reverse=False):
        return send_tokens(
            services,
            communication,
            hardware,
            control_events,
            workload,
            bundle,
            cohort,
            round_index,
            step,
            reverse,
        )

    def prefill(name, sessions):
        left = workload.new_tokens
        while left and not stop.triggered:
            query = min(workload.prefill_chunk, left)
            yield from services[name].forward(
                sessions, phase="prefill", query_tokens=query, emit_logits=query == left
            )
            left -= query
        return not left

    def stream(cohort, bundle):
        completed_steps = 0
        measured_turns = 0
        warmed = False
        rng = np.random.default_rng(np.random.SeedSequence([workload.seed, cohort]))
        round_index = 0
        while not stop.triggered:
            warm_turn = not warmed
            submitted = (
                env.now
                if warm_turn
                else (yield from arrivals.acquire(len(bundle["target"])))
            )
            if submitted is None:
                return
            started = env.now
            turn = dict(
                cohort=cohort,
                round=round_index,
                start=started,
                submitted_at=submitted,
                ttft_s=0.0,
                end=started,
                completed=False,
                warmup=warm_turn,
            )
            turns.append(turn)
            active_turns[cohort] = turn
            prefill_name = "prefill" if case.prefill else "target"
            do_prefill = not warm_turn or warmup_mode == "prefill_and_decode"
            draft_prefill = (
                env.process(prefill("draft", bundle["draft"]))
                if do_prefill and case.draft
                else None
            )
            if do_prefill and not (
                yield from prefill(prefill_name, bundle[prefill_name])
            ):
                if draft_prefill is not None:
                    yield draft_prefill
                return
            if do_prefill and case.prefill and not stop.triggered:
                for before, after in zip(bundle["prefill"], bundle["target"]):
                    context = workload.cached_tokens + workload.new_tokens
                    copies = plan_handoff(
                        services["prefill"],
                        before,
                        services["target"],
                        after,
                        context=context,
                        mode=handoff_mode,
                        cross_network="handoff",
                        replication_network="replication",
                    )
                    yield from execute_handoff(
                        env,
                        copies,
                        hardware=hardware,
                        resources=communication,
                        identity=(cohort, round_index, before.request_id),
                        target=services["target"],
                        target_session=after,
                        context=context,
                        events=handoff_events,
                    )
            if draft_prefill is not None:
                yield draft_prefill
            if stop.triggered:
                return
            if do_prefill and case.draft:
                yield from token_control(
                    bundle, cohort, round_index, completed_steps, reverse=True
                )
            first = env.now
            remaining = workload.output_tokens - 1
            for session in bundle["target"]:
                session.generated = 1
            for session in bundle.get("draft", ()):
                session.generated = 1
            token_events.append(
                dict(
                    time=env.now,
                    tokens=len(bundle["target"]),
                    phase="prefill",
                    cohort=cohort,
                    round=round_index,
                    warmup=warm_turn,
                )
            )
            while remaining and not stop.triggered:
                step_start = env.now
                if case.mode == "draft":
                    width = workload.draft_tokens + 1
                    for _ in range(workload.draft_tokens):
                        yield from services["draft"].forward(bundle["draft"])
                        for session in bundle["draft"]:
                            session.generated += 1
                    yield from token_control(
                        bundle, cohort, round_index, completed_steps
                    )
                    yield from services["target"].forward(
                        bundle["target"], query_tokens=width
                    )
                    accepted = 0
                    for _ in range(workload.draft_tokens):
                        if rng.random() >= workload.acceptance_probability:
                            break
                        accepted += 1
                    committed = min(remaining, accepted + 1)
                    # Roll back uncommitted speculative positions on both models.
                    valid = (
                        workload.cached_tokens
                        + workload.new_tokens
                        + bundle["target"][0].generated
                        + committed
                        - 1
                    )
                    missing = valid - bundle["draft"][0].cached_context
                    if missing > 0:
                        # Complete acceptance needs the final accepted draft
                        # token's KV before the verifier bonus token is reused.
                        yield from services["draft"].forward(
                            bundle["draft"], query_tokens=missing
                        )
                    yield from token_control(
                        bundle, cohort, round_index, completed_steps, reverse=True
                    )
                    for name in ("target", "draft"):
                        for session in bundle[name]:
                            services[name].memory.commit(session.request_id, valid)
                            session.cached_context = valid
                            if name == "draft":
                                session.generated = (
                                    bundle["target"][0].generated + committed
                                )
                    verification_events.append(
                        dict(
                            time=env.now,
                            cohort=cohort,
                            verification_width=width,
                            accepted_prefix=accepted,
                            committed_tokens=committed,
                            draft_catchup_tokens=max(0, missing),
                            warmup=warm_turn,
                        )
                    )
                else:
                    yield from services["target"].forward(bundle["target"])
                    committed = 1
                for session in bundle["target"]:
                    session.generated += committed
                remaining -= committed
                token_events.append(
                    dict(
                        time=env.now,
                        tokens=committed * len(bundle["target"]),
                        phase="decode",
                        cohort=cohort,
                        round=round_index,
                        warmup=warm_turn,
                    )
                )
                decode_intervals.append(
                    dict(
                        start=step_start,
                        end=env.now,
                        tokens=committed,
                        clients=len(bundle["target"]),
                        cohort=cohort,
                        round=round_index,
                        warmup=warm_turn,
                    )
                )
                completed_steps += 1
                if (
                    completed_steps >= warmup_steps
                    and env.now >= warmup_seconds
                    and not warm_events[cohort].triggered
                ):
                    warm_events[cohort].succeed()
                    yield begin_measurement
                    warmed = True
                    # Restore the fixed prefix and start a measured complete turn.
                    break
            turn.update(ttft_s=first - started, end=env.now, completed=remaining == 0)
            active_turns.pop(cohort, None)
            if not warm_turn and remaining == 0:
                measured_turns += 1
                if (
                    measured_turns >= measurement_turns
                    and not complete_events[cohort].triggered
                ):
                    complete_events[cohort].succeed()
            if (
                not stop.triggered
                and measurement_turns
                and all(event.triggered for event in complete_events)
                and env.now >= measurement["start"] + measurement_seconds
            ):
                measurement["end"] = env.now
                if not stop.triggered:
                    stop.succeed()
            if stop.triggered:
                break
            round_index += 1
            for name, sessions in bundle.items():
                for session in sessions:
                    session.cached_context = session_prefix_context(
                        workload,
                        case,
                        name,
                        handoff_mode=handoff_mode,
                        warmup_mode=warmup_mode,
                        warming=not warmed,
                    )
                    services[name].memory.commit(
                        session.request_id, session.cached_context
                    )
                    session.generated = 0

    def observe():
        yield env.all_of(warm_events)
        measurement["start"] = env.now
        begin_measurement.succeed()
        yield env.timeout(measurement_seconds)
        if measurement_turns:
            yield env.all_of(complete_events)
        if not stop.triggered:
            measurement["end"] = env.now
            stop.succeed()

    finished = [env.event() for _ in initial_sessions]

    def run_stream(index, bundle):
        try:
            yield from stream(index, bundle)
        finally:
            if index in active_turns:
                active_turns.pop(index).update(end=env.now)
            finished[index].succeed()

    for index, bundle in enumerate(initial_sessions):
        env.process(run_stream(index, bundle))
    env.process(observe())
    env.run(until=env.all_of(finished))
    metric, traces = collect_metrics(
        services,
        measurement,
        token_events,
        handoff_events,
        control_events,
        communication,
        windows,
        driver=driver,
    )
    intervals = [
        row
        for row in decode_intervals
        if not row["warmup"]
        and measurement["start"] <= row["end"] <= measurement["end"]
    ]
    tokens = sum(row["tokens"] * row["clients"] for row in intervals)
    user_seconds = sum(
        (row["end"] - row["start"]) * row["clients"] for row in intervals
    )
    if not user_seconds:
        raise ConfigurationError(
            "measurement interval contains no completed decode pass"
        )
    requests = request_metrics(turns, workload, clients, micro_batch, measurement)
    result = dict(
        case=case.name,
        mode=case.mode,
        batch=clients,
        clients=clients,
        micro_batch=micro_batch,
        status="ok",
        tokens_per_s_per_user=requests["generation_tps"]["p50"]
        or tokens / user_seconds,
        decode_weighted_tokens_per_s_per_user=tokens / user_seconds,
        interactivity_statistic=(
            "per_request_generation_tps_p50"
            if requests["completed_requests"]
            else "decode_time_weighted_no_complete_requests"
        ),
        request_statistics=requests,
        arrivals=arrivals.summary(requests["requests"]),
        output_tokens_per_s=metric["output_tokens_per_s"],
        tokens_per_s_per_mw=metric["output_tokens_per_s"] * 1e6 / power,
        power_watts=power,
        inventory=inventory,
        steady_state=metric,
        timings={
            "warmup_s": measurement["start"],
            "measurement_s": measurement["end"] - measurement["start"],
            "minimum_measurement_s": measurement_seconds,
            "minimum_complete_turns_per_cohort": measurement_turns,
            "completed_token_boundary": "[start, end]; warmup explicitly excluded",
            "drain_s": env.now - measurement["end"],
        },
        services={name: service.summary() for name, service in services.items()},
        handoff_events=handoff_events,
        control_events=control_events,
        verification_events=verification_events,
        turns=turns,
        token_events=token_events,
        decode_intervals=decode_intervals,
        execution_mode=execution_mode,
        endpoint_transfer_occupancy=endpoint_transfer_occupancy,
        graph_device_kernel_us_assumption=graph_kernel_us,
        warmup_mode=warmup_mode,
        handoff_mode=handoff_mode,
        native_executor="SampledNativeReplay" if driver else "ModelAFExecutor",
        replay_profiles=driver.describe() if driver else None,
        trace=[event.as_dict() for event in traces] if trace else [],
    )
    if warmup_seconds:
        result["minimum_warmup_seconds"] = warmup_seconds
    return result
