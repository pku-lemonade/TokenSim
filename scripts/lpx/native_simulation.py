"""Run LPX requests through native AF services in a steady-state interval."""

import numpy as np
import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.serving.service import AFService
from TokenSim.af.serving.handoff import plan_handoff, execute_handoff
from TokenSim.af.serving.metrics import steady_metrics
from TokenSim.af.serving.timing import (
    namespace_event,
    namespace_resource,
    transport_timing,
)
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.native_config import configuration
from scripts.lpx.simulator import draft_model


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
):
    for value, name in (
        (clients, "clients"),
        (micro_batch, "micro-batch"),
        (a_dp, "A replicas"),
        (warmup_steps, "warm-up steps"),
    ):
        integer(value, name, minimum=1)
    integer(measurement_turns, "measurement turns")
    number(measurement_seconds, "measurement interval", positive=True)
    number(facility_watts, "facility power")
    if workload.output_tokens < 2:
        raise ConfigurationError(
            "serving interactivity requires at least two output tokens"
        )
    peak_context = (
        workload.cached_tokens
        + workload.new_tokens
        + workload.output_tokens
        + (workload.draft_tokens if case.mode == "draft" else 0)
    )
    if peak_context > target.max_position_embeddings:
        raise ConfigurationError("workload exceeds the target context limit")
    runtimes = {
        "target": configuration(
            hardware,
            target,
            case.decode,
            namespace="target",
            micro_batch=micro_batch,
            a_dp=a_dp,
            execution_mode=execution_mode,
            graph_kernel_us=graph_kernel_us,
            endpoint_transfer_occupancy=endpoint_transfer_occupancy,
        )
    }
    if case.prefill:
        runtimes["prefill"] = configuration(
            hardware,
            target,
            case.prefill,
            namespace="prefill",
            micro_batch=micro_batch,
            execution_mode=execution_mode,
            graph_kernel_us=graph_kernel_us,
            endpoint_transfer_occupancy=endpoint_transfer_occupancy,
        )
    if case.draft:
        runtimes["draft"] = configuration(
            hardware,
            draft_spec or draft_model(target),
            case.draft,
            namespace="draft",
            micro_batch=micro_batch,
            execution_mode=execution_mode,
            graph_kernel_us=graph_kernel_us,
            endpoint_transfer_occupancy=endpoint_transfer_occupancy,
        )
    env = simpy.Environment()
    networks = {
        key: value
        for runtime in runtimes.values()
        for key, value in runtime.cluster.networks.items()
    }
    networks.update(
        handoff="ethernet1600Gb",
        replication=hardware.device(case.decode.a_device).scale_up_link,
    )
    identities = [
        worker.physical_worker_ref.identity
        for runtime in runtimes.values()
        for worker in runtime.workers
    ]
    communication = EndpointResources(
        env, networks, identities, transfer_occupancy=endpoint_transfer_occupancy
    )
    services = {
        name: AFService(
            env,
            runtime,
            communication=communication,
            seed=workload.seed,
            retain_trace=trace,
        )
        for name, runtime in runtimes.items()
    }
    inventory = {}
    for runtime in runtimes.values():
        for worker in runtime.workers:
            inventory[worker.hardware] = inventory.get(worker.hardware, 0) + 1
    power = facility_watts
    for device, count in inventory.items():
        watts = (power_overrides or {}).get(device, hardware.device(device).tdp_w)
        number(watts, f"{device} power", positive=True)
        power += count * watts
    stop = env.event()
    begin_measurement = env.event()
    token_events, handoff_events, control_events, decode_intervals, turns = (
        [],
        [],
        [],
        [],
        [],
    )
    verification_events = []
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
            prefix = (
                workload.cached_tokens
                if name != "target" or not case.prefill or handoff_mode == "incremental"
                else 0
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
        draft_owner = services["draft"].owners[("decode", bundle["draft"][0].dp_rank)]
        target_owner = services["target"].owners[
            ("decode", bundle["target"][0].dp_rank)
        ]
        pair = (
            (target_owner.physical_worker, draft_owner.physical_worker)
            if reverse
            else (draft_owner.physical_worker, target_owner.physical_worker)
        )
        size = len(bundle["target"]) * 4 * (1 if reverse else workload.draft_tokens)
        keys = tuple(
            communication.format_key(key)
            for key, _ in communication.for_step("handoff", "a_to_f", pair)
        )
        duration = hardware.link("ethernet1600Gb").analytical_transfer_us(size) / 1e6
        times = yield from communication.run_transfer(
            keys,
            duration,
            hardware.link("ethernet1600Gb").latency_us / 1e6,
            (cohort, round_index, step, int(reverse)),
        )
        control_events.append(
            dict(
                source=pair[0],
                target=pair[1],
                bytes=size,
                network="handoff",
                ready_s=times[0],
                start_s=times[1],
                end_s=times[2],
                injection_end_s=times[3],
            )
        )

    def prefill(name, sessions):
        left = workload.new_tokens
        while left:
            query = min(workload.prefill_chunk, left)
            yield from services[name].forward(
                sessions, phase="prefill", query_tokens=query
            )
            left -= query

    def stream(cohort, bundle):
        completed_steps = 0
        measured_turns = 0
        warmed = False
        rng = np.random.default_rng(np.random.SeedSequence([workload.seed, cohort]))
        round_index = 0
        while not stop.triggered:
            started = env.now
            warm_turn = not warmed
            prefill_name = "prefill" if case.prefill else "target"
            yield from prefill(prefill_name, bundle[prefill_name])
            if case.prefill:
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
            if case.draft:
                yield from prefill("draft", bundle["draft"])
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
                    )
                )
                decode_intervals.append(
                    dict(
                        start=step_start,
                        end=env.now,
                        tokens=committed,
                        clients=len(bundle["target"]),
                        cohort=cohort,
                    )
                )
                completed_steps += 1
                if (
                    completed_steps >= warmup_steps
                    and not warm_events[cohort].triggered
                ):
                    warm_events[cohort].succeed()
                    yield begin_measurement
                    warmed = True
                    # Restore the fixed prefix and start a measured complete turn.
                    break
            turns.append(
                dict(
                    cohort=cohort,
                    start=started,
                    ttft_s=first - started,
                    end=env.now,
                    completed=remaining == 0,
                    warmup=warm_turn,
                )
            )
            if not warm_turn and remaining == 0:
                measured_turns += 1
                if (
                    measured_turns >= measurement_turns
                    and not complete_events[cohort].triggered
                ):
                    complete_events[cohort].succeed()
            if stop.triggered:
                break
            round_index += 1
            for name, sessions in bundle.items():
                for session in sessions:
                    session.cached_context = (
                        workload.cached_tokens
                        if name != "target"
                        or handoff_mode == "incremental"
                        or not case.prefill
                        else 0
                    )
                    services[name].memory.commit(
                        session.request_id, session.cached_context
                    )
                    session.generated = 0

    measurement = {}

    def observe():
        yield env.all_of(warm_events)
        measurement["start"] = env.now
        begin_measurement.succeed()
        yield env.timeout(measurement_seconds)
        if measurement_turns:
            yield env.all_of(complete_events)
        measurement["end"] = env.now
        stop.succeed()

    for index, bundle in enumerate(initial_sessions):
        env.process(stream(index, bundle))
    env.process(observe())
    env.run()
    traces = [
        namespace_event(name, event)
        for name, service in services.items()
        for event in service.trace.events
    ]
    capacities = {
        namespace_resource(name, key): value
        for name, service in services.items()
        for key, value in service.executor.resources.resource_capacities().items()
    }
    metric = steady_metrics(
        token_events,
        [
            *traces,
            *(
                transport_timing(row, communication)
                for row in (*handoff_events, *control_events)
            ),
        ],
        start=measurement["start"],
        end=measurement["end"],
        capacities=capacities,
    )
    intervals = [
        row
        for row in decode_intervals
        if measurement["start"] <= row["end"] < measurement["end"]
    ]
    tokens = sum(row["tokens"] * row["clients"] for row in intervals)
    user_seconds = sum(
        (row["end"] - row["start"]) * row["clients"] for row in intervals
    )
    if not user_seconds:
        raise ConfigurationError(
            "measurement interval contains no completed decode pass"
        )
    return dict(
        case=case.name,
        mode=case.mode,
        batch=clients,
        clients=clients,
        micro_batch=micro_batch,
        status="ok",
        tokens_per_s_per_user=tokens / user_seconds,
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
            "drain_s": env.now - measurement["end"],
        },
        services={name: service.summary() for name, service in services.items()},
        handoff_events=handoff_events,
        control_events=control_events,
        verification_events=verification_events,
        turns=turns,
        execution_mode=execution_mode,
        endpoint_transfer_occupancy=endpoint_transfer_occupancy,
        graph_device_kernel_us_assumption=graph_kernel_us,
        handoff_mode=handoff_mode,
        native_executor="ModelAFExecutor",
        trace=[event.as_dict() for event in traces] if trace else [],
    )
