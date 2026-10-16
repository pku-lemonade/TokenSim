"""Resident capacity and decode-only concurrency probes on production AF DAGs."""

import simpy

from TokenSim.af.runtime.admission import rank_kv_bytes
from TokenSim.af.runtime.host import run_environment
from TokenSim.af.serving.metrics import steady_metrics
from TokenSim.af.serving.service import AFService
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


def cohort_loads(clients, micro_batch, replicas):
    for value, name in (
        (clients, "clients"),
        (micro_batch, "micro-batch"),
        (replicas, "replicas"),
    ):
        integer(value, name, minimum=1)
    cohorts, tail = divmod(clients, micro_batch)
    cycles, extra = divmod(cohorts, replicas)
    return [
        (cycles + (rank < extra)) * micro_batch + (tail if rank == extra else 0)
        for rank in range(replicas)
    ]


def resident_capacity(runtime, maximum_context, *, micro_batch=1):
    integer(maximum_context, "maximum context", minimum=1)
    integer(micro_batch, "micro-batch", minimum=1)
    if maximum_context > runtime.model.max_position_embeddings:
        raise ConfigurationError("capacity probe exceeds model context limit")
    owners = sorted(
        (owner for owner in runtime.plan.request_owners if owner.phase == "decode"),
        key=lambda owner: owner.dp_rank,
    )
    limits = []
    for owner in owners:
        sizes = {}
        for binding in runtime.plan.bindings_for_instance(owner.domain, owner.instance):
            sizes[binding.physical_identity] = sizes.get(
                binding.physical_identity, 0
            ) + rank_kv_bytes(runtime.plan, binding, maximum_context)
        ranks = []
        for physical, size in sizes.items():
            if not size:
                continue
            state = runtime.plan.physical_state(physical)
            free = (
                state.capacity_bytes - state.model_bytes - state.protocol_buffer_bytes
            )
            ranks.append(
                dict(
                    worker=physical,
                    clients=free // size,
                    kv_bytes_per_client=size,
                    free_bytes=free,
                )
            )
        if not ranks:
            raise ConfigurationError("capacity probe needs resident attention KV")
        limits.append(
            dict(dp_rank=owner.dp_rank, **min(ranks, key=lambda row: row["clients"]))
        )
    low, high = (
        0,
        (max(row["clients"] for row in limits) + 1) * len(limits) * micro_batch,
    )
    while low + 1 < high:
        middle = (low + high) // 2
        loads = cohort_loads(middle, micro_batch, len(limits))
        if all(load <= row["clients"] for load, row in zip(loads, limits)):
            low = middle
        else:
            high = middle
    return dict(maximum_resident_clients=low, replicas=limits)


def decode_probe(
    runtime,
    clients,
    *,
    micro_batch=1,
    context,
    steps=8,
    warmup_steps=2,
    seed=42,
    maximum_context=None,
    execution_factory=None,
):
    """Measure resident decode only; no suffix prefill or factory efficiency."""
    for value, name in (
        (clients, "clients"),
        (steps, "steps"),
        (warmup_steps, "warm-up steps"),
    ):
        integer(value, name, minimum=1)
    integer(micro_batch, "micro-batch", minimum=1)
    integer(context, "decode context")
    cohorts = -(-clients // micro_batch)
    peak = (
        context + warmup_steps + steps * (cohorts + 1)
        if maximum_context is None
        else maximum_context
    )
    integer(peak, "maximum context", minimum=1)
    if peak < context + warmup_steps + steps:
        raise ConfigurationError("decode probe context reservation is too short")
    capacity = resident_capacity(runtime, peak, micro_batch=micro_batch)
    if clients > capacity["maximum_resident_clients"]:
        raise ConfigurationError("decode probe exceeds resident capacity")
    measurement, events, intervals = {}, [], []
    driver = (
        execution_factory({"target": runtime}, measurement)
        if execution_factory
        else None
    )
    env = driver.env if driver else simpy.Environment()
    communication = (
        driver.communication(
            runtime.cluster.networks,
            [worker.physical_worker_ref.identity for worker in runtime.workers],
            "wire",
        )
        if driver
        else None
    )
    service = AFService(
        env,
        runtime,
        seed=seed,
        retain_trace=False,
        trace=driver is None,
        communication=communication,
        execution_driver=driver,
    )
    if driver:
        driver.bind({"target": service}, communication)
    replicas = len(capacity["replicas"])
    bundles = [
        [
            service.admit(
                index, cohort % replicas, maximum_context=peak, current_context=context
            )
            for index in range(start, min(start + micro_batch, clients))
        ]
        for cohort, start in enumerate(range(0, clients, micro_batch))
    ]
    ready = [env.event() for _ in bundles]
    complete = [env.event() for _ in bundles]
    begin = env.event()
    stop = env.event()

    def stream(cohort, sessions):
        for _ in range(warmup_steps):
            yield from service.forward(sessions)
            for session in sessions:
                session.generated += 1
        ready[cohort].succeed()
        yield begin
        completed = 0
        while not stop.triggered:
            start = env.now
            yield from service.forward(sessions)
            for session in sessions:
                session.generated += 1
            events.append(
                dict(time=env.now, tokens=len(sessions), phase="decode", cohort=cohort)
            )
            intervals.append((start, env.now, len(sessions)))
            completed += 1
            if completed >= steps and not complete[cohort].triggered:
                complete[cohort].succeed()
            if all(event.triggered for event in complete) and not stop.triggered:
                measurement["end"] = env.now
                stop.succeed()

    def observe():
        yield env.all_of(ready)
        measurement["start"] = env.now
        begin.succeed()

    for cohort, bundle in enumerate(bundles):
        env.process(stream(cohort, bundle))
    env.process(observe())
    run_environment(env)
    metric = (
        driver.metrics(events)
        if driver
        else steady_metrics(
            events,
            service.trace.events,
            start=measurement["start"],
            end=measurement["end"],
            capacities=service.executor.resources.resource_capacities(),
            include_end=True,
        )
    )
    selected = [row for row in events if row["time"] <= measurement["end"]]
    user_seconds = sum(
        (end - start) * count
        for start, end, count in intervals
        if end <= measurement["end"]
    )
    if metric["decode_tokens"] < clients * steps:
        raise ConfigurationError("decode probe lost measured token completions")
    compute_utilization = [
        value
        for key, value in metric["resource_utilization"].items()
        if key.split("/", 1)[-1].startswith("worker:")
    ]
    result = dict(
        clients=clients,
        micro_batch=micro_batch,
        cohorts=len(bundles),
        minimum_steps_per_client=steps,
        completed_steps_per_cohort=[
            sum(row["cohort"] == cohort for row in selected)
            for cohort in range(len(bundles))
        ],
        context=context,
        decode_tokens=metric["decode_tokens"],
        measurement_s=metric["duration_s"],
        decode_tokens_per_s=metric["decode_tokens_per_s"],
        tokens_per_s_per_user=metric["decode_tokens"] / user_seconds,
        drain_s=env.now - measurement["end"],
        maximum_compute_utilization=max(compute_utilization, default=0),
        mean_compute_utilization=sum(compute_utilization) / len(runtime.workers),
        maximum_resource_utilization=max(
            metric["resource_utilization"].values(), default=0
        ),
        memory=service.memory.summary(),
        metric_scope="resident_decode_only",
    )
    if driver:
        result["replay_profiles"] = driver.describe()
    return result
