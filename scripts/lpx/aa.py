"""Serial full-model LPU estimates using TokenSim's ordinary operator backend.

An isolated client has no inter-request resource contention. TP ranks of a
dense model have equal local shapes; the representative of each pipeline
stage prices its full-model kernels and physical collective transfers.
"""

from dataclasses import replace
import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import DeviceCatalog
from TokenSim.latency.operator_table.stats import OperatorStats
from TokenSim.llm.llm_request import Request
from scripts.lpx.native_config import configuration
from scripts.lpx.requests import distribution


def scaled_efficiency(hardware, scale):
    number(scale, "AA shared efficiency scale", positive=True)
    device = hardware.device("groq_lp30")
    keys = ("gemm_mfu", "attention_mfu", "memory_efficiency")
    values = {key: device.analytical[key] * scale for key in keys}
    if max(values.values()) > 1 + 1e-12:
        raise ConfigurationError("AA efficiency fit exceeds physical peak efficiency")
    updated = replace(device, analytical={**device.analytical, **values})
    return replace(
        hardware,
        devices=DeviceCatalog(
            [
                updated if d.device_id == device.device_id else d
                for d in hardware.devices
            ]
        ),
    )


def serial(
    hardware,
    model,
    layout,
    *,
    input_tokens,
    output_tokens,
    requests=50,
    warmup=5,
    prefill_chunk=4096,
    rho_in=1.0,
    rho_out=1.0,
):
    for name, value in (
        ("requests", requests),
        ("input tokens", input_tokens),
        ("output tokens", output_tokens),
        ("prefill chunk", prefill_chunk),
    ):
        integer(value, name, minimum=1)
    integer(warmup, "warm-up requests")
    number(rho_in, "input tokenizer ratio", positive=True)
    number(rho_out, "output tokenizer ratio", positive=True)
    if model.is_moe or not layout.colocated or layout.a_device != "groq_lp30":
        raise ConfigurationError(
            "AA serial estimates require a dense full-model LPU layout"
        )
    native_in, native_out = round(input_tokens * rho_in), round(output_tokens * rho_out)
    if native_in < 1 or native_out < 2:
        raise ConfigurationError(
            "AA native lengths require positive input and two output tokens"
        )
    runtime = configuration(hardware, model, layout, namespace="aa")
    env = simpy.Environment()
    service = AFService(env, runtime, trace=False, retain_trace=False)
    owner = next(iter(service.owners.values()))
    bindings = runtime.plan.bindings_for_instance(owner.domain, owner.instance)
    leaders = sorted([b for b in bindings if b.tp_rank == 0], key=lambda b: b.pp_rank)
    backends = [service.prefill._backend(binding) for binding in leaders]
    for backend in backends:
        backend.decode_context_bucket = 1
        for child in backend._layer_backends:
            child.decode_context_bucket = 1
    rows, memory = [], {}

    def latency(request):
        elapsed = 0.0
        for backend in backends:
            backend.communicator.operations.clear()
            elapsed += backend.estimate_step_latency([request])
        return elapsed

    def client():
        for i in range(warmup + requests):
            start = env.now
            session = service.admit(
                i, 0, maximum_context=native_in + native_out, current_context=0
            )
            memory.update(service.memory.summary())
            request = Request(i, native_in, native_out, 16)
            left = native_in
            while left:
                query = min(prefill_chunk, left)
                request.scheduled_tokens = request.scheduled_query_tokens = query
                request.emit_logits = left == query
                yield env.timeout(latency(request))
                request.num_computed_tokens += query
                left -= query
            first = env.now
            request.generation_idx = 1
            request.emit_logits = True
            for _ in range(native_out - 1):
                request.scheduled_tokens = request.scheduled_query_tokens = 1
                yield env.timeout(latency(request))
                request.num_computed_tokens += 1
                request.generation_idx += 1
            service.memory.commit(session.request_id, request.num_computed_tokens)
            speed = (native_out - 1) / (env.now - first)
            rows.append(
                dict(
                    request_id=i,
                    user=0,
                    warmup=i < warmup,
                    submitted_at=start,
                    admitted_at=start,
                    first_output_at=first,
                    last_output_at=env.now,
                    native_input_tokens=native_in,
                    native_output_tokens=native_out,
                    native_generation_tps=speed,
                    standardized_generation_tps=speed / rho_out,
                    ttft_s=first - start,
                    latency_s=env.now - start,
                )
            )
            service.release(session)

    env.run(until=env.process(client()))
    measured = [r for r in rows if not r["warmup"]]
    stats = OperatorStats()
    for backend in backends:
        stats = stats.aggregate(backend.stats)
    return dict(
        status="ok",
        executor="OperatorTableLatencyBackend_isolated_dense_pipeline",
        clients=1,
        requests=len(measured),
        warmup_requests=warmup,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        rho_in=rho_in,
        rho_out=rho_out,
        layout=vars(layout),
        native_generation_tps=distribution(
            [r["native_generation_tps"] for r in measured]
        ),
        standardized_generation_tps=distribution(
            [r["standardized_generation_tps"] for r in measured]
        ),
        ttft_s=distribution([r["ttft_s"] for r in measured]),
        request_timings=rows,
        memory=memory,
        inventory=layout.inventory(),
        operator_statistics=stats.as_dict(),
        operator_statistics_scope="stage representatives; includes warmup and prefill; no inter-request contention",
        limitations=[
            "Identical deterministic token-length workloads are repeated; no independent variance estimate.",
            "AA raw texts, streamed chunks and hidden-reasoning intervals are unavailable.",
            "Native/o200k ratios and deployment precision are declared hypotheses.",
            "Whole-model TP/PP analytical composition; no speculative decoding or network ingress latency.",
        ],
    )
