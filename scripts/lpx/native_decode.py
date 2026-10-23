"""Sample native decode DAGs and replay strictly serial resident-prefix clients."""

from bisect import bisect_left
from dataclasses import asdict
from functools import lru_cache
from math import isclose
from types import SimpleNamespace

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.comm.streaming import StreamingConfig
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.replay_profiles import capture_profile
from scripts.lpx.requests import distribution


class DecodeCurve:
    """Adaptive integer-context interpolation with explicit native probe evidence.

    Probe agreement is a numerical check, not a proof between unsampled points.
    This curve is valid only for an isolated, deterministic, dense decode client.
    """

    def __init__(self, probe, low, high, *, knots=(), tolerance_s=1e-10):
        integer(low, "decode context lower bound", minimum=1)
        integer(high, "decode context upper bound", minimum=low)
        number(tolerance_s, "interpolation tolerance", positive=True)
        self.probe, self.low, self.high = probe, low, high
        self.tolerance_s = tolerance_s
        self.samples, self.segments = {}, []
        self.guard_refinements = []
        self.guarded = False
        boundaries = sorted({low, high, *(k for k in knots if low < k < high)})
        if low == high:
            self.sample(low)
        for left, right in zip(boundaries, boundaries[1:]):
            self.refine(left, right)

    def sample(self, context):
        if context not in self.samples:
            value = self.probe(context)
            number(value["latency_s"], "native decode latency", positive=True)
            self.samples[context] = value
        return self.samples[context]["latency_s"]

    def refine(self, left, right):
        first, last = self.sample(left), self.sample(right)
        middle = (left + right) // 2
        actual = self.sample(middle)
        predicted = first + (last - first) * (middle - left) / (right - left)
        error = abs(actual - predicted)
        if error > self.tolerance_s and right - left > 1:
            self.refine(left, middle)
            self.refine(middle, right)
        else:
            self.segments.append(dict(low=left, high=right, probe_error_s=error))

    def __call__(self, context):
        integer(context, "decode context", minimum=self.low)
        if context > self.high:
            raise ConfigurationError("decode context exceeds sampled interval")
        points = sorted(self.samples)
        index = bisect_left(points, context)
        if points[index] == context:
            return self.sample(context)
        left, right = points[index - 1], points[index]
        return self.sample(left) + (self.sample(right) - self.sample(left)) * (
            context - left
        ) / (right - left)

    def check(self, contexts):
        """Check withheld contexts and fail instead of silently changing a curve."""
        rows = []
        for context in sorted(set(contexts)):
            if context in self.samples:
                continue
            predicted = self(context)
            result = self.probe(context)
            error = abs(predicted - result["latency_s"])
            rows.append(dict(context=context, error_s=error, **result))
            if error > self.tolerance_s:
                raise SimulationStateError(
                    f"withheld native context {context} interpolation error {error}"
                )
        return rows

    def guard(self):
        """Refine exposed quarter-point errors without relaxing the tolerance.

        Guard points that pass remain outside the interpolation sample set.
        A failed guard becomes a new knot, and all resulting segments are checked.
        Each failed pass adds an integer context, so refinement terminates.
        """
        while True:
            failed = {}
            for segment in self.segments:
                left, right = segment["low"], segment["high"]
                for context in sorted({left + (right - left) * k // 4 for k in (1, 3)}):
                    if context in self.samples:
                        continue
                    predicted = self(context)
                    value = self.probe(context)
                    number(value["latency_s"], "native decode latency", positive=True)
                    error = abs(predicted - value["latency_s"])
                    if error > self.tolerance_s:
                        self.samples[context] = value
                        failed[left, right] = context
                        self.guard_refinements.append(
                            dict(context=context, error_s=error)
                        )
                        break
            if not failed:
                self.guarded = True
                return
            original, self.segments = self.segments, []
            for segment in original:
                left, right = segment["low"], segment["high"]
                context = failed.get((left, right))
                if context is None:
                    self.segments.append(segment)
                else:
                    self.refine(left, context)
                    self.refine(context, right)

    def report(self):
        return dict(
            low=self.low,
            high=self.high,
            tolerance_s=self.tolerance_s,
            segments=self.segments,
            probes=[dict(context=k, **v) for k, v in sorted(self.samples.items())],
            guarded_interpolation=self.guarded,
            guard_refinements=self.guard_refinements,
            approximation="Piecewise linear latency between sampled native DAG contexts.",
        )


def native_probe(runtime, streaming, invocation_us=0.0):
    settings = asdict(StreamingConfig(**streaming))
    if runtime.model.is_moe or runtime.model.components:
        raise ConfigurationError("native decode search requires a dense model")

    def probe(context):
        profile = capture_profile(
            runtime,
            "search",
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=context,
            dp_rank=0,
            seed=42,
            sample=0,
            streaming=settings,
            invocation_us=invocation_us,
        )
        meta = profile.metadata
        return dict(
            latency_s=meta["modeled_duration_s"],
            native_duration_s=meta["native_duration_s"],
            native_events=meta["native_events"],
            refined_nodes=len(profile.nodes),
            streaming_collectives=meta["streaming"]["streaming_collectives"],
            source_sha256=meta["source_sha256"],
            invocation_us=invocation_us,
            fabric=meta["streaming"].get("fabric"),
        )

    return lru_cache(maxsize=None)(probe)


def curve_for(runtime, inputs, outputs, streaming, invocation_us=0.0):
    integer(inputs, "input tokens", minimum=1)
    integer(outputs, "output tokens", minimum=2)
    knots = set(runtime.model.attention_windows or ())
    policy = runtime.hardware.device("groq_lp30").analytical.get("lpx_pipeline")
    if policy:
        from TokenSim.operator_data.analytical.lp30 import LP30Pipeline

        tile = LP30Pipeline(**policy).attention_tile_keys
        for boundary in range((inputs // tile) * tile, inputs + outputs, tile):
            knots.update((boundary, boundary + 1))
    domain = runtime.plan.domain_runtime(runtime.cluster.af_domains[0].name)
    if domain.kv_parallel_size > 1:
        # The busiest KVP rank changes slope at every append-block boundary.
        block = domain.kv_append_block
        for boundary in range((inputs // block) * block, inputs + outputs, block):
            knots.update((boundary, boundary + 1))
    curve = DecodeCurve(
        native_probe(runtime, streaming, invocation_us),
        inputs + 1,
        inputs + outputs - 1,
        knots=knots,
    )
    curve.guard()
    return curve


def interval_summary(curve, inputs, outputs):
    seconds = sum(curve(context) for context in range(inputs + 1, inputs + outputs))
    return dict(
        input_tokens=inputs,
        output_tokens=outputs,
        generation_seconds=seconds,
        generation_tps=(outputs - 1) / seconds,
        interpolation=curve.report(),
    )


class SerialDecodeDriver:
    """Use sampled DAG completion times only when no other client can contend."""

    def __init__(self, curve):
        self.curve, self.active, self.steps = curve, False, 0

    def forward(self, service, requests, phase):
        if (
            self.active
            or phase != "decode"
            or len(requests) != 1
            or len(service.sessions) != 1
            or requests[0].scheduled_query_tokens != 1
            or not requests[0].emit_logits
        ):
            raise ConfigurationError("native latency curve requires one serial client")
        self.active = True
        try:
            yield service.env.timeout(self.curve(requests[0].context_len))
            self.steps += 1
            return SimpleNamespace(iteration_id=self.steps)
        finally:
            self.active = False


def serial_replay(runtime, curve, *, inputs, outputs, requests=50, warmup=5):
    integer(requests, "measured requests", minimum=1)
    integer(warmup, "warm-up requests")
    if (curve.low, curve.high) != (inputs + 1, inputs + outputs - 1):
        raise ConfigurationError("serial workload differs from native decode curve")
    env, rows = simpy.Environment(), []
    driver = SerialDecodeDriver(curve)
    service = AFService(
        env, runtime, trace=False, retain_trace=False, execution_driver=driver
    )
    expected = interval_summary(curve, inputs, outputs)["generation_seconds"]

    def client():
        for index in range(warmup + requests):
            session = service.admit(
                index, 0, maximum_context=inputs + outputs, current_context=inputs
            )
            # The prefill-produced first output is available at this boundary.
            session.generated = 1
            first = env.now
            for _ in range(outputs - 1):
                yield env.process(service.forward([session], phase="decode"))
                session.generated += 1
            elapsed = env.now - first
            if not isclose(elapsed, expected, rel_tol=1e-9, abs_tol=1e-11):
                raise SimulationStateError("serial token clocks disagree with curve")
            rows.append(
                dict(
                    request_id=index,
                    warmup=index < warmup,
                    first_output_at=first,
                    last_output_at=env.now,
                    final_context=session.cached_context,
                    generation_tps=(outputs - 1) / elapsed,
                )
            )
            service.release(session)
            service.completed_passes.clear()

    env.run(until=env.process(client()))
    if (
        service.sessions
        or service.memory.leases
        or any(service.memory.reserved.values())
    ):
        raise SimulationStateError("serial request leaked a KV lease")
    return dict(
        executor="native_DAG_sampled_serial_token_clock",
        token_unit="native_model_tokens",
        clients=1,
        measured_requests=requests,
        warmup_requests=warmup,
        executed_decode_steps=driver.steps,
        generation_tps=distribution(
            [r["generation_tps"] for r in rows if not r["warmup"]]
        ),
        request_timings=rows,
        ttft_mode="Excluded; each request starts with resident prompt KV and its first output.",
        limitations=[
            "Native DAG samples supply latency; full resource DAGs are not rerun for every token.",
            "Interpolation is numerically checked, not proven at unsampled contexts.",
            "Deterministic equal-length requests do not provide independent statistical evidence.",
            "This decode replay alone does not validate prefill, workspace, request text or tokenizer conversion.",
        ],
    )
