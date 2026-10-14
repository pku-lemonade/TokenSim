"""Af host fixture."""

from __future__ import annotations

from types import SimpleNamespace

import simpy

from TokenSim.af import AFExecutor, AFServiceEstimate
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.model_executor import ModelAFExecutor
from TokenSim.af.resources import AFResourceRegistry
from tests.support import af_native_fixture as native_fixture


class FixedHostFixture:
    @classmethod
    def setUpClass(cls):
        native_fixture.NativeFixture.setUpClass()
        cls.fixture = native_fixture.NativeFixture()

    def executor(self, *, model=False, trace=True, cache=True, **costs):
        fixture = self.fixture
        runtime = build_model_runtime(
            fixture.document(
                trace_enabled=trace,
                service_estimate_cache=cache,
                dag_template_cache=cache,
                communication_timing=dict(enabled=True, host_submit_us=0, **costs),
            ),
            hardware=fixture.base.hardware,
            model=fixture.model,
        )

        class Provider:
            def estimate(self, query):
                payload = getattr(query, "payload_bytes", None)
                duration = (
                    5
                    if payload is not None
                    else (100 if query.stage == "attention" else 0)
                )
                return AFServiceEstimate(
                    duration / 1e6,
                    "oracle",
                    payload_bytes=payload,
                    details={
                        "timing_scope": (
                            "transport_service"
                            if payload is not None
                            else "device_service"
                        )
                    },
                )

        env = simpy.Environment()
        executor_type = ModelAFExecutor if model else AFExecutor
        executor = executor_type(
            env,
            runtime.cluster,
            runtime.workers,
            Provider(),
            Provider(),
            fixture.model.model_id,
            "bf16",
            2,
            256,
            moe_config=fixture.model.moe,
            runtime_plan=runtime.plan,
            resources=AFResourceRegistry(
                env, runtime.cluster, runtime.workers, runtime.placement
            ),
            **({"model": fixture.model} if model else {}),
        )
        return env, executor

    def run_step(self, env, executor, count=1):
        requests = [
            SimpleNamespace(
                id=i,
                is_prefill=False,
                prefill_len=128,
                context_len=129,
                generation_idx=0,
            )
            for i in range(count)
        ]
        iteration = executor.submit(0, requests, "decode")
        process = env.process(executor.run_iteration(iteration))
        env.run(until=process)
        return iteration
