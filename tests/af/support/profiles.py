"""Native protocol and calibration document fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.af.calibration.host import (
    HOST_COVERAGE,
    HostOverheadProfile,
    runtime_fingerprint,
)
from TokenSim.af.calibration.services import OperationServiceProfile
from TokenSim.af.communication.resources import AFResourceRegistry
from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.model_config import ModelSpec
from scripts.run_deepseek_mla import configuration


class NativeFixture:
    @classmethod
    def setUpClass(cls):
        cls.base = configuration(a_tp=1, f_tp=1, af_batch_size=8)
        cls.model = ModelSpec.from_mapping(
            dict(
                model_id="qwen3-moe-tiny-random",
                hidden_size=256,
                intermediate_size=512,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                vocab_size=128,
                dtype="bf16",
                kv_cache_dtype="bf16",
                attention_type="gqa",
                qk_norm=True,
                moe=dict(
                    num_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=256,
                    num_shared_experts=0,
                    num_moe_layers=2,
                ),
            )
        )

    def document(self, **overrides):
        doc = copy.deepcopy(self.base.doc)
        doc["af"].update(
            execution_protocol="vllm_afd",
            communication_timing=dict(enabled=True),
        )
        doc["af"].update(overrides)
        return doc


class FixedHostFixture:
    @classmethod
    def setUpClass(cls):
        NativeFixture.setUpClass()
        cls.fixture = NativeFixture()

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
        iteration = executor.submit(0, requests)
        process = env.process(executor.run_iteration(iteration))
        env.run(until=process)
        return iteration


class HostProfileFixture:
    @classmethod
    def setUpClass(cls):
        NativeFixture.setUpClass()
        cls.fixture = NativeFixture()

    def runtime(self, costs=None):
        doc = self.fixture.document(
            communication_timing=dict(enabled=True, native_control_enabled=True)
        )
        if costs:
            doc["af"]["communication_timing"].update(costs)
        runtime = build_model_runtime(
            doc, hardware=self.fixture.base.hardware, model=self.fixture.model
        )
        return runtime

    def document(self, runtime):
        return dict(
            schema_version=1,
            profile_id="unit-test-host",
            unit="us",
            timing_scope="host_service",
            contains_wait=False,
            excludes_device_launch=True,
            protocol="vllm_afd",
            runtime_fingerprint=runtime_fingerprint(
                runtime.plan, runtime.cluster, runtime.workers, runtime.hardware
            ),
            environment=dict(execution_mode="eager", software_sha256="test-software"),
            batch_range=[1, 8],
            context_range=[129, 160],
            costs={
                name: dict(
                    fixed_us=value, per_token_us=0, coverage=[HOST_COVERAGE[name]]
                )
                for name, value in [("a_step_prepare_us", 10), ("host_submit_us", 2)]
            },
            evidence=dict(
                source="unit-test://not-real-calibration",
                kind="synthetic_test",
                sample_count=100,
                window_count=3,
                training_batches=[1, 8],
                heldout_batches=[4],
                method="constant",
                residual_us=0,
                validation_status="pending",
                quantiles_us={
                    "a_step_prepare_us": dict(p50=10, p95=10),
                    "host_submit_us": dict(p50=2, p95=2),
                },
            ),
        )

    def load(self, doc, runtime=None, explicit=()):
        runtime = runtime or self.runtime()
        return HostOverheadProfile(
            doc,
            expected_runtime=runtime_fingerprint(
                runtime.plan, runtime.cluster, runtime.workers, runtime.hardware
            ),
            expected_environment=dict(
                execution_mode="eager", software_sha256="test-software"
            ),
            explicit_costs=explicit,
        )


class OperationProfileFixture:
    @classmethod
    def setUpClass(cls):
        HostProfileFixture.setUpClass()
        FixedHostFixture.setUpClass()
        cls.host = HostProfileFixture()
        cls.fixed = FixedHostFixture()

    def document(self, runtime):
        doc = json.loads(Path("tests/af/fixtures/operation_service.json").read_text())
        doc["runtime_fingerprint"] = runtime_fingerprint(
            runtime.plan, runtime.cluster, runtime.workers, runtime.hardware
        )
        doc["environment"] = {
            "execution_mode": "eager",
            "software_sha256": "test-software",
        }
        doc["evidence"]["kind"] = "synthetic_test"
        for op in doc["operations"].values():
            op["fixed_us"] = 11
        return doc

    def load(self, doc, explicit=()):
        runtime = self.host.runtime()
        return OperationServiceProfile(
            doc,
            expected_runtime=runtime_fingerprint(
                runtime.plan, runtime.cluster, runtime.workers, runtime.hardware
            ),
            expected_environment={
                "execution_mode": "eager",
                "software_sha256": "test-software",
            },
            explicit_costs=explicit,
        )

    def configured(self, profile=None, **options):
        runtime = self.host.runtime()
        doc = copy.deepcopy(runtime.doc)
        profile = profile or self.document(runtime)
        doc["af"].update(
            operation_service_profile=profile,
            calibration_environment=profile["environment"],
            **options,
        )
        return build_model_runtime(doc, hardware=runtime.hardware, model=runtime.model)

    def sampled_profile(self):
        doc = self.document(self.host.runtime())
        doc["schema_version"] = 2
        doc["evidence"]["heldout_batches"] = [2]
        doc["operations"]["sampled_token_readback"] = dict(
            fixed_us=11,
            coverage=["sampled_token_readback"],
            input_ready="sampled_tokens_device_ready",
            completion="cpu_api_return_data_ready",
            resource_model="calling_thread_single_inflight",
        )
        doc["evidence"]["statistics"]["sampled_token_readback"] = dict(
            p50_us=11, p95_us=11, samples=100
        )
        return doc

    def oracle(self):
        env, ex = self.fixed.executor(native_control_enabled=True)
        ex.operation_profile = self.load(self.document(self.host.runtime()))
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        block = ex.submit(0, [request]).blocks[0]
        node = ex._control_nodes(block)[0]
        return env, ex, block, node

    def input_profile(self):
        profile = self.document(self.host.runtime())
        profile["operations"]["a_input_prepare"] = dict(
            fixed_us=100,
            coverage=["a_input_prepare"],
            input_ready="prior_device_work_complete",
            completion="device_synchronize_after_operation",
            resource_model="calling_thread_single_inflight",
        )
        profile["evidence"]["statistics"]["a_input_prepare"] = dict(
            p50_us=100, p95_us=101, samples=200
        )
        return profile
