"""Af profile fixture."""

from __future__ import annotations

from TokenSim.af.calibration.host import (
    HOST_COVERAGE,
    HostOverheadProfile,
    runtime_fingerprint,
)
from TokenSim.af.factory import build_model_runtime
from tests.af.support import native_fixture as native


class HostProfileFixture:
    @classmethod
    def setUpClass(cls):
        native.NativeFixture.setUpClass()
        cls.fixture = native.NativeFixture()

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
