"""Af operation fixture."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from TokenSim.af.calibration.host import runtime_fingerprint
from TokenSim.af.calibration.services import OperationServiceProfile
from TokenSim.af.factory import build_model_runtime
from tests.af.support import host_fixture as fixed, profile_fixture as host


class OperationProfileFixture:
    @classmethod
    def setUpClass(cls):
        host.HostProfileFixture.setUpClass()
        fixed.FixedHostFixture.setUpClass()
        cls.host = host.HostProfileFixture()
        cls.fixed = fixed.FixedHostFixture()

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
        block = ex.submit(0, [request], "decode").blocks[0]
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
