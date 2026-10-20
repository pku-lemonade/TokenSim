"""Execution assumptions agree across complete prefill and replay workers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.hardware_overrides import communication_hypothesis
from scripts.lpx.native_config import configuration
from scripts.lpx.runner import evaluate
from tests import test_lpx_native


class ExecutionSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_lpx_native.NativeLPXTest.setUpClass()
        cls.fixture = test_lpx_native.NativeLPXTest()

    def prefill(self, device, mode, overhead):
        f = self.fixture
        layout = Layout(a_device=device, f_device=device, a_tp=2, f_ep=2)
        runtime = configuration(
            f.hardware,
            f.model,
            layout,
            namespace="settings",
            execution_mode=mode,
            graph_kernel_us=overhead,
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        env.run(
            until=env.process(
                service.forward([session], phase="prefill", query_tokens=2)
            )
        )
        return env.now, service

    def test_gpu_prefill_honors_graph_kernel_assumption_without_mutating_catalog(self):
        original = self.fixture.hardware.device("rubin").analytical["kernel_launch_us"]
        eager, eager_service = self.prefill("rubin", "eager", 0.1)
        graph, graph_service = self.prefill("rubin", "cuda_graph", 0.1)
        self.assertLess(graph, eager)
        for service, expected in ((eager_service, original), (graph_service, 0.1)):
            self.assertTrue(service.prefill.backends)
            for backend in service.prefill.backends.values():
                self.assertEqual(
                    backend.device.analytical["kernel_launch_us"], expected
                )
            self.assertEqual(service.sessions[0].cached_context, 18)
        self.assertEqual(
            self.fixture.hardware.device("rubin").analytical["kernel_launch_us"],
            original,
        )

    def test_gpu_graph_setting_does_not_override_lpu_execution_costs(self):
        eager, _ = self.prefill("groq_lp30", "eager", 0.1)
        graph, service = self.prefill("groq_lp30", "cuda_graph", 0.1)
        self.assertEqual(eager, graph)
        expected = self.fixture.hardware.device("groq_lp30").analytical[
            "kernel_launch_us"
        ]
        self.assertTrue(
            all(
                b.device.analytical["kernel_launch_us"] == expected
                for b in service.prefill.backends.values()
            )
        )

    def test_communication_override_validates_and_preserves_catalog(self):
        original = self.fixture.hardware
        self.assertIs(communication_hypothesis(original), original)
        changed = communication_hypothesis(original, 0.35, 0)
        for name in ("groq_lp30_c2c_112g", "groq_lp30_injection"):
            before, after = original.link(name), changed.link(name)
            self.assertEqual(after.latency_us, 0.35)
            self.assertEqual(
                after.bandwidth_per_direction_bytes_per_s,
                before.bandwidth_per_direction_bytes_per_s,
            )
            self.assertEqual(after.grade, "D")
            self.assertEqual(before.latency_us, 0.05)
        self.assertEqual(changed.link("nvlink6"), original.link("nvlink6"))
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                communication_hypothesis(original, value)

    def test_native_and_replay_workers_apply_overrides_and_separate_checkpoints(self):
        f = self.fixture
        workload = Workload(cached_tokens=16, new_tokens=2, output_tokens=4)
        case = Case(
            "lpu-settings",
            "rubin",
            Layout(a_device="groq_lp30", f_device="groq_lp30", a_tp=2, f_ep=2),
        )
        with TemporaryDirectory() as directory:
            for backend in ("native", "sampled-replay"):
                values = []
                for latency in (0.05, 0.35):
                    settings = dict(
                        warmup_steps=1,
                        measurement_seconds=1e-9,
                        execution_backend=backend,
                        replay_samples=1,
                        replay_cache=str(Path(directory) / "profiles"),
                        communication_overrides={"latency_us": latency},
                        checkpoint_output=directory,
                        result_format="reference",
                    )
                    job = (f.model, workload, case, 1, settings)
                    result = evaluate(job)
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["point_audit"], "passed")
                    self.assertTrue(evaluate(job)["reused"])
                    values.append(result)
                self.assertNotEqual(
                    values[0]["checkpoint_key"], values[1]["checkpoint_key"]
                )
                self.assertLess(
                    values[1]["tokens_per_s_per_user"],
                    values[0]["tokens_per_s_per_user"],
                )
