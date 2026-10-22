"""Execution boundaries conserve chip service and charge invocation once."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.execution import ExecutionAssumptions, add_invocation, with_execution
from scripts.lpx.native_search import build_runtime, streaming_settings
from scripts.lpx.config import Layout
from scripts.lpx.replay_engine import ReplayNode
from scripts.lpx.replay_profiles import NativeProfile, cached_profile, capture_profile
from scripts.lpx.streaming import duration
from scripts.search_lpx_native import main


class InvocationTest(unittest.TestCase):
    def test_parallel_roots_share_one_invocation_and_keep_all_service(self):
        nodes = (
            ReplayNode((), ("a",), "a", 0, 4e-6, 4e-6, 0),
            ReplayNode((), ("b",), "b", 1e-6, 7e-6, 7e-6, 1),
            ReplayNode((0, 1), ("a",), "a", 0, 3e-6, 3e-6, 2),
        )
        profile = NativeProfile(nodes, dict(native_duration_s=11e-6))
        result = add_invocation(profile, 2)
        self.assertEqual(result.nodes[-1].resources, ())
        self.assertEqual(result.nodes[-1].occupancy, 0)
        self.assertEqual(result.nodes[0].predecessors, (3,))
        self.assertEqual(result.nodes[1].predecessors, (3,))
        self.assertEqual(result.nodes[2], nodes[2])
        for before, after in zip(nodes, result.nodes):
            self.assertEqual(before.duration, after.duration)
            self.assertEqual(before.occupancy, after.occupancy)
            self.assertEqual(before.resources, after.resources)
            self.assertEqual(before.delay, after.delay)
        self.assertAlmostEqual(duration(result.nodes), 13e-6)
        result.verify()
        self.assertIs(add_invocation(profile, 0), profile)

    def test_invalid_values_and_unrooted_graphs_fail(self):
        empty = NativeProfile((), dict(native_duration_s=0))
        for value in (-1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                add_invocation(empty, value)
        with self.assertRaises(ConfigurationError):
            add_invocation(empty, 1)
        for settings in (
            dict(mode="unknown"),
            dict(mode="logical", instruction_floor_us=0.1),
            dict(mode="compiled", kernel_launch_us=0.3),
            dict(invocation_us=-1),
            dict(instruction_floor_us=float("nan")),
            dict(c2c_hop_us=True),
        ):
            with self.subTest(settings=settings), self.assertRaises(ConfigurationError):
                ExecutionAssumptions(**settings)
        self.assertEqual(
            main(["--execution-model", "compiled", "--kernel-launch-us", "0.3"]),
            1,
        )


class ExecutionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = replace(
            cls.hardware.model("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
        )
        cls.layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=16,
            f_tp=16,
            a_pp=9,
            f_pp=9,
            f_ep=1,
        )

    def test_overrides_are_local_and_retain_physical_rate_parameters(self):
        config = ExecutionAssumptions(
            mode="compiled",
            instruction_floor_us=0.01,
            invocation_us=3,
            c2c_hop_us=0.35,
        )
        altered = with_execution(self.hardware, config)
        before = self.hardware.device("groq_lp30")
        after = altered.device("groq_lp30")
        self.assertEqual(before.analytical["kernel_launch_us"], 0.2)
        self.assertEqual(after.analytical["kernel_launch_us"], 0.01)
        self.assertEqual(before.peak_compute, after.peak_compute)
        self.assertEqual(
            before.memory_bandwidth_bytes_per_s, after.memory_bandwidth_bytes_per_s
        )
        self.assertEqual(before.on_chip_memory_bytes, after.on_chip_memory_bytes)
        self.assertIs(self.hardware.device("rubin"), altered.device("rubin"))
        self.assertIs(
            self.hardware.link("ethernet1600Gb"), altered.link("ethernet1600Gb")
        )
        self.assertEqual(self.hardware.link("groq_lp30_injection").latency_us, 0.05)
        self.assertEqual(altered.link("groq_lp30_injection").latency_us, 0.35)
        self.assertEqual(config.report()["effective_operator_startup_us"], 0.01)

    def test_native_compiled_profile_invocation_shift_and_cache_identity(self):
        config = ExecutionAssumptions(mode="compiled", instruction_floor_us=0.01)
        hardware = with_execution(self.hardware, config)
        runtime = build_runtime(hardware, self.model, self.layout)
        settings = dict(
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=10001,
            dp_rank=0,
            seed=42,
            sample=0,
            streaming=streaming_settings(self.layout),
        )
        baseline = capture_profile(runtime, "execution-test", **settings)
        with tempfile.TemporaryDirectory() as directory:
            invocation = cached_profile(
                runtime, "execution-test", directory, invocation_us=3, **settings
            )
            repeated = cached_profile(
                runtime, "execution-test", directory, invocation_us=3, **settings
            )
            plain = cached_profile(runtime, "execution-test", directory, **settings)
        self.assertEqual(invocation.metadata["invocation_count"], 1)
        self.assertEqual(len(invocation.nodes), len(baseline.nodes) + 1)
        self.assertAlmostEqual(
            invocation.metadata["modeled_duration_s"]
            - baseline.metadata["modeled_duration_s"],
            3e-6,
        )
        self.assertEqual(invocation.nodes, repeated.nodes)
        self.assertNotEqual(
            invocation.metadata["input_sha256"], plain.metadata["input_sha256"]
        )
        with patch.object(
            runtime, "workers", [type("Worker", (), {"hardware": "rubin"})()]
        ):
            with self.assertRaises(ConfigurationError):
                capture_profile(runtime, "execution-test", invocation_us=1, **settings)


if __name__ == "__main__":
    unittest.main()
