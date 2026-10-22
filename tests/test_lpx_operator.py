"""LP30 unit service, activation traffic and live workspace boundaries."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.runtime.lpx_memory import (
    LP30Memory,
    validate_queries,
    workspace_components,
)
from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import DeviceCatalog, HardwareContext
from TokenSim.operator_data.analytical import analytical_model_for
from TokenSim.operator_data.analytical.lp30 import LP30Pipeline
from scripts.lpx_runtime import Layout, configuration
from tests.support.lpx import tiny_moe

ROOT = Path(__file__).resolve().parents[1]


class UnitAccountingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")

    def estimate(self, table, key, **policy):
        base = self.hardware.device("groq_lp30")
        device = replace(base, analytical={**base.analytical, "lpx_pipeline": policy})
        return analytical_model_for(device.family).estimate(table, key, device)

    def test_mixed_precision_counts_external_bytes_and_vector_casts(self):
        key = dict(dtype="fp8", m=3, n=64, k=128)
        estimate = self.estimate("gemm", key)
        self.assertEqual(estimate.work.weight_bytes, 64 * 128)
        self.assertEqual(estimate.work.activation_bytes, 3 * (64 + 128) * 2)
        self.assertEqual(estimate.details["conversion_elements"], 3 * (64 + 128))
        self.assertGreater(estimate.details["unit_service_us"]["vxm"], 0)
        slow = self.estimate("gemm", key, vector_ops_per_s=1e8)
        self.assertGreater(slow.latency_us, estimate.latency_us)
        self.assertEqual(
            slow.details["unit_service_us"]["mxm"],
            estimate.details["unit_service_us"]["mxm"],
        )

    def test_softmax_has_distinct_exponential_cost_and_context_scaling(self):
        key = dict(
            attn_dtype="bf16",
            kv_cache_dtype="fp8",
            batch_size=1,
            context_len=1000,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            window_size=0,
        )
        short = self.estimate("generation_attention", key)
        long = self.estimate("generation_attention", {**key, "context_len": 2000})
        self.assertEqual(short.details["exponential_operations"], 2000)
        self.assertEqual(long.details["exponential_operations"], 4000)
        self.assertGreater(
            long.details["vector_operations"], short.details["vector_operations"]
        )
        slower = self.estimate("generation_attention", key, exp_ops_per_s=1e7)
        self.assertGreater(slower.latency_us, short.latency_us)
        capped = self.estimate(
            "generation_attention", {**key, "context_len": 2000, "window_size": 1000}
        )
        self.assertEqual(capped.latency_us, short.latency_us)

    def test_unit_schedule_preserves_serial_upper_bound(self):
        key = dict(dtype="fp8", m=1, n=128, k=256)
        pipelined = self.estimate("gemm", key)
        serialized = self.estimate("gemm", key, schedule="serialized")
        services = pipelined.details["unit_service_us"]
        self.assertGreaterEqual(pipelined.latency_us, max(services.values()))
        self.assertLessEqual(pipelined.latency_us, serialized.latency_us)
        self.assertAlmostEqual(
            serialized.latency_us, sum(services.values()) + serialized.overhead_us
        )

    def test_logit_softcap_counts_vector_work_and_nonlinearity(self):
        estimate = self.estimate(
            "elementwise",
            dict(
                op_name="other_residual",
                dtype="bf16",
                num_tokens=2,
                hidden_size=128,
                lpx_vector_operation="logit_softcap",
            ),
        )
        self.assertEqual(estimate.details["exponential_operations"], 256)
        self.assertEqual(estimate.details["vector_operations"], 1024)

    def test_invalid_unit_rates_and_memory_bounds_are_rejected(self):
        for values in (
            {"exp_ops_per_s": 0},
            {"vector_efficiency": 2},
            {"schedule": "unknown"},
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                LP30Pipeline(**values)
        with self.assertRaises(ConfigurationError):
            LP30Memory(prefill_chunk=17, maximum_context=16)


class WorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")
        cls.model = tiny_moe(cls.hardware)

    def test_workspace_reuses_sequential_arenas_and_scales_prefill(self):
        c = SimpleNamespace(
            category="dense_ffn",
            shape=(("hidden_size", 128), ("local_intermediate_size", 256)),
        )
        one = dict(
            workspace_components(
                (c,), self.model, dict(prefill_chunk=1, reserve_bytes=0)
            )
        )
        many = dict(
            workspace_components(
                (c, c), self.model, dict(prefill_chunk=4, reserve_bytes=0)
            )
        )
        self.assertEqual(one["lp30_live_scratch"], (3 * 256 + 2 * 128) * 2)
        self.assertEqual(many["lp30_live_scratch"], one["lp30_live_scratch"] * 4)
        with self.assertRaises(ConfigurationError):
            validate_queries((LP30Memory(prefill_chunk=4),), 5, 1000)

    def test_reservations_limit_native_admission_and_forward_queries(self):
        device = self.hardware.device("groq_lp30")
        altered = replace(
            device,
            analytical={
                **device.analytical,
                "lpx_memory": {"prefill_chunk": 2, "maximum_context": 24},
            },
        )
        hardware = replace(
            self.hardware,
            devices=DeviceCatalog(
                altered if d.device_id == device.device_id else d
                for d in self.hardware.devices
            ),
        )
        runtime = configuration(
            hardware,
            self.model,
            Layout(a_device="groq_lp30", f_device="groq_lp30", a_tp=2, f_tp=2, f_ep=1),
            namespace="memory",
        )
        self.assertTrue(
            all(
                "lp30_live_scratch" in dict(s.protocol_buffers)
                for s in runtime.plan.physical_states
            )
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        with self.assertRaisesRegex(
            ConfigurationError, "reserved prefill chunk or maximum context"
        ):
            service.admit(0, 0, maximum_context=25, current_context=16)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        with self.assertRaisesRegex(
            ConfigurationError, "reserved prefill chunk or maximum context"
        ):
            env.run(
                until=env.process(
                    service.forward([session], phase="prefill", query_tokens=3)
                )
            )
        self.assertEqual(session.cached_context, 16)
        service.release(session)
        self.assertFalse(service.memory.leases)

    def test_pipeline_activation_precision_matches_model(self):
        device = self.hardware.device("groq_lp30")
        altered = replace(
            device,
            analytical={
                **device.analytical,
                "lpx_pipeline": {"activation_dtype": "fp8"},
            },
        )
        hardware = replace(
            self.hardware,
            devices=DeviceCatalog(
                altered if d.device_id == device.device_id else d
                for d in self.hardware.devices
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "activation dtype must match"):
            configuration(
                hardware,
                self.model,
                Layout(
                    a_device="groq_lp30", f_device="groq_lp30", a_tp=2, f_tp=2, f_ep=1
                ),
                namespace="precision",
            )
