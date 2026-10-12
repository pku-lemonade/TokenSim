"""Operator dataset workload behavior and failure paths."""

from __future__ import annotations

import unittest
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceCatalog
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.workload import (
    context_attention_work,
    elementwise_work,
    gemm_work,
    generation_attention_work,
    moe_work,
)
from tests.support.operator_data import ROOT


class WorkloadFormulaTest(unittest.TestCase):
    def test_gemm_flops_and_bytes(self):
        work = gemm_work(4, 6144, 4096, "fp16")
        self.assertEqual(work.flops, 2 * 4 * 6144 * 4096)
        self.assertEqual(work.weight_bytes, 6144 * 4096 * 2)
        self.assertEqual(work.activation_bytes, (4 * 4096 + 4 * 6144) * 2)

    def test_weight_only_int4_keeps_fp16_activations(self):
        work = gemm_work(1, 1024, 1024, "int4_wo")
        self.assertEqual(work.weight_bytes, 1024 * 1024 * 0.5)
        self.assertEqual(work.activation_bytes, (1024 + 1024) * 2)

    def test_attention_accounts_for_gqa_and_causal_mask(self):
        prefill = context_attention_work(1, 1024, 32, 8, 128, "fp16", "fp16")
        self.assertEqual(prefill.flops, 4 * 1024 * 1024 * 4096 * 0.5)
        self.assertEqual(prefill.kv_bytes, 1024 * 4 * 1024 * 2)
        decode = generation_attention_work(8, 4096, 32, 8, 128, "fp16", "fp8")
        self.assertEqual(decode.flops, 4 * 8 * 4096 * 4096)
        self.assertEqual(decode.kv_bytes, 8 * (2 * 4096 * 1024 * 1 + 2 * 1024 * 1))
        windowed = generation_attention_work(
            1, 8192, 32, 8, 128, "fp16", "fp16", window_size=1024
        )
        self.assertEqual(windowed.flops, 4 * 1024 * 4096)

    def test_moe_work_only_counts_active_local_experts(self):
        work = moe_work(64, 4096, 14336, 2, 8, tp_size=1, ep_size=4, dtype="bf16")
        self.assertEqual(work.details["local_experts"], 2)
        self.assertEqual(work.details["local_pairs"], 32)
        self.assertEqual(work.flops, 32 * 2 * 4096 * 14336 * 3)
        single = moe_work(1, 4096, 14336, 2, 8, 1, 1, "bf16")
        self.assertEqual(single.details["active_local_experts"], 2)

    def test_elementwise_rejects_unknown_ops(self):
        self.assertGreater(
            elementwise_work("rmsnorm", 4, 4096, "fp16").activation_bytes, 0
        )
        with self.assertRaises(ValueError):
            elementwise_work("softmax", 4, 4096, "fp16")


class AnalyticalModelTest(unittest.TestCase):
    def test_gpu_and_groq_models_produce_bounded_estimates(self):
        catalog = DeviceCatalog.load(ROOT / "data/devices")
        for name in ("a100_sxm_80g", "groqchip_v1", "rtx_4090"):
            device = catalog.get(name)
            model = analytical_model_for(device.family)
            small = model.estimate(
                "gemm", {"dtype": "fp16", "m": 1, "n": 4096, "k": 4096}, device
            )
            large = model.estimate(
                "gemm", {"dtype": "fp16", "m": 4096, "n": 4096, "k": 4096}, device
            )
            self.assertGreater(large.latency_us, small.latency_us)
            self.assertEqual(large.bottleneck, "compute")
            roofline_floor = 2 * 4096**3 / device.peak_compute_for("fp16").value * 1e6
            self.assertGreaterEqual(large.latency_us, roofline_floor)

    def test_groq_reports_sram_fit(self):
        device = DeviceCatalog.load(ROOT / "data/devices").get("groqchip_v1")
        model = analytical_model_for(device.family)
        fits = model.estimate(
            "gemm", {"dtype": "int8", "m": 1, "n": 4096, "k": 4096}, device
        )
        too_big = model.estimate(
            "gemm", {"dtype": "fp16", "m": 1, "n": 28672, "k": 8192}, device
        )
        self.assertTrue(fits.details["fits_on_chip"])
        self.assertFalse(too_big.details["fits_on_chip"])

    def test_calibration_is_affine_and_rejects_negative(self):
        self.assertEqual(Calibration(2.0, 3.0).apply(10.0), 23.0)
        with self.assertRaises(ConfigurationError):
            Calibration(1.0, -100.0).apply(10.0)
