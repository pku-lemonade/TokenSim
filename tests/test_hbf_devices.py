"""Experimental GPU memory is selected independently from storage media."""

from pathlib import Path
import unittest

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.hardware import HardwareContext
from TokenSim.operator_data.analytical.gpu import GPURooflineModel
from tests.support.hardware import make_model

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30


class HBFDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")

    def test_memory_variants_preserve_compute_and_links_with_separate_provenance(self):
        expected = {
            "h100_sxm_hbf1": ("h100_sxm", "GDDR7", 32, 1.792),
            "h100_sxm_hbf1_48gib": ("h100_sxm", "GDDR7", 48, 1.344),
            "h100_sxm_hbf2": ("h100_sxm", "HBM3e", 48, 1.5),
            "h200_sxm_hbf1": ("h200_sxm", "GDDR7", 32, 1.792),
            "h200_sxm_hbf2": ("h200_sxm", "HBM3e", 48, 1.5),
            "b200_sxm_hbf1": ("b200_sxm", "GDDR7", 64, 3.584),
            "b200_sxm_hbf1_48gib": ("b200_sxm", "GDDR7", 96, 2.688),
            "b200_sxm_hbf2": ("b200_sxm", "HBM3e", 96, 4.0),
        }
        for device_id, (base_id, kind, gib, bandwidth) in expected.items():
            with self.subTest(device_id=device_id):
                device, base = self.hardware.device(device_id), self.hardware.device(
                    base_id
                )
                self.assertEqual(device.memory_kind, kind)
                self.assertEqual(device.memory_capacity_bytes.value, gib * GIB)
                self.assertEqual(
                    device.memory_bandwidth_bytes_per_s.value, bandwidth * 1e12
                )
                self.assertEqual(device.memory_capacity_bytes.grade, "D")
                self.assertEqual(device.memory_bandwidth_bytes_per_s.grade, "D")
                self.assertEqual(device.analytical_grade, "D")
                self.assertEqual(device.peak_compute, base.peak_compute)
                self.assertEqual(device.scale_up_link, base.scale_up_link)
                self.assertEqual(device.host_link, base.host_link)
                self.assertIsNone(self.hardware.operator_package(device_id))

    def test_storage_selection_does_not_change_device_or_kv_budget(self):
        base = self.hardware.device("H100")
        cache = CacheConfig(16, base, make_model())
        KVTransferConfig.from_file(
            ROOT / "data/kv_transfer/mooncake_store_h100_hbf1.json"
        )
        self.assertIs(self.hardware.device("H100"), base)
        self.assertEqual(base.memory_capacity_bytes.value, 80 * GIB)
        self.assertEqual(
            CacheConfig(16, base, make_model()).num_gpu_blocks, cache.num_gpu_blocks
        )
        variant = self.hardware.device("h100_sxm_hbf1")
        self.assertLess(
            CacheConfig(16, variant, make_model()).num_gpu_blocks, cache.num_gpu_blocks
        )

    def test_analytical_latency_responds_to_memory_variant_bandwidth(self):
        estimator = GPURooflineModel()
        key = {"m": 1, "n": 4096, "k": 4096, "dtype": "fp16"}
        base = estimator.estimate("gemm", key, self.hardware.device("H100"))
        variant = estimator.estimate("gemm", key, self.hardware.device("h100_sxm_hbf1"))
        self.assertEqual(base.work, variant.work)
        self.assertEqual(base.compute_time_us, variant.compute_time_us)
        self.assertGreater(variant.memory_time_us, base.memory_time_us)
        self.assertGreater(variant.latency_us, base.latency_us)
