from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceCatalog
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.coverage import coverage_report
from TokenSim.operator_data.generator import generate_analytical_package, merge_packages
from TokenSim.operator_data.importers.nccl_tests import parse_nccl_tests_output
from TokenSim.operator_data.lookup import LookupPolicy, MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.manifest import ShapeManifest, WorkloadGrid, build_manifest
from TokenSim.operator_data.package import OperatorDataPackage, OperatorDataValidationError, PackageMeta, SourceRecord
from TokenSim.operator_data.workload import (
    context_attention_work,
    elementwise_work,
    gemm_work,
    generation_attention_work,
    moe_work,
)
from tests.hardware_fixtures import moe_test_model, test_device, test_model

SOURCE = SourceRecord("measured:test", "A", "measured", device="TestGPU", backend="test")


def _meta(**calibration) -> PackageMeta:
    return PackageMeta("test-v1", "TestGPU", "test", sources={"measured:test": SOURCE}, calibration=calibration)


def _gemm_rows():
    return [
        {"dtype": "bf16", "m": m, "n": n, "k": k, "latency_us": float(m * n * k) / 1e6 + 5.0, "source_id": "measured:test"}
        for m in (1, 16, 256)
        for n in (1024, 4096)
        for k in (1024, 4096)
    ]


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
        windowed = generation_attention_work(1, 8192, 32, 8, 128, "fp16", "fp16", window_size=1024)
        self.assertEqual(windowed.flops, 4 * 1024 * 4096)

    def test_moe_work_only_counts_active_local_experts(self):
        work = moe_work(64, 4096, 14336, 2, 8, tp_size=1, ep_size=4, dtype="bf16")
        self.assertEqual(work.details["local_experts"], 2)
        self.assertEqual(work.details["local_pairs"], 32)
        self.assertEqual(work.flops, 32 * 2 * 4096 * 14336 * 3)
        single = moe_work(1, 4096, 14336, 2, 8, 1, 1, "bf16")
        self.assertEqual(single.details["active_local_experts"], 2)

    def test_elementwise_rejects_unknown_ops(self):
        self.assertGreater(elementwise_work("rmsnorm", 4, 4096, "fp16").activation_bytes, 0)
        with self.assertRaises(ValueError):
            elementwise_work("softmax", 4, 4096, "fp16")


class AnalyticalModelTest(unittest.TestCase):
    def test_gpu_and_groq_models_produce_bounded_estimates(self):
        catalog = DeviceCatalog.load("data/devices")
        for name in ("a100_sxm_80g", "groqchip_v1", "rtx_4090"):
            device = catalog.get(name)
            model = analytical_model_for(device.family)
            small = model.estimate("gemm", {"dtype": "fp16", "m": 1, "n": 4096, "k": 4096}, device)
            large = model.estimate("gemm", {"dtype": "fp16", "m": 4096, "n": 4096, "k": 4096}, device)
            self.assertGreater(large.latency_us, small.latency_us)
            self.assertEqual(large.bottleneck, "compute")
            roofline_floor = 2 * 4096**3 / device.peak_compute_for("fp16").value * 1e6
            self.assertGreaterEqual(large.latency_us, roofline_floor)

    def test_groq_reports_sram_fit(self):
        device = DeviceCatalog.load("data/devices").get("groqchip_v1")
        model = analytical_model_for(device.family)
        fits = model.estimate("gemm", {"dtype": "int8", "m": 1, "n": 4096, "k": 4096}, device)
        too_big = model.estimate("gemm", {"dtype": "fp16", "m": 1, "n": 28672, "k": 8192}, device)
        self.assertTrue(fits.details["fits_on_chip"])
        self.assertFalse(too_big.details["fits_on_chip"])

    def test_calibration_is_affine_and_rejects_negative(self):
        self.assertEqual(Calibration(2.0, 3.0).apply(10.0), 23.0)
        with self.assertRaises(ConfigurationError):
            Calibration(1.0, -100.0).apply(10.0)


class PackageValidationTest(unittest.TestCase):
    def test_duplicate_keys_and_bad_latency_are_rejected(self):
        rows = _gemm_rows()
        with self.assertRaises(OperatorDataValidationError):
            OperatorDataPackage.from_rows(_meta(), {"gemm": rows + [rows[0]]})
        bad = dict(rows[0], latency_us=float("nan"))
        with self.assertRaises(OperatorDataValidationError):
            OperatorDataPackage.from_rows(_meta(), {"gemm": [bad]})
        unknown_source = dict(rows[0], source_id="nobody")
        with self.assertRaises(OperatorDataValidationError):
            OperatorDataPackage.from_rows(_meta(), {"gemm": [unknown_source]})

    def test_round_trip_through_parquet(self):
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        with tempfile.TemporaryDirectory() as tmpdir:
            package.write(tmpdir)
            loaded = OperatorDataPackage.load(tmpdir)
        self.assertEqual(loaded.dataset_version, "test-v1")
        self.assertEqual(len(loaded.tables["gemm"]), len(_gemm_rows()))
        self.assertEqual(loaded.meta.table_rows["gemm"], len(_gemm_rows()))

    def test_repository_packages_validate(self):
        root = Path("data/operator_data")
        packages = [p for p in root.glob("*/*/generation_meta.yaml")]
        self.assertTrue(packages, "expected checked-in operator packages")
        for meta_path in packages:
            package = OperatorDataPackage.load(meta_path.parent)
            self.assertEqual(package.device_id, meta_path.parent.parent.name)


class LookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.package = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        self.lookup = OperatorLookup(self.package)

    def test_exact_interpolated_and_extrapolated(self):
        exact = self.lookup.lookup("gemm", {"dtype": "bfloat16", "m": 16, "n": 1024, "k": 1024})
        self.assertEqual(exact.match_type, "exact")
        mid = self.lookup.lookup("gemm", {"dtype": "bf16", "m": 64, "n": 1024, "k": 1024})
        self.assertEqual(mid.match_type, "interpolated")
        lo = self.lookup.lookup("gemm", {"dtype": "bf16", "m": 16, "n": 1024, "k": 1024}).latency_us
        hi = self.lookup.lookup("gemm", {"dtype": "bf16", "m": 256, "n": 1024, "k": 1024}).latency_us
        self.assertTrue(lo < mid.latency_us < hi)
        two_axis = self.lookup.lookup("gemm", {"dtype": "bf16", "m": 64, "n": 2048, "k": 2048})
        self.assertEqual(two_axis.match_type, "interpolated")
        beyond = self.lookup.lookup("gemm", {"dtype": "bf16", "m": 1024, "n": 1024, "k": 1024})
        self.assertEqual(beyond.match_type, "extrapolated")
        self.assertGreater(beyond.latency_us, hi)

    def test_missing_discrete_key_and_disabled_extrapolation_fail(self):
        with self.assertRaises(MissingOperatorDataError):
            self.lookup.lookup("gemm", {"dtype": "fp8", "m": 16, "n": 1024, "k": 1024})
        strict = OperatorLookup(self.package, LookupPolicy(extrapolate="none"))
        with self.assertRaises(MissingOperatorDataError):
            strict.lookup("gemm", {"dtype": "bf16", "m": 1024, "n": 1024, "k": 1024})
        with self.assertRaises(MissingOperatorDataError):
            self.lookup.lookup("gemm", {"dtype": "bf16", "m": 1024 * 1024, "n": 1024, "k": 1024})
        with self.assertRaises(MissingOperatorDataError):
            self.lookup.lookup("gemm", {"dtype": "bf16", "m": 1})


class ManifestGeneratorTest(unittest.TestCase):
    def test_manifest_covers_dense_and_moe_operators(self):
        grid = WorkloadGrid(batch_sizes=(1, 8), prefill_tokens=(128,), context_lens=(512,), tp_sizes=(1, 2), ep_sizes=(1, 2), group_sizes=(2,), message_bytes=(1 << 20,))
        manifest = build_manifest("unit", [test_model(), moe_test_model()], grid)
        counts = manifest.count()
        for table in ("gemm", "context_attention", "generation_attention", "moe", "elementwise", "collective"):
            self.assertGreater(counts[table], 0, table)
        roles = {r for rs in manifest.roles["gemm"].values() for r in rs}
        self.assertTrue(any(r.endswith("router") for r in roles))
        self.assertTrue(any(r.endswith("lm_head") for r in roles))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = manifest.write(Path(tmpdir) / "m.yaml")
            loaded = ShapeManifest.load(path)
        self.assertEqual(loaded.count(), counts)

    def test_generate_analytical_package_and_coverage(self):
        device = test_device()
        grid = WorkloadGrid(batch_sizes=(1, 4), prefill_tokens=(64,), context_lens=(256,), group_sizes=(2,), message_bytes=(1 << 16,))
        manifest = build_manifest("unit", [test_model()], grid)
        report = generate_analytical_package(device, manifest, calibrations={"gemm": Calibration(1.1, 0.0, "unit")})
        package = report.package
        self.assertEqual(report.skipped.get("collective"), manifest.count()["collective"])
        self.assertIn("gemm", package.tables)
        self.assertIn("gemm", package.analysis)
        coverage = coverage_report(package, manifest)
        summary = coverage.summary()
        self.assertEqual(summary["tables"]["gemm"], {"exact": manifest.count()["gemm"]})
        self.assertEqual(summary["tables"]["collective"], {"missing": manifest.count()["collective"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            package.write(tmpdir)
            loaded = OperatorDataPackage.load(tmpdir)
        self.assertEqual(loaded.meta.calibration["gemm"]["calibration_id"], "unit")

    def test_merge_prefers_measured_rows(self):
        analytical = OperatorDataPackage.from_rows(
            PackageMeta("a", "TestGPU", "test", sources={"x": SourceRecord("x", "D", "analytical")}),
            {"gemm": [{"dtype": "bf16", "m": 1, "n": 1024, "k": 1024, "latency_us": 99.0, "source_id": "x"}]},
        )
        measured = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        merged = merge_packages(analytical, measured, prefer="extra")
        lookup = OperatorLookup(merged)
        self.assertEqual(lookup.lookup("gemm", {"dtype": "bf16", "m": 1, "n": 1024, "k": 1024}).source_id, "measured:test")


class NcclImportTest(unittest.TestCase):
    def test_parse_nccl_tests_output(self):
        text = """# nThread 1 nGpus 8
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
        1048576        262144     float     sum      -1    72.75   14.41   25.22      0    71.9   14.58   25.52      0
       33554432       8388608     float     sum      -1    491.6   68.26  119.46      0    489.9   68.49  119.87      0
# Avg bus bandwidth    : 72.3
"""
        rows = parse_nccl_tests_output(text, operation="all_reduce", group_size=8, nodes=1, source_id="nccl:unit")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dtype"], "fp32")
        self.assertEqual(rows[0]["latency_us"], 72.75)
        self.assertEqual(rows[1]["message_bytes"], 33554432)
        with self.assertRaises(ConfigurationError):
            parse_nccl_tests_output("# nothing here\n", operation="all_reduce", group_size=8, nodes=1, source_id="x")


if __name__ == "__main__":
    unittest.main()
