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

    def test_analytical_extrapolation_keeps_boundary_efficiency(self):
        # Measured rows are exactly half of a synthetic "model" that grows
        # linearly with m; past the range the ratio must stay 0.5.
        def reference(table, key):
            return 100.0 * key["m"] + 1000.0

        rows = [
            {"dtype": "bf16", "m": m, "n": 1024, "k": 1024, "latency_us": 0.5 * reference("gemm", {"m": m}), "source_id": "measured:test"}
            for m in (8, 64, 256)
        ]
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": rows})
        lookup = OperatorLookup(package, LookupPolicy(extrapolate="analytical"), analytical_scaler=reference)
        beyond = lookup.lookup("gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024})
        self.assertEqual(beyond.match_type, "extrapolated")
        self.assertIn("analytical_scaled", beyond.detail["flags"])
        self.assertAlmostEqual(beyond.latency_us, 0.5 * reference("gemm", {"m": 2048}))
        # below the range the same rule applies (linear "scale" would not shrink)
        below = lookup.lookup("gemm", {"dtype": "bf16", "m": 1, "n": 1024, "k": 1024})
        self.assertAlmostEqual(below.latency_us, 0.5 * reference("gemm", {"m": 1}))
        # a scaler that cannot price the key falls back to linear scaling
        fallback = OperatorLookup(package, LookupPolicy(extrapolate="analytical"), analytical_scaler=lambda table, key: None)
        linear = fallback.lookup("gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024})
        self.assertIn("linear_scaled", linear.detail["flags"])
        self.assertAlmostEqual(linear.latency_us, 0.5 * reference("gemm", {"m": 256}) * 8)
        held = OperatorLookup(package, LookupPolicy(extrapolate="hold")).lookup("gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024})
        self.assertAlmostEqual(held.latency_us, 0.5 * reference("gemm", {"m": 256}))

    def test_empirical_extrapolation_fits_sublinear_scaling(self):
        # Measured rows grow as m^0.5 (sub-linear); empirical mode should
        # recover this exponent and produce lower estimates than analytical.
        import math as _math
        rows = [
            {"dtype": "bf16", "m": m, "n": 1024, "k": 1024,
             "latency_us": 10.0 * _math.sqrt(m), "source_id": "measured:test"}
            for m in (4, 16, 64, 256)
        ]
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": rows})
        # Empirical mode (new default).
        empirical = OperatorLookup(package)
        result = empirical.lookup("gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024})
        self.assertEqual(result.match_type, "extrapolated")
        self.assertIn("empirical_scaled", result.detail["flags"])
        # The empirical fit should detect alpha~0.5, so the estimate at
        # m=2048 should be close to 10 * sqrt(2048) ≈ 452.5.
        expected = 10.0 * _math.sqrt(2048)
        self.assertAlmostEqual(result.latency_us, expected, delta=expected * 0.1)
        # In contrast, analytical (linear) would give: boundary_lat * (2048/256)
        # = 10*sqrt(256) * 8 = 160 * 8 = 1280, much higher.
        analytical = OperatorLookup(
            package, LookupPolicy(extrapolate="analytical"),
            analytical_scaler=lambda table, key: float(key["m"]),
        )
        ana_result = analytical.lookup("gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024})
        self.assertGreater(ana_result.latency_us, result.latency_us * 1.5)

    def test_empirical_falls_back_to_analytical_with_few_points(self):
        # Only 2 measured points — too few for fitting.
        def reference(table, key):
            return 100.0 * key["m"]
        rows = [
            {"dtype": "bf16", "m": m, "n": 1024, "k": 1024,
             "latency_us": 50.0 * m, "source_id": "measured:test"}
            for m in (8, 64)
        ]
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": rows})
        lookup = OperatorLookup(package, analytical_scaler=reference)
        result = lookup.lookup("gemm", {"dtype": "bf16", "m": 512, "n": 1024, "k": 1024})
        self.assertEqual(result.match_type, "extrapolated")
        # Should fall back to analytical since <3 points.
        self.assertIn("analytical_scaled", result.detail["flags"])

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


class AIConfiguratorImportTest(unittest.TestCase):
    def test_flat_collector_run_with_csv_staging_and_version_backfill(self):
        import csv
        from TokenSim.operator_data.importers.aiconfigurator import import_aiconfigurator_system

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # flat collector run: CSV staging file exactly as collector/helper.log_perf writes it
            with (root / "gemm_perf.txt").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["framework", "version", "device", "op_name", "kernel_source", "gemm_dtype", "m", "n", "k", "latency"])
                writer.writeheader()
                writer.writerow({"framework": "TRTLLM", "version": "1.3.0rc20", "device": "NVIDIA A100", "op_name": "gemm", "kernel_source": "torch_flow", "gemm_dtype": "bfloat16", "m": 1, "n": 4096, "k": 4096, "latency": 0.05})
                writer.writerow({"framework": "TRTLLM", "version": "1.3.0rc20", "device": "NVIDIA A100", "op_name": "gemm", "kernel_source": "cutlass", "gemm_dtype": "bfloat16", "m": 1, "n": 4096, "k": 4096, "latency": 0.04})
                writer.writerow({"framework": "TRTLLM", "version": "1.3.0rc20", "device": "NVIDIA A100", "op_name": "gemm", "kernel_source": "torch_flow", "gemm_dtype": "w4a8_mxfp4_mxfp8", "m": 1, "n": 4096, "k": 4096, "latency": 0.03})
            package = import_aiconfigurator_system(root, device_id="TestGPU", backend="trtllm", upstream_commit="abc")
            rows = {(r["dtype"], r["m"]): r for r in package.tables["gemm"]}
            self.assertAlmostEqual(rows[("bf16", 1)]["latency_us"], 40.0)  # fastest kernel wins, ms -> us
            self.assertEqual(rows[("bf16", 1)]["kernel"], "cutlass")
            self.assertIn(("mxfp4", 1), rows)
            source = next(iter(package.meta.sources.values()))
            self.assertEqual(source.method, "measured")
            self.assertTrue(source.source_id.startswith("aiconfigurator-collector:TestGPU:gemm:trtllm:1.3.0rc20"))
            self.assertEqual(package.meta.extra["aiconfigurator_layout"], "collector_run")

        with tempfile.TemporaryDirectory() as tmpdir:
            # upstream layout with a newer partial version that must not shadow the older full one
            root = Path(tmpdir) / "sys"
            import pyarrow as pa
            import pyarrow.parquet as pq

            def write(version, rows):
                path = root / "moe" / "trtllm" / version
                path.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist(rows), path / "moe_perf.parquet")

            base = {"framework": "TRTLLM", "device": "x", "op_name": "moe", "kernel_source": "k", "hidden_size": 4096, "inter_size": 1536, "topk": 8, "num_experts": 128, "moe_tp_size": 1, "moe_ep_size": 1, "distribution": "uniform"}
            write("1.3.0rc20", [{**base, "version": "1.3.0rc20", "moe_dtype": "bfloat16", "num_tokens": t, "latency": 1.0 * t} for t in (1, 8, 64)])
            write("1.3.0rc23", [{**base, "version": "1.3.0rc23", "moe_dtype": "bfloat16", "num_tokens": 8, "latency": 0.5}, {**base, "version": "1.3.0rc23", "moe_dtype": "w4a8_mxfp4_mxfp8", "num_tokens": 8, "latency": 0.2}])
            package = import_aiconfigurator_system(root, device_id="TestGPU", backend="trtllm")
            rows = {(r["dtype"], r["num_tokens"]): r for r in package.tables["moe"]}
            self.assertEqual(len(rows), 4)  # 3 bf16 points (one replaced) + 1 mxfp4
            self.assertAlmostEqual(rows[("bf16", 8)]["latency_us"], 500.0)  # newer version wins
            self.assertAlmostEqual(rows[("bf16", 64)]["latency_us"], 64000.0)  # older version backfills
            self.assertEqual(package.meta.extra["resolved_versions"]["moe"], ["1.3.0rc23", "1.3.0rc20"])


class EPAllToAllImportTest(unittest.TestCase):
    def test_deepep_tables_become_ep_all2all_rows(self):
        from TokenSim.operator_data.importers.aiconfigurator import import_aiconfigurator_system
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "sys"
            comm_vllm = root / "comm" / "vllm" / "0.24.0"
            comm_vllm.mkdir(parents=True)
            base = {"framework": "vLLM", "version": "0.24.0", "device": "x", "op_name": "moe_a2a", "kernel_source": "deepep", "comm_dtype": "default", "ep_size": 8, "node_num": 1, "hidden_size": 4096, "topk": 8, "num_experts": 256, "sms": 20, "notify_us": 0.0}
            rows = []
            for tokens, lat in ((16, 40.0), (64, 60.0)):
                rows.append({**base, "comm_backend": "deepep_ht", "phase": "dispatch", "num_tokens": tokens, "transmit_us": lat, "latency": lat})
                rows.append({**base, "comm_backend": "deepep_ll", "phase": "combine", "num_tokens": tokens, "transmit_us": lat / 2, "latency": lat / 2})
            pq.write_table(pa.Table.from_pylist(rows), comm_vllm / "moe_a2a_perf.parquet")
            comm_sglang = root / "comm" / "sglang" / "0.5.14"
            comm_sglang.mkdir(parents=True)
            wide = [{"framework": "sglang", "version": "0.5.14", "device": "x", "op_name": "ll", "node_num": 2, "kernel_source": "deepep", "hidden_size": 7168, "num_token": 8, "num_topk": 8, "num_experts": 256, "combine_avg_t_us": 30.0, "combine_bandwidth_gbps": 1.0, "dispatch_avg_t_us": 20.0, "dispatch_bandwidth_gbps": 1.0}]
            pq.write_table(pa.Table.from_pylist(wide), comm_sglang / "wideep_deepep_ll_perf.parquet")

            vllm_pkg = import_aiconfigurator_system(root, device_id="TestGPU", backend="vllm")
            rows = {(r["phase"], r["mode"], r["num_tokens"]): r for r in vllm_pkg.tables["ep_all2all"]}
            # latency column is already in microseconds; dtype 'default' -> bf16
            self.assertAlmostEqual(rows[("dispatch", "deepep_high_throughput", 16)]["latency_us"], 40.0)
            self.assertAlmostEqual(rows[("combine", "deepep_low_latency", 64)]["latency_us"], 30.0)
            self.assertEqual(rows[("dispatch", "deepep_high_throughput", 16)]["dtype"], "bf16")

            sglang_pkg = import_aiconfigurator_system(root, device_id="TestGPU", backend="sglang", gpus_per_node=4)
            rows = {(r["phase"], r["ep_size"], r["nodes"]): r for r in sglang_pkg.tables["ep_all2all"]}
            self.assertAlmostEqual(rows[("dispatch", 8, 2)]["latency_us"], 20.0)
            self.assertAlmostEqual(rows[("combine", 8, 2)]["latency_us"], 30.0)
            self.assertEqual(rows[("combine", 8, 2)]["mode"], "deepep_low_latency")


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
