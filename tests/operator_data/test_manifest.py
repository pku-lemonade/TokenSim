"""Operator dataset manifest behavior and failure paths."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from TokenSim.operator_data.analytical import Calibration
from TokenSim.operator_data.coverage import coverage_report
from TokenSim.operator_data.generator import generate_analytical_package, merge_packages
from TokenSim.operator_data.lookup import OperatorLookup
from TokenSim.operator_data.manifest import ShapeManifest, WorkloadGrid, build_manifest
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.support.hardware import moe_test_model, make_device, make_model
from tests.support.operator_data import _meta, _gemm_rows


class ManifestGeneratorTest(unittest.TestCase):
    def test_manifest_covers_dense_and_moe_operators(self):
        grid = WorkloadGrid(
            batch_sizes=(1, 8),
            prefill_tokens=(128,),
            context_lens=(512,),
            tp_sizes=(1, 2),
            ep_sizes=(1, 2),
            group_sizes=(2,),
            message_bytes=(1 << 20,),
        )
        manifest = build_manifest("unit", [make_model(), moe_test_model()], grid)
        counts = manifest.count()
        for table in (
            "gemm",
            "context_attention",
            "generation_attention",
            "moe",
            "elementwise",
            "collective",
        ):
            self.assertGreater(counts[table], 0, table)
        roles = {r for rs in manifest.roles["gemm"].values() for r in rs}
        self.assertTrue(any(r.endswith("router") for r in roles))
        self.assertTrue(any(r.endswith("lm_head") for r in roles))
        self.assertTrue(any(r.endswith("shared_down") for r in roles))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = manifest.write(Path(tmpdir) / "m.yaml")
            loaded = ShapeManifest.load(path)
        self.assertEqual(loaded.count(), counts)
        # Weight-quantized GEMMs; norms, attention and collectives on activations.
        quantized = replace(make_model(), dtype="fp8", activation_dtype="bf16")
        quantized_manifest = build_manifest("unit", [quantized], grid)
        dtypes = {
            table: {key.get("dtype", key.get("attn_dtype")) for key in keys}
            for table, keys in quantized_manifest.keys.items()
        }
        self.assertEqual(dtypes["gemm"], {"fp8"})
        for table in ("elementwise", "context_attention", "collective"):
            self.assertEqual(dtypes[table], {"bf16"}, table)

    def test_generate_analytical_package_and_coverage(self):
        device = make_device()
        grid = WorkloadGrid(
            batch_sizes=(1, 4),
            prefill_tokens=(64,),
            context_lens=(256,),
            group_sizes=(2,),
            message_bytes=(1 << 16,),
        )
        manifest = build_manifest("unit", [make_model()], grid)
        report = generate_analytical_package(
            device, manifest, calibrations={"gemm": Calibration(1.1, 0.0, "unit")}
        )
        package = report.package
        self.assertEqual(
            report.skipped.get("collective"), manifest.count()["collective"]
        )
        self.assertIn("gemm", package.tables)
        self.assertIn("gemm", package.analysis)
        coverage = coverage_report(package, manifest)
        summary = coverage.summary()
        self.assertEqual(summary["tables"]["gemm"], {"exact": manifest.count()["gemm"]})
        self.assertEqual(
            summary["tables"]["collective"], {"missing": manifest.count()["collective"]}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            package.write(tmpdir)
            loaded = OperatorDataPackage.load(tmpdir)
        self.assertEqual(loaded.meta.calibration["gemm"]["calibration_id"], "unit")

    def test_merge_prefers_measured_rows(self):
        analytical = OperatorDataPackage.from_rows(
            PackageMeta(
                "a",
                "TestGPU",
                "test",
                sources={"x": SourceRecord("x", "D", "analytical")},
            ),
            {
                "gemm": [
                    {
                        "dtype": "bf16",
                        "m": 1,
                        "n": 1024,
                        "k": 1024,
                        "latency_us": 99.0,
                        "source_id": "x",
                    }
                ]
            },
        )
        measured = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        merged = merge_packages(analytical, measured, prefer="extra")
        lookup = OperatorLookup(merged)
        self.assertEqual(
            lookup.lookup(
                "gemm", {"dtype": "bf16", "m": 1, "n": 1024, "k": 1024}
            ).source_id,
            "measured:test",
        )
