"""Operator dataset package behavior and failure paths."""

from __future__ import annotations

import tempfile
import unittest
import yaml
from pathlib import Path
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    OperatorDataValidationError,
    PackageMeta,
)
from tests.support.hardware import make_device, make_model
from tests.support.operator_data import SOURCE, ROOT, _meta, _gemm_rows


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
        with self.assertRaises(OperatorDataValidationError):
            OperatorDataPackage.from_rows(
                PackageMeta("unregistered", "TestGPU", "test"), {"gemm": rows}
            )

    def test_round_trip_through_parquet(self):
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        with tempfile.TemporaryDirectory() as tmpdir:
            old = OperatorDataPackage.from_rows(
                _meta(),
                {
                    "gemm": _gemm_rows(),
                    "elementwise": [
                        {
                            "op_name": "rmsnorm",
                            "dtype": "bf16",
                            "num_tokens": 1,
                            "hidden_size": 16,
                            "latency_us": 1,
                            "source_id": SOURCE.source_id,
                        }
                    ],
                },
            )
            old.write(tmpdir)
            package.write(tmpdir)
            loaded = OperatorDataPackage.load(tmpdir)
            self.assertEqual(set(loaded.tables), {"gemm"})
            meta_path = Path(tmpdir) / "generation_meta.yaml"
            meta = yaml.safe_load(meta_path.read_text())
            meta["tables"]["gemm"]["rows"] += 1
            meta_path.write_text(yaml.safe_dump(meta))
            with self.assertRaisesRegex(
                OperatorDataValidationError, "manifest declares"
            ):
                OperatorDataPackage.load(tmpdir)
        self.assertEqual(loaded.dataset_version, "test-v1")
        self.assertEqual(len(loaded.tables["gemm"]), len(_gemm_rows()))
        self.assertEqual(loaded.meta.table_rows["gemm"], len(_gemm_rows()))

    def test_registering_a_package_preserves_disk_backends_and_checks_identity(self):
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package.write(root / "TestGPU" / "requested")
            hardware = HardwareContext.in_memory([make_device()], [make_model()])
            hardware.operator_data_root = root
            hardware.register_package(package)
            self.assertEqual(
                set(hardware.available_backends("TestGPU")), {"test", "requested"}
            )
            self.assertIs(hardware.operator_package("TestGPU", "test"), package)
            with self.assertRaisesRegex(ConfigurationError, "declares backend"):
                hardware.operator_package("TestGPU", "requested")
