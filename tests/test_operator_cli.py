from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from TokenSim.operator_data.cli import main
from TokenSim.operator_data.package import OperatorDataPackage

NCCL_TEXT = """# nThread 1 nGpus 2
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
          65536         32768      half     sum      -1    12.50    5.24    5.24      0    12.4    5.28    5.28      0
        1048576        524288      half     sum      -1    30.00   34.95   34.95      0    29.9   35.07   35.07      0
       16777216       8388608      half     sum      -1   250.00   67.11   67.11      0   249.0   67.38   67.38      0
"""


def _run(*argv: str) -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv))
    assert code == 0, buffer.getvalue()
    text = buffer.getvalue()
    start = text.index("{")
    return json.loads(text[start:])


class OperatorCliTest(unittest.TestCase):
    def test_manifest_generate_validate_coverage_and_nccl_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = _run(
                "manifest",
                "--models",
                "llama-3-8b",
                "--tp",
                "1,2",
                "--batch",
                "1,8",
                "--prefill-tokens",
                "128",
                "--context",
                "512",
                "--out",
                str(root / "m.yaml"),
            )
            self.assertGreater(manifest["counts"]["gemm"], 0)

            package_dir = root / "pkg"
            generated = _run(
                "generate",
                "--device",
                "rtx_4090",
                "--manifest",
                str(root / "m.yaml"),
                "--out",
                str(package_dir),
                "--dataset-version",
                "unit-v1",
            )
            self.assertEqual(generated["rows"]["gemm"], manifest["counts"]["gemm"])
            self.assertTrue((package_dir / "gemm_analysis.parquet").is_file())
            self.assertTrue((package_dir / "shape_manifests" / "llama-3-8b.yaml").is_file())

            validated = _run("validate", str(package_dir))
            self.assertTrue(validated["valid"])
            self.assertEqual(validated["dataset_version"], "unit-v1")

            nccl_file = root / "all_reduce.txt"
            nccl_file.write_text(NCCL_TEXT)
            imported = _run(
                "import-nccl",
                "--device",
                "rtx_4090",
                "--backend",
                "analytical",
                "--file",
                str(nccl_file),
                "--operation",
                "all_reduce",
                "--group-size",
                "2",
                "--out",
                str(package_dir),
            )
            self.assertEqual(imported["rows_added"], 3)
            package = OperatorDataPackage.load(package_dir)
            self.assertEqual(len(package.tables["collective"]), 3)
            # merged package keeps the analytical rows and the measured source
            self.assertIn("gemm", package.tables)
            self.assertTrue(any(s.method == "measured" for s in package.meta.sources.values()))

    def test_coverage_against_repository_package(self):
        summary = _run(
            "coverage",
            "--device",
            "a100_sxm_80g",
            "--backend",
            "trtllm",
            "--models",
            "llama-3-8b",
            "--tp",
            "1",
            "--batch",
            "1,16",
            "--prefill-tokens",
            "256",
            "--context",
            "1024",
            "--no-collectives",
        )
        gemm = summary["tables"]["gemm"]
        self.assertGreater(gemm.get("exact", 0) + gemm.get("interpolated", 0), 0)
        # AIConfigurator ships no elementwise measurements
        self.assertEqual(set(summary["tables"]["elementwise"]), {"missing"})


if __name__ == "__main__":
    unittest.main()
