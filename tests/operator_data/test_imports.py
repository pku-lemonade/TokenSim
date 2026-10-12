"""Operator dataset imports behavior and failure paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.importers.nccl_tests import parse_nccl_tests_output


class AIConfiguratorImportTest(unittest.TestCase):
    def test_flat_collector_run_with_csv_staging_and_version_backfill(self):
        import csv
        from TokenSim.operator_data.importers.aiconfigurator import (
            import_aiconfigurator_system,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # flat collector run: CSV staging file exactly as collector/helper.log_perf writes it
            with (root / "gemm_perf.txt").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "framework",
                        "version",
                        "device",
                        "op_name",
                        "kernel_source",
                        "gemm_dtype",
                        "m",
                        "n",
                        "k",
                        "latency",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "framework": "TRTLLM",
                        "version": "1.3.0rc20",
                        "device": "NVIDIA A100",
                        "op_name": "gemm",
                        "kernel_source": "torch_flow",
                        "gemm_dtype": "bfloat16",
                        "m": 1,
                        "n": 4096,
                        "k": 4096,
                        "latency": 0.05,
                    }
                )
                writer.writerow(
                    {
                        "framework": "TRTLLM",
                        "version": "1.3.0rc20",
                        "device": "NVIDIA A100",
                        "op_name": "gemm",
                        "kernel_source": "cutlass",
                        "gemm_dtype": "bfloat16",
                        "m": 1,
                        "n": 4096,
                        "k": 4096,
                        "latency": 0.04,
                    }
                )
                writer.writerow(
                    {
                        "framework": "TRTLLM",
                        "version": "1.3.0rc20",
                        "device": "NVIDIA A100",
                        "op_name": "gemm",
                        "kernel_source": "torch_flow",
                        "gemm_dtype": "w4a8_mxfp4_mxfp8",
                        "m": 1,
                        "n": 4096,
                        "k": 4096,
                        "latency": 0.03,
                    }
                )
            package = import_aiconfigurator_system(
                root, device_id="TestGPU", backend="trtllm", upstream_commit="abc"
            )
            rows = {(r["dtype"], r["m"]): r for r in package.tables["gemm"]}
            self.assertAlmostEqual(
                rows[("bf16", 1)]["latency_us"], 40.0
            )  # fastest kernel wins, ms -> us
            self.assertEqual(rows[("bf16", 1)]["kernel"], "cutlass")
            self.assertIn(("mxfp4", 1), rows)
            source = next(iter(package.meta.sources.values()))
            self.assertEqual(source.method, "measured")
            self.assertTrue(
                source.source_id.startswith(
                    "aiconfigurator-collector:TestGPU:gemm:trtllm:1.3.0rc20"
                )
            )
            self.assertEqual(
                package.meta.extra["aiconfigurator_layout"], "collector_run"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # upstream layout with a newer partial version that must not shadow the older full one
            root = Path(tmpdir) / "sys"
            import pyarrow as pa
            import pyarrow.parquet as pq

            def write(version, rows):
                path = root / "moe" / "trtllm" / version
                path.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist(rows), path / "moe_perf.parquet")

            base = {
                "framework": "TRTLLM",
                "device": "x",
                "op_name": "moe",
                "kernel_source": "k",
                "hidden_size": 4096,
                "inter_size": 1536,
                "topk": 8,
                "num_experts": 128,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "distribution": "uniform",
            }
            write(
                "1.3.0rc20",
                [
                    {
                        **base,
                        "version": "1.3.0rc20",
                        "moe_dtype": "bfloat16",
                        "num_tokens": t,
                        "latency": 1.0 * t,
                    }
                    for t in (1, 8, 64)
                ],
            )
            write(
                "1.3.0rc23",
                [
                    {
                        **base,
                        "version": "1.3.0rc23",
                        "moe_dtype": "bfloat16",
                        "num_tokens": 8,
                        "latency": 0.5,
                    },
                    {
                        **base,
                        "version": "1.3.0rc23",
                        "moe_dtype": "w4a8_mxfp4_mxfp8",
                        "num_tokens": 8,
                        "latency": 0.2,
                    },
                ],
            )
            package = import_aiconfigurator_system(
                root, device_id="TestGPU", backend="trtllm"
            )
            rows = {(r["dtype"], r["num_tokens"]): r for r in package.tables["moe"]}
            self.assertEqual(len(rows), 4)  # 3 bf16 points (one replaced) + 1 mxfp4
            self.assertAlmostEqual(
                rows[("bf16", 8)]["latency_us"], 500.0
            )  # newer version wins
            self.assertAlmostEqual(
                rows[("bf16", 64)]["latency_us"], 64000.0
            )  # older version backfills
            self.assertEqual(
                package.meta.extra["resolved_versions"]["moe"],
                ["1.3.0rc23", "1.3.0rc20"],
            )


class EPAllToAllImportTest(unittest.TestCase):
    def test_deepep_tables_become_ep_all2all_rows(self):
        from TokenSim.operator_data.importers.aiconfigurator import (
            import_aiconfigurator_system,
        )
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data" / "sys"
            comm_vllm = root / "comm" / "vllm" / "0.24.0"
            comm_vllm.mkdir(parents=True)
            base = {
                "framework": "vLLM",
                "version": "0.24.0",
                "device": "x",
                "op_name": "moe_a2a",
                "kernel_source": "deepep",
                "comm_dtype": "default",
                "ep_size": 8,
                "node_num": 1,
                "hidden_size": 4096,
                "topk": 8,
                "num_experts": 256,
                "sms": 20,
                "notify_us": 0.0,
            }
            rows = []
            for tokens, lat in ((16, 40.0), (64, 60.0)):
                rows.append(
                    {
                        **base,
                        "comm_backend": "deepep_ht",
                        "phase": "dispatch",
                        "num_tokens": tokens,
                        "transmit_us": lat,
                        "latency": lat,
                    }
                )
                rows.append(
                    {
                        **base,
                        "comm_backend": "deepep_ll",
                        "phase": "combine",
                        "num_tokens": tokens,
                        "transmit_us": lat / 2,
                        "latency": lat / 2,
                    }
                )
            pq.write_table(
                pa.Table.from_pylist(rows), comm_vllm / "moe_a2a_perf.parquet"
            )
            comm_sglang = root / "comm" / "sglang" / "0.5.14"
            comm_sglang.mkdir(parents=True)
            wide = [
                {
                    "framework": "sglang",
                    "version": "0.5.14",
                    "device": "x",
                    "op_name": "ll",
                    "node_num": 2,
                    "kernel_source": "deepep",
                    "hidden_size": 7168,
                    "num_token": 8,
                    "num_topk": 8,
                    "num_experts": 256,
                    "combine_avg_t_us": 30.0,
                    "combine_bandwidth_gbps": 1.0,
                    "dispatch_avg_t_us": 20.0,
                    "dispatch_bandwidth_gbps": 1.0,
                }
            ]
            pq.write_table(
                pa.Table.from_pylist(wide),
                comm_sglang / "wideep_deepep_ll_perf.parquet",
            )

            vllm_pkg = import_aiconfigurator_system(
                root, device_id="TestGPU", backend="vllm"
            )
            rows = {
                (r["phase"], r["mode"], r["num_tokens"]): r
                for r in vllm_pkg.tables["ep_all2all"]
            }
            # latency column is already in microseconds; dtype 'default' -> bf16
            self.assertAlmostEqual(
                rows[("dispatch", "deepep_high_throughput", 16)]["latency_us"], 40.0
            )
            self.assertAlmostEqual(
                rows[("combine", "deepep_low_latency", 64)]["latency_us"], 30.0
            )
            self.assertEqual(
                rows[("dispatch", "deepep_high_throughput", 16)]["dtype"], "bf16"
            )

            # The node width comes from the upstream system description.
            with self.assertRaises(ConfigurationError):
                import_aiconfigurator_system(
                    root, device_id="TestGPU", backend="sglang"
                )
            (Path(tmpdir) / "sys.yaml").write_text(
                "data_dir: data/sys\nnode:\n  num_gpus_per_node: 4\n"
            )
            sglang_pkg = import_aiconfigurator_system(
                root, device_id="TestGPU", backend="sglang"
            )
            rows = {
                (r["phase"], r["ep_size"], r["nodes"]): r
                for r in sglang_pkg.tables["ep_all2all"]
            }
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
        rows = parse_nccl_tests_output(
            text, operation="all_reduce", group_size=8, nodes=1, source_id="nccl:unit"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dtype"], "fp32")
        self.assertEqual(rows[0]["latency_us"], 72.75)
        self.assertEqual(rows[1]["message_bytes"], 33554432)
        with self.assertRaises(ConfigurationError):
            parse_nccl_tests_output(
                "# nothing here\n",
                operation="all_reduce",
                group_size=8,
                nodes=1,
                source_id="x",
            )
