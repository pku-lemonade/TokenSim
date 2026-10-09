"""Module source registration, boundary consistency and connector capabilities."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.kv_transfer.connectors.factory import KVConnectorFactory
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from TokenSim.operator_data.importers.aiconfigurator import import_aiconfigurator_system

ROOT = Path(__file__).resolve().parents[1]


class ComponentContractTest(unittest.TestCase):
    def test_complete_module_import_registers_its_sources_for_any_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "custom-system"
            target = root / "mhc/vllm/1.0"
            target.mkdir(parents=True)
            rows = [
                {
                    "architecture": "CustomTransformer",
                    "hidden_size": 128,
                    "hc_mult": 2,
                    "num_sites": 1,
                    "sinkhorn_iters": 10,
                    "op_name": op,
                    "latency": 0.01,
                    "num_tokens": 8,
                    "kernel_source": f"custom_{op}",
                    "framework": "vllm",
                    "version": "1.0",
                }
                for op in ("pre", "post")
            ]
            pq.write_table(
                pa.Table.from_pylist(rows), target / "mhc_module_perf.parquet"
            )
            package = import_aiconfigurator_system(
                root, device_id="custom-device", backend="vllm"
            )
            self.assertEqual(len(package.tables["mhc"]), 2)
            for row in package.tables["mhc"]:
                self.assertIn(row["source_id"], package.meta.sources)
                self.assertEqual(row["architecture"], "customtransformer")
                self.assertEqual(row["hidden_size"], 128)
                self.assertEqual(row["latency_us"], 10)

    def test_small_batch_clamp_is_consistent_off_grid_and_cache_reports_it(self):
        hardware = HardwareContext.load(ROOT / "data")
        model = hardware.model("DeepSeek-V4-Pro")
        rows = []
        for kind, ratio in (("csa", 4), ("hca", 128)):
            for context in (8192, 16384):
                rows.append(
                    {
                        **model.operator_modules["attention"]["key"],
                        "layer_kind": kind,
                        "phase": "generation",
                        "num_heads": 128,
                        "tp_size": 1,
                        "batch_size": 64,
                        "input_seq_len": 1,
                        "context_len": context,
                        "compress_ratio": ratio,
                        "latency_us": context / 100,
                        "source_id": "fixture",
                    }
                )
        package = OperatorDataPackage.from_rows(
            PackageMeta(
                "fixture",
                "b300_sxm",
                "vllm",
                sources={"fixture": SourceRecord("fixture", "B", "measured")},
            ),
            {"dsv4_attention": rows},
        )
        backend = OperatorTableLatencyBackend(
            device=hardware.device("b300"),
            model=model,
            package=package,
            attention_backend="module",
        )
        estimator = backend.native_attention
        boundary = estimator._query("csa", "generation", 1, 1, 8192)
        neighbor = estimator._query("csa", "generation", 1, 1, 8320)
        self.assertLess(abs(neighbor.latency_us / boundary.latency_us - 1), 0.05)
        self.assertEqual(neighbor.source_batch_size, 64)
        clamped = estimator._query("csa", "generation", 1, 1, 8192)
        exact = estimator._query("csa", "generation", 64, 1, 8192)
        self.assertEqual(clamped.match, "clamped")
        self.assertEqual(exact.match, "exact")

    def test_grouped_connector_is_explicit_and_unknown_options_are_rejected(self):
        cache = SimpleNamespace(hybrid_layout=object())
        config = KVTransferConfig(kv_connector="MooncakeStoreConnector")
        with self.assertRaisesRegex(ConfigurationError, "grouped KV"):
            KVConnectorFactory.create_connector(config, cache)
        config = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_connector_extra_config={"cpu_btyes_to_use": 123},
        )
        with self.assertRaisesRegex(ConfigurationError, "unknown SimpleCPU"):
            KVConnectorFactory.create_connector(config, cache)
