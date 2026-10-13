"""HBF examples: device isolation, media estimates and runtime reuse."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.cli import build_parser, main
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.kv_transfer.connectors.factory import KVConnectorFactory
from TokenSim.mooncake.config import parse_mooncake_config
from TokenSim.operator_data.analytical.gpu import GPURooflineModel
from TokenSim.reporting.stats import get_mooncake_stats
from TokenSim.storage import BackendSpec, IORequest, StorageBackendFactory
from tests.support.hardware import make_model

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "data/kv_transfer/mooncake_store_hbf.json"


class HBFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = patch.object(
            sys, "path", [str(ROOT / "examples/storage/hbf"), *sys.path]
        )
        path.start()
        cls.addClassCleanup(path.stop)

    def test_nested_devices_keep_gpu_and_storage_choices_independent(self):
        hardware = HardwareContext.load(ROOT / "data")
        for path in (ROOT / "data/devices/hbf").glob("*.yaml"):
            variant = hardware.device(path.stem)
            base = hardware.device(path.stem.split("_hbf", 1)[0])
            self.assertEqual(variant.peak_compute, base.peak_compute)
            self.assertEqual(variant.scale_up_link, base.scale_up_link)
            self.assertEqual(variant.memory_capacity_bytes.grade, "D")
            self.assertIsNone(hardware.operator_package(variant.device_id))
        base, variant = hardware.device("H100"), hardware.device("h100_sxm_hbf1")
        config = CacheConfig(16, base, make_model())
        KVTransferConfig.from_file(PRESET)
        self.assertIs(hardware.device("H100"), base)
        self.assertEqual(
            CacheConfig(16, base, make_model()).num_gpu_blocks, config.num_gpu_blocks
        )
        self.assertLess(
            CacheConfig(16, variant, make_model()).num_gpu_blocks, config.num_gpu_blocks
        )
        estimator = GPURooflineModel()
        work = {"m": 1, "n": 4096, "k": 4096, "dtype": "fp16"}
        normal, changed = [
            estimator.estimate("gemm", work, device) for device in (base, variant)
        ]
        self.assertEqual(normal.compute_time_us, changed.compute_time_us)
        self.assertGreater(changed.memory_time_us, normal.memory_time_us)

    def test_media_curve_and_backend_owned_validation(self):
        spec = parse_mooncake_config(
            KVTransferConfig.from_file(PRESET).kv_connector_extra_config
        ).offload
        backend = StorageBackendFactory.create(spec)
        for operation, latency in (("read", 12e-6), ("write", 120e-6)):
            for size, bandwidth in (
                (4096, 0.8 * 128 * (1 << 30)),
                (10 << 20, 128 * (1 << 30)),
            ):
                self.assertAlmostEqual(
                    backend.estimate_io(IORequest(operation, size)).service_seconds,
                    latency + size / bandwidth,
                )
            costs = [
                backend.estimate_io(
                    IORequest(operation, 1 << 20, pattern)
                ).service_seconds
                for pattern in ("sequential", "auto", "random")
            ]
            self.assertLess(costs[0], costs[1])
            self.assertLess(costs[1], costs[2])
        for options in (
            {"capacity_bytes": 0},
            {"seq_read_bandwidth_bytes_per_s": float("nan")},
            {"write_latency_us": -1},
            {"unknown_option": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ConfigurationError):
                StorageBackendFactory.create(
                    BackendSpec(spec.backend, {**spec.config, **options})
                )
        with self.assertRaisesRegex(ConfigurationError, "single-replica DRAM"):
            KVConnectorFactory.create_connector(
                KVTransferConfig(
                    kv_connector="MooncakeStoreConnector",
                    kv_connector_extra_config={
                        "offload": spec,
                        "cache_layout": "grouped",
                    },
                ),
                SimpleNamespace(hybrid_layout=object()),
            )

    def test_runtime_reuses_flash_after_gpu_eviction_and_resets_metrics(self):
        cluster = json.loads((ROOT / "data/clusters/1_a100/h1.json").read_text())
        cluster["worker_groups"][0]["hardware"] = "h100_sxm"
        cluster["kv_cache_capacity_tokens_total"] = 80
        kv = json.loads(PRESET.read_text())
        kv["kv_connector_extra_config"]["memory_capacity_blocks"] = 0
        kv["kv_connector_extra_config"]["offload"]["config"]["capacity_bytes"] = (
            64 << 20
        )
        requests = [
            [64, 2, {"hash_ids": hashes}]
            for hashes in ([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4])
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, value in (
                ("cluster", cluster),
                ("kv", kv),
                ("requests", requests),
            ):
                (path / f"{name}.json").write_text(json.dumps(value))
            options = {
                "--cluster": str(path / "cluster.json"),
                "--kv_transfer_config": str(path / "kv.json"),
                "--dataset_path": str(path / "requests.json"),
                "--workload_type": "json_pairs",
                "--request_count": "3",
                "--distribution": "uniform",
                "--qps": "1",
                "--batching": "paged-attn",
                "--latency_backend": "analytical",
                "--verbose": "none",
                "--results_path": str(path / "results"),
            }
            args = build_parser().parse_args(
                [item for pair in options.items() for item in pair]
            )
            with redirect_stdout(io.StringIO()):
                engine = main(args)
            result = json.loads(
                next((path / "results").glob("result_*.json")).read_text()
            )
        self.assertEqual(engine.workers[0].device.device_id, "h100_sxm")
        self.assertEqual(result["notdone"], [])
        metrics = result["storage_metrics"]["default"]["hbf"]
        self.assertGreater(metrics["read_bytes"], 0)
        self.assertGreater(metrics["write_bytes"], 0)
        self.assertEqual(result["mooncake_ssd_read_bytes"], 0)
        engine.reset_profile_stats()
        cleared = get_mooncake_stats(engine)["storage_metrics"]["default"]["hbf"]
        self.assertEqual(cleared["read_bytes"], 0)
        self.assertEqual(cleared["write_bytes"], 0)
        self.assertEqual(cleared["profile"], metrics["profile"])
