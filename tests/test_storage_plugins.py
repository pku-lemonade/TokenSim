"""External media loading, shared ownership, statistics and SSD compatibility."""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.service import get_mooncake_service
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.storage import BackendSpec, IORequest, StorageBackendFactory

MODULE = "tokensim_test_external_media"
PLUGIN = """from TokenSim.storage import IOCost
class LabBackend:
    def __init__(self, config):
        self.capacity_bytes = config["capacity_bytes"]
        self.rate = config.get("rate", 1024)
    def estimate_io(self, request):
        return IOCost(0.01 + request.nbytes / self.rate)
    def describe(self):
        return {"capacity_bytes": self.capacity_bytes, "rate": self.rate}
"""


class StoragePluginTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        (Path(directory.name) / f"{MODULE}.py").write_text(PLUGIN)
        for context in (
            patch.object(sys, "path", [directory.name, *sys.path]),
            patch.dict(StorageBackendFactory._registry, {}, clear=True),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(sys.modules.pop, MODULE, None)

    def options(self, *, master="default", **config):
        return {
            "master_server_address": master,
            "memory_capacity_blocks": 0,
            "offload": {
                "backend": f"{MODULE}:LabBackend",
                "name": "lab-cache",
                "config": {"capacity_bytes": 64, **config},
            },
        }

    def test_external_backend_shares_pool_and_preserves_cache_ownership(self):
        owner = SimpleNamespace()
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config=self.options(),
        )
        service = get_mooncake_service(config, block_bytes=64, owner=owner)
        same = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config=self.options(rate=1024),
        )
        self.assertIs(get_mooncake_service(same, block_bytes=64, owner=owner), service)
        different = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config=self.options(rate=2048),
        )
        with self.assertRaisesRegex(ConfigurationError, "compatible store"):
            get_mooncake_service(different, block_bytes=64, owner=owner)
        store = service.store
        a, b = [PoolKey(KeyMetadata("test"), key) for key in "ab"]
        latency = store.put([a])
        self.assertAlmostEqual(latency, 0.01 + 64 / 1024)
        self.assertEqual(store.lookup([a], now=latency / 2).hit_blocks, 0)
        self.assertEqual(store.lookup([a], now=latency).tier, "lab-cache")
        store.get([a], "lab-cache", now=latency)
        store.put([b], now=latency)
        self.assertNotIn(b.to_string(), store.objects)
        store.put([b], now=2 * latency)
        self.assertNotIn(a.to_string(), store.objects)
        # A malformed estimate must fail before evicting resident data.
        with patch.object(type(store.offload.backend), "estimate_io", return_value=0.1):
            with self.assertRaisesRegex(SimulationStateError, "IOCost"):
                store.put([a], now=1)
        self.assertTrue(store.offload.contains(b.to_string()))
        metrics = store.stats.as_dict()["storage_metrics"]["default"]["lab-cache"]
        self.assertEqual((metrics["read_bytes"], metrics["write_bytes"]), (64, 128))
        self.assertEqual(store.stats.ssd_write_bytes, 0)

    def test_registration_and_invalid_selections(self):
        builder = importlib.import_module(MODULE).LabBackend
        StorageBackendFactory.register_backend("lab", builder)
        backend = StorageBackendFactory.create(
            BackendSpec("lab", {"capacity_bytes": 64})
        )
        self.assertAlmostEqual(
            backend.estimate_io(IORequest("write", 1024)).service_seconds, 1.01
        )
        with self.assertRaisesRegex(ConfigurationError, "already registered"):
            StorageBackendFactory.register_backend("lab", builder)
        with self.assertRaisesRegex(ConfigurationError, "cannot load"):
            StorageBackendFactory.create(BackendSpec("missing_storage_package:Backend"))
        for extra in ({"ssd_capacity_blocks": 1}, {"enable_offload": False}):
            with self.subTest(extra=extra), self.assertRaises(ConfigurationError):
                parse_mooncake_config({**self.options(), **extra})

    def test_unconfigured_plugins_are_not_imported(self):
        program = """import sys
class RejectOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("tokensim_hbf", "tokensim_test_external_media")):
            raise AssertionError("unconfigured backend was imported")
sys.meta_path.insert(0, RejectOptionalImports())
from TokenSim.mooncake.config import MooncakeConfig
from TokenSim.mooncake.store import MooncakeStore
assert MooncakeStore(MooncakeConfig(), 64).offload is None
assert MooncakeStore(MooncakeConfig(enable_offload=True, ssd_capacity_blocks=1), 64).offload.tier == "ssd"
"""
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ssd_adapter_matches_generic_backend(self):
        options = {
            "capacity_bytes": 128,
            "read_bandwidth_bytes_per_s": 1000,
            "write_bandwidth_bytes_per_s": 500,
            "read_latency_us": 10,
            "write_latency_us": 20,
        }
        generic = MooncakeStore(
            MooncakeConfig(
                memory_capacity_blocks=0, offload={"backend": "ssd", "config": options}
            ),
            64,
        )
        legacy = MooncakeStore(
            MooncakeConfig(
                memory_capacity_blocks=0,
                enable_offload=True,
                **{f"ssd_{name}": value for name, value in options.items()},
            ),
            64,
        )
        key = PoolKey(KeyMetadata("test"), "key")
        self.assertEqual(generic.put([key]), legacy.put([key]))
        self.assertEqual(
            generic.get([key], "offload", now=1), legacy.get([key], "ssd", now=1)
        )
        metrics = generic.stats.as_dict()["storage_metrics"]["default"]["offload"]
        self.assertEqual(metrics["write_bytes"], legacy.stats.ssd_write_bytes)
        self.assertEqual(generic.stats.ssd_write_bytes, 0)

    def test_metrics_aggregate_without_aliasing_and_reset_profiles(self):
        a = MooncakeStore(parse_mooncake_config(self.options(master="a")), 64)
        b = MooncakeStore(
            parse_mooncake_config(self.options(master="b", capacity_bytes=128)), 64
        )
        key = PoolKey(KeyMetadata("test"), "key")
        a.put([key])
        b.put([key])
        total = a.stats.aggregate(b.stats)
        self.assertEqual(set(total.as_dict()["storage_metrics"]), {"a", "b"})
        doubled = a.stats.aggregate(a.stats).as_dict()["storage_metrics"]["a"][
            "lab-cache"
        ]
        self.assertEqual(doubled["write_bytes"], 128)
        self.assertEqual(
            a.stats.as_dict()["storage_metrics"]["a"]["lab-cache"]["write_bytes"], 64
        )
        cleared = total.storage_metrics.reset().as_dict()
        self.assertEqual(cleared["b"]["lab-cache"]["profile"]["capacity_bytes"], 128)
        self.assertEqual(cleared["a"]["lab-cache"]["write_bytes"], 0)
