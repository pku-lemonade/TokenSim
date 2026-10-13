"""Third-party media reuse storage ownership without edits to the core."""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.service import get_mooncake_service
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.storage import BackendSpec, IOCost, IORequest, StorageBackendFactory

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
        paths = patch.object(sys, "path", [directory.name, *sys.path])
        paths.start()
        self.addCleanup(paths.stop)
        self.addCleanup(sys.modules.pop, MODULE, None)
        registry = patch.dict(StorageBackendFactory._registry, {}, clear=True)
        registry.start()
        self.addCleanup(registry.stop)

    def options(self, *, name="lab-cache", master="default", **config):
        return {
            "master_server_address": master,
            "memory_capacity_blocks": 0,
            "offload": {
                "backend": f"{MODULE}:LabBackend",
                "name": name,
                "config": {"capacity_bytes": 256, **config},
            },
        }

    def test_external_module_uses_common_readiness_pinning_and_generic_metrics(self):
        store = MooncakeStore(
            parse_mooncake_config(self.options(capacity_bytes=64)), 64
        )
        a, b = [PoolKey(KeyMetadata("test"), key) for key in "ab"]
        latency = store.put([a], now=0)
        self.assertEqual(latency, 0.01 + 64 / 1024)
        self.assertEqual(store.lookup([a], now=latency / 2).hit_blocks, 0)
        self.assertEqual(store.lookup([a], now=latency).tier, "lab-cache")
        store.get([a], "lab-cache", now=latency)
        store.put([b], now=latency)
        self.assertIn(a.to_string(), store.objects)
        self.assertNotIn(b.to_string(), store.objects)
        store.put([b], now=2 * latency)
        self.assertNotIn(a.to_string(), store.objects)
        metrics = store.stats.as_dict()["storage_metrics"]["default"]["lab-cache"]
        self.assertEqual(metrics["backend"], f"{MODULE}:LabBackend")
        self.assertEqual(metrics["read_bytes"], 64)
        self.assertEqual(metrics["write_bytes"], 128)
        self.assertEqual(metrics["hit_count"], 1)
        self.assertEqual(store.stats.ssd_write_bytes, 0)
        self.assertNotIn(MODULE, StorageBackendFactory._registry)

    def test_registered_builder_can_be_selected_without_a_module_reference(self):
        backend_class = importlib.import_module(MODULE).LabBackend
        StorageBackendFactory.register_backend("lab", backend_class)
        backend = StorageBackendFactory.create(
            BackendSpec("lab", {"capacity_bytes": 10})
        )
        self.assertEqual(
            backend.estimate_io(IORequest("write", 1024)).service_seconds, 1.01
        )
        with self.assertRaisesRegex(ConfigurationError, "already registered"):
            StorageBackendFactory.register_backend("lab", backend_class)

    def test_builtin_ssd_plugin_matches_the_legacy_adapter(self):
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
        self.assertEqual(
            generic.stats.as_dict()["storage_metrics"]["default"]["offload"][
                "write_bytes"
            ],
            legacy.stats.ssd_write_bytes,
        )
        self.assertEqual(generic.stats.ssd_write_bytes, 0)

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

    def test_bad_references_and_backend_contracts_report_configuration_errors(self):
        for reference in (
            "unknown",
            "missing_storage_package:Backend",
            f"{MODULE}:Missing",
            f"{MODULE}:__name__",
            f"{MODULE}:LabBackend",
        ):
            with (
                self.subTest(reference=reference),
                self.assertRaises(ConfigurationError),
            ):
                StorageBackendFactory.create(BackendSpec(reference, {}))
        for builder in (
            lambda config: object(),
            lambda config: SimpleNamespace(capacity_bytes=1.5),
            lambda config: SimpleNamespace(
                capacity_bytes=10, estimate_io=None, describe=lambda: {}
            ),
            lambda config: SimpleNamespace(
                capacity_bytes=10,
                estimate_io=lambda io: IOCost(0),
                describe=lambda: {"value": float("nan")},
            ),
        ):
            with patch.dict(StorageBackendFactory._registry, {"bad": builder}):
                with self.assertRaises(ConfigurationError):
                    StorageBackendFactory.create(BackendSpec("bad"))

    def test_selection_rejects_ambiguous_or_invalid_options(self):
        for extra in (
            {"ssd_capacity_blocks": 1},
            {"ssd_read_latency_us": 100},
            {"enable_offload": False},
            {"enable_offload": "false"},
            {"offload_tier": "unknown"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ConfigurationError):
                parse_mooncake_config({**self.options(), **extra})
        with self.assertRaises(ConfigurationError):
            MooncakeConfig(offload=self.options()["offload"], ssd_capacity_blocks=1)
        for reference in (
            {},
            [],
            {"backend": "lab", "config": []},
            {"backend": "lab", "name": "memory"},
        ):
            with (
                self.subTest(reference=reference),
                self.assertRaises(ConfigurationError),
            ):
                BackendSpec.from_mapping(reference)

    def test_invalid_estimates_cannot_evict_resident_data(self):
        backend_class = importlib.import_module(MODULE).LabBackend
        store = MooncakeStore(
            parse_mooncake_config(self.options(capacity_bytes=64)), 64
        )
        a, b = [PoolKey(KeyMetadata("test"), key) for key in "ab"]
        store.put([a], now=0)
        with patch.object(backend_class, "estimate_io", return_value=0.1):
            with self.assertRaisesRegex(SimulationStateError, "IOCost"):
                store.put([b], now=1)
        self.assertTrue(store.offload.contains(a.to_string()))
        self.assertFalse(store.offload.contains(b.to_string()))

    def test_shared_pools_compare_normalized_plugin_profiles(self):
        owner = SimpleNamespace()
        first = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config=self.options(),
        )
        same = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config=self.options(rate=1024),
        )
        different = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config=self.options(rate=2048),
        )
        service = get_mooncake_service(first, block_bytes=64, owner=owner)
        self.assertIs(get_mooncake_service(same, block_bytes=64, owner=owner), service)
        with self.assertRaisesRegex(ConfigurationError, "compatible store"):
            get_mooncake_service(different, block_bytes=64, owner=owner)

    def test_aggregation_and_reset_preserve_instance_identity_without_aliasing(self):
        a = MooncakeStore(parse_mooncake_config(self.options(master="a")), 64)
        b = MooncakeStore(
            parse_mooncake_config(self.options(master="b", capacity_bytes=128)), 64
        )
        key = PoolKey(KeyMetadata("test"), "key")
        a.put([key])
        b.put([key])
        total = a.stats.aggregate(b.stats)
        self.assertEqual(set(total.as_dict()["storage_metrics"]), {"a", "b"})
        combined = a.stats.aggregate(a.stats).as_dict()["storage_metrics"]["a"][
            "lab-cache"
        ]
        self.assertEqual(combined["write_bytes"], 128)
        self.assertEqual(
            a.stats.as_dict()["storage_metrics"]["a"]["lab-cache"]["write_bytes"], 64
        )
        cleared = total.storage_metrics.reset().as_dict()
        self.assertEqual(cleared["b"]["lab-cache"]["profile"]["capacity_bytes"], 128)
        self.assertEqual(cleared["a"]["lab-cache"]["write_bytes"], 0)

    def test_io_contract_rejects_invalid_durations_and_requests(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                IOCost(value)
        for args in (
            ("erase", 1),
            ("read", -1),
            ("write", 1.5),
            ("read", 1, "unknown"),
        ):
            with self.subTest(args=args), self.assertRaises(ConfigurationError):
                IORequest(*args)

    def test_core_has_no_example_media_names(self):
        core = Path(__file__).resolve().parents[1] / "TokenSim"
        self.assertEqual(
            [
                str(path)
                for path in core.rglob("*.py")
                if "hbf" in path.read_text().lower()
            ],
            [],
        )
