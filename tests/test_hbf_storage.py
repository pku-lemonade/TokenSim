"""HBF media retain the main runtime's readiness and ownership rules."""

from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest

from TokenSim.cli import build_parser, main
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer.connectors.factory import KVConnectorFactory
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.ssd import OffloadTier
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.reporting.stats import get_mooncake_stats
from TokenSim.storage import IORequest
from tests.support.storage import example_storage_package

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30


def hbf_config(capacity_bytes=4 * 4096, **options):
    return MooncakeConfig(
        memory_capacity_blocks=0,
        offload={
            "backend": "tokensim_hbf:HBFBackend",
            "name": "hbf",
            "config": {"capacity_bytes": capacity_bytes, **options},
        },
    )


def hbf_metrics(stats):
    return stats.storage_metrics.as_dict()["default"]["hbf"]


class HBFStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        context = example_storage_package()
        context.__enter__()
        cls.addClassCleanup(context.__exit__, None, None, None)

    def test_runtime_reads_hbf_after_gpu_eviction_and_exports_media_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "cluster.json").write_text(
                json.dumps(
                    {
                        "num_workers": 1,
                        "networks": {"node": "ethernet100Gb"},
                        "worker_groups": [
                            {
                                "role": "hybrid",
                                "hardware": "h100_sxm",
                                "num_workers": 1,
                                "network": "node",
                            }
                        ],
                        "kv_cache_capacity_tokens_total": 80,
                    }
                )
            )
            (path / "kv.json").write_text(
                json.dumps(
                    {
                        "kv_connector": "MooncakeStoreConnector",
                        "kv_connector_extra_config": {
                            "memory_capacity_blocks": 0,
                            "offload": {
                                "backend": "tokensim_hbf:HBFBackend",
                                "name": "hbf",
                                "config": {"capacity_bytes": 64 << 20},
                            },
                        },
                    }
                )
            )
            (path / "requests.json").write_text(
                json.dumps(
                    [
                        [64, 2, {"hash_ids": [1, 2, 3, 4]}],
                        [64, 2, {"hash_ids": [5, 6, 7, 8]}],
                        [64, 2, {"hash_ids": [1, 2, 3, 4]}],
                    ]
                )
            )
            args = build_parser().parse_args(
                [
                    "--cluster",
                    str(path / "cluster.json"),
                    "--kv_transfer_config",
                    str(path / "kv.json"),
                    "--dataset_path",
                    str(path / "requests.json"),
                    "--workload_type",
                    "json_pairs",
                    "--request_count",
                    "3",
                    "--distribution",
                    "uniform",
                    "--qps",
                    "1",
                    "--batching",
                    "paged-attn",
                    "--latency_backend",
                    "analytical",
                    "--verbose",
                    "none",
                    "--results_path",
                    str(path / "results"),
                ]
            )
            with redirect_stdout(io.StringIO()):
                engine = main(args)
            result = json.loads(
                next((path / "results").glob("result_*.json")).read_text()
            )
        self.assertEqual(engine.workers[0].device.device_id, "h100_sxm")
        self.assertEqual(result["notdone"], [])
        self.assertGreater(
            result["storage_metrics"]["default"]["hbf"]["read_blocks"], 0
        )
        self.assertGreater(
            result["storage_metrics"]["default"]["hbf"]["write_blocks"], 0
        )
        self.assertEqual(result["mooncake_ssd_read_blocks"], 0)
        self.assertEqual(result["mooncake_offload_tiers"], ["hbf"])
        engine.reset_profile_stats()
        reset = get_mooncake_stats(engine)["storage_metrics"]["default"]["hbf"]
        self.assertEqual(reset["read_blocks"], 0)
        self.assertEqual(reset["write_blocks"], 0)
        self.assertEqual(reset["profile"]["capacity_bytes"], 64 << 20)

    def test_packaged_example_supports_prefix_reuse(self):
        config = KVTransferConfig.from_file(
            ROOT / "data/kv_transfer/mooncake_store_hbf.json"
        )
        cache = SimpleNamespace(
            block_size=16, size_per_token=1, model="test", rank_info=None
        )
        connector = KVConnectorFactory.create_connector(config, cache)
        request = Request(0, 64, 2, 16, hash_ids=[1, 2, 3, 4])
        keys = connector._keys_for_request(request)
        store = connector.service.store
        delay = store.put(keys)
        self.assertGreater(delay, 0)
        self.assertEqual(store.lookup(keys, now=0).hit_blocks, 0)
        hit = store.lookup(keys, now=delay)
        self.assertEqual(hit.hit_blocks, 4)
        self.assertEqual(hit.tier, "hbf")
        self.assertEqual(hbf_metrics(store.stats)["write_bytes"], 64)

    def test_bandwidth_curve_has_random_and_sequential_limits(self):
        tier = OffloadTier(hbf_config(seq_read_bandwidth_bytes_per_s=100 * GIB), 4096)
        backend = tier.backend
        for size, bandwidth in ((4096, 80 * GIB), (10 << 20, 100 * GIB)):
            latency = backend.estimate_io(IORequest("read", size)).service_seconds
            self.assertAlmostEqual(latency, 12e-6 + size / bandwidth)
        middle = backend.estimate_io(IORequest("read", 1 << 20)).service_seconds
        self.assertLess(12e-6 + (1 << 20) / (100 * GIB), middle)
        self.assertLess(middle, 12e-6 + (1 << 20) / (80 * GIB))
        self.assertEqual(tier.read("missing").latency, 0)
        expected = 120e-6 + 4096 / (0.8 * 128 * GIB)
        self.assertAlmostEqual(tier.write("key", 1).latency, expected)

    def test_invalid_options_are_validated_by_the_plugin(self):
        for values in (
            {"capacity_bytes": -1},
            {"capacity_bytes": 1.5},
            {"seq_read_bandwidth_bytes_per_s": 0},
            {"random_4k_write_bandwidth_bytes_per_s": float("nan")},
            {"read_latency_us": -1},
            {"unknown_option": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                hbf_config(**values)
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config(
                {
                    "offload": {"backend": "tokensim_hbf:HBFBackend", "config": {}},
                    "ssd_capacity_blocks": 10,
                }
            )

    def test_descriptions_normalize_explicit_defaults_and_access_patterns(self):
        default = OffloadTier(hbf_config(), 4096).backend
        explicit = OffloadTier(
            hbf_config(read_latency_us=12, write_latency_us=120), 4096
        ).backend
        self.assertEqual(
            json.dumps(default.describe(), sort_keys=True),
            json.dumps(explicit.describe(), sort_keys=True),
        )
        size = 1 << 20
        random = default.estimate_io(IORequest("read", size, "random")).service_seconds
        sequential = default.estimate_io(
            IORequest("read", size, "sequential")
        ).service_seconds
        automatic = default.estimate_io(IORequest("read", size)).service_seconds
        self.assertLess(sequential, automatic)
        self.assertLess(automatic, random)

    def test_writes_become_visible_on_completion_and_reads_pin_objects(self):
        store = MooncakeStore(hbf_config(capacity_bytes=4096), 4096)
        a, b = [PoolKey(KeyMetadata("test"), key) for key in "ab"]
        timing = store.put([a], now=0, initial_delay=1)
        self.assertGreater(timing, 0)
        self.assertEqual(store.lookup([a], now=1).hit_blocks, 0)
        with self.assertRaises(SimulationStateError):
            store.get([a], "hbf", now=1)
        ready = 1 + timing
        self.assertEqual(store.lookup([a], now=ready).tier, "hbf")
        read_latency = store.get([a], "hbf", now=ready, initial_delay=2)
        store.put([b], now=ready + 1)
        self.assertIn(a.to_string(), store.objects)
        self.assertNotIn(b.to_string(), store.objects)
        store.put([b], now=ready + 2 + read_latency)
        self.assertNotIn(a.to_string(), store.objects)
        self.assertIn(b.to_string(), store.objects)
        self.assertEqual(hbf_metrics(store.stats)["read_bytes"], 4096)
        self.assertEqual(store.stats.ssd_read_bytes, 0)
        self.assertEqual(hbf_metrics(store.stats)["write_blocks"], 2)

    def test_eviction_removes_stale_metadata_and_overwrite_preserves_capacity(self):
        config = hbf_config(capacity_bytes=2 * 4096)
        tier = OffloadTier(config, 4096)
        tier.write("a", 1)
        tier.write("b", 1)
        self.assertEqual(tier.write("a", 1).evicted_keys, [])
        self.assertTrue(tier.contains("b"))
        store = MooncakeStore(config, 4096)
        for i in range(10):
            key = PoolKey(KeyMetadata("test"), str(i))
            store.put([key], now=i)
            self.assertEqual(len(store.objects), store.offload.used_blocks)
            self.assertLessEqual(len(store.objects), 2)
        hit = store.lookup([key], now=10)
        self.assertEqual(hit.tier, "hbf")
        profile = store.stats.as_dict()["mooncake_offload_profiles"][0]
        self.assertTrue(profile["write_blocking"])
        self.assertEqual(profile["profile"]["read_latency_us"], 12)

    def test_connector_uses_shared_prefix_policy_and_counts_hbf_hits(self):
        cache = SimpleNamespace(
            block_size=16, size_per_token=1, model="test", rank_info=None
        )
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "offload": {
                    "backend": "tokensim_hbf:HBFBackend",
                    "name": "hbf",
                    "config": {"capacity_bytes": 128},
                },
                "memory_capacity_blocks": 0,
            },
        )
        connector = KVConnectorFactory.create_connector(config, cache)
        original = Request(0, 64, 2, 16, hash_ids=[1, 2, 3, 4])
        connector.service.store.put(connector._keys_for_request(original), now=-1)
        request = Request(1, 64, 2, 16, hash_ids=[1, 2, 3, 4])
        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), 48)
        connector.update_state_after_alloc(request, [], 48)
        request.num_computed_tokens = 48
        request.scheduled_tokens = 16
        metadata = connector.build_connector_meta(
            SimpleNamespace(running=[request], preempted=[])
        )
        self.assertEqual(metadata.loads[0].tier, "hbf")
        connector.bind_connector_metadata(metadata)
        self.assertGreater(connector.start_load_kv(), 0)
        self.assertEqual(connector.mooncake_stats.mooncake_disk_hit_tokens, 48)
        self.assertEqual(hbf_metrics(connector.mooncake_stats)["read_blocks"], 3)

    def test_grouped_cache_offload_is_rejected_by_existing_runtime_boundary(self):
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "cache_layout": "grouped",
                "offload": {
                    "backend": "tokensim_hbf:HBFBackend",
                    "name": "hbf",
                    "config": {"capacity_bytes": 128},
                },
            },
        )
        with self.assertRaisesRegex(ConfigurationError, "single-replica DRAM"):
            KVConnectorFactory.create_connector(
                config, SimpleNamespace(hybrid_layout=object())
            )
