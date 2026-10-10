import unittest
import simpy
from types import SimpleNamespace
from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.transfer_engine import TransferEngineSimulator, Segment
from TokenSim.mooncake.ssd import OffloadTier
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.kv_transfer.connectors.factory import KVConnectorFactory
from TokenSim.llm.llm_request import Request
from TokenSim.kv_transfer.connectors.store_pages import _SharedService
from TokenSim.kv_transfer.native_storage import NativePageStore, QueuedKVLink
from tests.support.hardware import make_hardware


class ExternalStorageTest(unittest.TestCase):
    def test_composed_stores_choose_one_load_and_route_each_transfer_once(self):
        cache = SimpleNamespace(
            block_size=16, size_per_token=1, model="test", rank_info=None
        )
        for nested in (False, True):
            children = [
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_connector_extra_config": {"load_async": True},
                }
                for _ in range(2)
            ]
            raw = {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_rank": 7,
                "kv_connector_extra_config": {"connectors": children},
            }
            if nested:
                raw = {**raw, "kv_connector_extra_config": {"connectors": [raw]}}
            connector = KVConnectorFactory.create_connector(
                KVTransferConfig.from_mapping(raw), cache
            )
            stores = connector.children[0].children if nested else connector.children
            original = Request(0, 64, 2, 16, hash_ids=[1, 2, 3, 4])
            for count, child in zip((2, 3), stores):
                self.assertEqual(child.config.kv_role, "kv_both")
                self.assertEqual(child.config.kv_rank, 7)
                child.service.store.put(child._keys_for_request(original)[:count])
            request = Request(1, 64, 2, 16, hash_ids=[1, 2, 3, 4])
            tokens = connector.get_num_new_matched_tokens(request, 0)
            self.assertEqual(tokens, 48)
            connector.update_state_after_alloc(request, [], tokens)
            self.assertEqual(request.cached_prefill_tokens, 48)
            request.num_computed_tokens = 48
            request.scheduled_tokens = 16
            metadata = connector.build_connector_meta(
                SimpleNamespace(running=[request], preempted=[])
            )
            self.assertEqual(len(metadata.loads), 1)
            self.assertEqual(len(metadata.saves), 1)
            connector.bind_connector_metadata(metadata)
            self.assertEqual(connector.start_load_kv(), metadata.loads[0].latency)
            self.assertEqual(connector.wait_for_save(), metadata.saves[0].latency)
            self.assertEqual([child.stats.transfer_count for child in stores], [1, 1])
            released = []
            connector.set_block_releaser(
                lambda req, ids: released.append((req.id, ids))
            )
            self.assertTrue(connector.request_finished(request, [0, 1])[0])
            connector.update_connector_output(connector.build_connector_worker_meta())
            self.assertEqual(released, [(request.id, [0, 1])])

    def test_store_namespaces_cannot_collide_through_string_separators(self):
        left = KeyMetadata("test", tenant_id="a@b", cache_prefix="c")
        right = KeyMetadata("test", tenant_id="a", cache_prefix="b@c")
        self.assertNotEqual(
            PoolKey(left, "hash").to_string(), PoolKey(right, "hash").to_string()
        )

    def test_uniform_store_readiness_pins_and_holes_survive_shared_access(self):
        store = MooncakeStore(MooncakeConfig(memory_capacity_blocks=2), 100)
        a, b, c = [PoolKey(KeyMetadata("test"), key) for key in "abc"]
        store.put([a], now=0, initial_delay=1)
        self.assertEqual(store.lookup([a], now=0.9).hit_blocks, 0)
        self.assertEqual(store.lookup([a], now=1).hit_blocks, 1)
        store.put([c], now=1)
        # A suffix may have been saved by another client without its prefix.
        self.assertEqual(store.missing_indices([a, b, c]), [1])
        store.get([a, c], "memory", now=1, initial_delay=2)
        store.put([b], now=2)
        self.assertEqual(store.lookup([a], now=2).hit_blocks, 1)
        self.assertNotIn(b.to_string(), store.objects)
        store.put([b], now=3)
        self.assertIn(b.to_string(), store.objects)
        self.assertLessEqual(store.memory_used_bytes, store.memory_capacity_bytes)

    def test_ssd_overwrite_and_eviction_preserve_capacity_and_live_metadata(self):
        config = MooncakeConfig(
            memory_capacity_blocks=0, enable_offload=True, ssd_capacity_blocks=2
        )
        disk = OffloadTier(config, 100)
        disk.write("a", 1)
        disk.write("b", 1)
        self.assertEqual(disk.write("a", 1).evicted_keys, [])
        self.assertTrue(disk.contains("b"))
        disk.pin("a", 10)
        self.assertEqual(disk.write("c", 1, now=1).evicted_keys, ["b"])
        disk.pin("c", 10)
        self.assertEqual(disk.write("d", 1, now=1).blocks, 0)
        self.assertEqual(disk.used_blocks, 2)
        store = MooncakeStore(config, 100)
        for i in range(20):
            key = PoolKey(KeyMetadata("test"), str(i))
            store.put([key], now=i)
            self.assertEqual(len(store.objects), store.offload.used_blocks)
            self.assertLessEqual(len(store.objects), 2)
        self.assertEqual(store.lookup([key], now=20).hit_blocks, 1)

    def test_transfer_units_and_invalid_configuration_cannot_hide_bad_budgets(self):
        config = MooncakeConfig(
            bandwidth_bytes_per_s=1000,
            transfer_utilization=0.5,
            fixed_latency_us=0,
            ssd_capacity_bytes=2000,
            ssd_read_bandwidth_bytes_per_s=1000,
            ssd_write_bandwidth_bytes_per_s=1000,
            ssd_read_latency_us=0,
            ssd_write_latency_us=0,
        )
        transfer = TransferEngineSimulator(config).create_transfer(
            source_segment=Segment("gpu", "vram"),
            target_segment=Segment("host", "dram"),
            blocks=1,
            bytes_=1000,
        )
        self.assertEqual(transfer.latency, 2)
        disk = OffloadTier(config, 1000)
        self.assertEqual(disk.write("prefix", 1).latency, 1)
        self.assertEqual(disk.read("prefix").latency, 1)
        invalid = (
            {"bandwidth_bytes_per_s": 0},
            {"fixed_latency_us": float("nan")},
            {"global_segment_size": 1.5},
            {"enable_offload": "false"},
            {"global_segment_size": 10, "global_segment_size_gb": 1},
            {"protocol_bandwidth_bytes_per_s": {"rdm": 10}},
            {"bandwidth_gbps": 10},
            {"staging_buffer_size": 100},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                parse_mooncake_config(values)

    def test_uniform_store_publishes_only_computed_prompt_and_respects_roles(self):
        cache = SimpleNamespace(
            block_size=16, size_per_token=1, model="test", rank_info=None
        )
        producer = KVConnectorFactory.create_connector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_role="kv_producer",
            ),
            cache,
        )
        request = Request(1, 64, 2, 16, hash_ids=[1, 2, 3, 4])
        request.scheduled_tokens = 32
        output = SimpleNamespace(scheduled=[request], preempted=[])
        self.assertEqual(producer.build_connector_meta(output).saves, [])
        request.num_computed_tokens = 32
        saved = producer.build_connector_meta(output)
        self.assertEqual(sum(plan.blocks for plan in saved.saves), 4)
        producer.bind_connector_metadata(saved)
        producer.wait_for_save()
        self.assertEqual(producer.get_num_new_matched_tokens(request, 0), 0)
        self.assertEqual(producer.build_connector_meta(output).saves, [])
        consumer = KVConnectorFactory.create_connector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_role="kv_consumer",
            ),
            cache,
        )
        self.assertEqual(consumer.build_connector_meta(output).saves, [])

    def test_mooncake_pool_and_staging_share_one_checked_host_budget(self):
        worker = SimpleNamespace(
            engine=SimpleNamespace(workers=[None, None]),
            parallel_config=SimpleNamespace(world_size=2),
            env=simpy.Environment(),
            hardware_context=make_hardware(),
            nettype="ethernet-test",
        )
        for mode, capacity in (("embedded", 2000), ("standalone-store", 1000)):
            config = MooncakeConfig(
                mode=mode,
                global_segment_size=1000,
                local_buffer_size=20,
                host_available_bytes=capacity + 40,
            )
            service = _SharedService(worker, config)
            self.assertEqual(service.store.capacity_bytes, capacity)
            self.assertEqual(service.staging_bytes, 40)
            other = SimpleNamespace(
                **{**vars(worker), "engine": SimpleNamespace(workers=[None])}
            )
            variant = MooncakeConfig(
                mode=mode,
                global_segment_size=500 if mode == "embedded" else 1000,
                local_buffer_size=10,
                host_available_bytes=10000,
            )
            config.host_available_bytes = 10000
            service.register(worker, config)
            service.register(other, variant)
            service.register(other, variant)
            self.assertEqual(
                service.capacity_bytes, capacity + (500 if mode == "embedded" else 0)
            )
            self.assertEqual(service.staging_bytes, 50)
            config.host_available_bytes = capacity + 40
            config.host_available_bytes -= 1
            with self.assertRaisesRegex(ConfigurationError, "available host"):
                _SharedService(worker, config)

    def test_incomplete_write_is_invisible_and_pinned_read_survives_eviction(self):
        store = NativePageStore(200)
        a = store.reserve_write(("a",), 100)
        self.assertIsNone(store.lookup(("a",)))
        store.finish_write(a)
        store.acquire(a)
        b = store.reserve_write(("b",), 100)
        store.finish_write(b)
        c = store.reserve_write(("c",), 100)
        self.assertIsNotNone(c)
        self.assertIs(store.lookup(("a",)), a)
        self.assertIsNone(store.lookup(("b",)))
        store.release(a)
        store.finish_write(c)
        store.validate()

    def test_simple_cpu_counts_packed_stride_while_objects_use_actual_group_bytes(self):
        cpu = NativePageStore(1000, allocation_unit=128, scope="worker0")
        self.assertEqual(cpu.capacity_bytes // cpu.allocation_unit, 7)
        page = cpu.reserve_write(("state",), 32)
        self.assertEqual(page.bytes, 128)
        shared = NativePageStore(1000)
        p = shared.reserve_write(("model", 0, "hash"), 32)
        self.assertEqual(p.bytes, 32)
        shared.finish_write(p)
        self.assertIs(shared.lookup(("model", 0, "hash")), p)

    def test_all_pinned_storage_refuses_new_write_without_exceeding_budget(self):
        store = NativePageStore(100)
        page = store.reserve_write(("a",), 100)
        self.assertIsNone(store.reserve_write(("b",), 100))
        self.assertFalse(page.ready)
        store.validate()

    def test_directional_link_serializes_competing_callers_and_counts_wait(self):
        env = simpy.Environment()
        link = QueuedKVLink(env, 100, utilization=0.5)
        a = env.process(link.transfer(100))
        b = env.process(link.transfer(100))
        env.run(until=simpy.AllOf(env, [a, b]))
        self.assertEqual(env.now, 4)
        self.assertEqual(link.queue_wait_seconds, 2)
        self.assertEqual(link.bytes, 200)

    def test_utilization_changes_service_without_multiplying_independent_bandwidth(
        self,
    ):
        for utilization, expected in ((0.5, 2), (0.75, 4 / 3), (1, 1)):
            env = simpy.Environment()
            link = QueuedKVLink(env, 100, utilization=utilization)
            env.run(until=env.process(link.transfer(100)))
            self.assertAlmostEqual(env.now, expected)
