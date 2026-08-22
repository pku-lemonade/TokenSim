from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import (
    KVConnectorFactory,
    KVConnectorMetadata,
    MooncakeConnector,
    MooncakeStoreConnector,
    MultiConnector,
)
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.mooncake import (
    KeyMetadata,
    MooncakeStore,
    PoolKey,
    Segment,
    TransferEngineSimulator,
    parse_mooncake_config,
    pool_keys_for_request,
)
from TokenSim.mooncake.service import reset_mooncake_services
from TokenSim.workload.loaders import load_json_pairs_workload
from util.request import LLMSource
from util.results import export_result, get_mooncake_stats


class _Roofline:
    def __init__(self):
        self.links = {
            "nvlink": type("Link", (), {"Latency": 1e-6, "UniBW": 300.0})(),
            "ethernet-test": type("Link", (), {"Latency": 1e-5, "UniBW": 12.5})(),
        }
        self.hardwares = {
            "TestGPU": type(
                "Hardware",
                (),
                {
                    "Name": "TestGPU",
                    "MM_Card_Num": 1,
                    "Capacity": 80,
                    "Nvlink": "nvlink",
                    "Pcie": "ethernet-test",
                    "MM_TFLOPS": 100,
                },
            )()
        }
        self.models = {
            "TestModel": type(
                "Model",
                (),
                {
                    "Name": "TestModel",
                    "Nlayer": 2,
                    "Dmodel": 128,
                    "Nhead": 8,
                    "FFN_Hidden": 256,
                },
            )()
        }

    def Compute_Timebreakdown_Iteration(self, *args, **kwargs):
        return 0.001, 0.0005


class _CacheConfig:
    block_size = 16
    size_per_token = 8
    model = "TestModel"
    rank_info = ParallelRankInfo()


def _psla() -> PSLAConfig:
    return PSLAConfig(
        name="test",
        model="TestModel",
        distribution="burst",
        prefill_mean_len=32,
        prefill_range_len=0,
        decode_mean_len=2,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
    )


class MooncakeConfigTransferTest(unittest.TestCase):
    def test_config_files_parse_and_invalid_protocol_fails(self):
        for path in (
            "./data/kv_transfer/mooncake_p2p.json",
            "./data/kv_transfer/mooncake_store_embedded.json",
            "./data/kv_transfer/mooncake_store_standalone_ssd.json",
            "./data/kv_transfer/mooncake_multi_connector.json",
        ):
            config = KVTransferConfig.from_file(path)
            self.assertIn(config.kv_connector, {"MooncakeConnector", "MooncakeStoreConnector", "MultiConnector"})

        with self.assertRaises(ConfigurationError):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"protocol": "bad"},
            )
        with self.assertRaises(ConfigurationError):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"memory_capacity_blocks": -1},
            )
        with self.assertRaises(ConfigurationError):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"offload_tier": "bad"},
            )
    def test_transfer_engine_protocol_latency_and_overlap(self):
        config = parse_mooncake_config(
            {
                "protocol": "rdma",
                "num_nics": 2,
                "load_async": True,
                "transfer_overlap": True,
            }
        )
        engine = TransferEngineSimulator(config)
        rdma = engine.create_transfer(
            source_segment=Segment("a", "dram"),
            target_segment=Segment("b", "vram"),
            blocks=1,
            bytes_=1 << 20,
            protocol="rdma",
        )
        tcp = engine.create_transfer(
            source_segment=Segment("a", "dram"),
            target_segment=Segment("b", "vram"),
            blocks=1,
            bytes_=1 << 20,
            protocol="tcp",
        )
        job = engine.submit(rdma)

        self.assertLess(rdma.latency, tcp.latency)
        self.assertGreater(
            engine.create_transfer(
                source_segment=Segment("a", "dram"),
                target_segment=Segment("b", "vram"),
                blocks=1,
                bytes_=1 << 20,
                protocol="rdma",
                topology="cross_node",
            ).latency,
            rdma.latency,
        )
        self.assertEqual(job.blocking_latency, 0)
        self.assertEqual(engine.pending_jobs(0), 1)

    def test_json_pairs_loader_preserves_hash_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reuse.json"
            path.write_text(
                json.dumps(
                    [[32, 2, {"hash_ids": ["a", "b"], "reuse_group": "tenant"}]]
                )
            )
            workload = load_json_pairs_workload(str(path), request_count=1)

        self.assertEqual(workload[0].hash_ids, ["a", "b"])
        self.assertEqual(workload[0].reuse_group, "tenant")


class MooncakeStoreModelTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_pool_key_includes_rank_context(self):
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"], reuse_group="tenant")
        rank0 = ParallelRankInfo(tp_rank=0, pp_rank=0, dp_rank=0, kv_cache_group_id="dp0-pp0")
        rank1 = ParallelRankInfo(tp_rank=1, pp_rank=0, dp_rank=0, kv_cache_group_id="dp0-pp0")

        key0 = pool_keys_for_request(req, model_name="m", rank_info=rank0, engine_id="e")[0]
        key1 = pool_keys_for_request(req, model_name="m", rank_info=rank1, engine_id="e")[0]

        self.assertNotEqual(key0, key1)
        self.assertIn("@tp_rank:0", key0.to_string())
        self.assertIn("@tp_rank:1", key1.to_string())

    def test_store_memory_hit_eviction_and_ssd_hit(self):
        config = parse_mooncake_config(
            {
                "memory_capacity_blocks": 1,
                "enable_offload": True,
                "ssd_capacity_blocks": 4,
            }
        )
        store = MooncakeStore(config, block_bytes=128)
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")

        store.put([key0])
        self.assertEqual(store.lookup([key0]).tier, "memory")
        store.put([key1])
        hit = store.lookup([key0])

        self.assertEqual(hit.tier, "ssd")
        self.assertGreater(store.get(hit.keys, hit.tier), 0)
        self.assertGreater(store.stats.ssd_read_blocks, 0)
        self.assertGreater(store.stats.ssd_write_blocks, 0)
        self.assertEqual(store.objects[key0.to_string()].replicas, 1)
        self.assertTrue(store.objects[key0.to_string()].persisted)

    def test_ssd_media_writes_are_blocking(self):
        block_bytes = 10 * (1 << 20)
        ssd_store = MooncakeStore(
            parse_mooncake_config(
                {
                    "memory_capacity_blocks": 0,
                    "enable_offload": True,
                    "offload_tier": "ssd",
                    "ssd_capacity_blocks": 4,
                }
            ),
            block_bytes=block_bytes,
        )
        key = PoolKey(KeyMetadata("m"), "ssd-blocking")
        timing = ssd_store.put_with_timing([key], now=1.0)
        expected = 200e-6 + block_bytes / (1 << 30) / 3.0
        self.assertAlmostEqual(timing.accounting_latency, expected)
        self.assertAlmostEqual(timing.blocking_latency, expected)

    def test_ssd_offload_profile_exports_media_parameters(self):
        ssd_store = MooncakeStore(
            parse_mooncake_config(
                {
                    "enable_offload": True,
                    "offload_tier": "ssd",
                    "ssd_capacity_gb": 2048,
                }
            ),
            block_bytes=4096,
        )
        ssd_profile = ssd_store.stats.as_dict()["mooncake_offload_profiles"][0]
        self.assertEqual(ssd_profile["tier"], "ssd")
        self.assertEqual(ssd_profile["ssd_capacity_gb"], 2048)
        self.assertEqual(ssd_profile["ssd_read_bw_gbps"], 7.0)
        self.assertEqual(ssd_profile["ssd_write_bw_gbps"], 3.0)
        self.assertTrue(ssd_profile["write_blocking"])
        self.assertEqual(ssd_profile["write_persistence"], "blocking")

    def test_store_rejects_when_no_eviction_or_admission_disabled(self):
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")
        no_evict = MooncakeStore(
            parse_mooncake_config(
                {
                    "memory_capacity_blocks": 1,
                    "eviction_policy": "none",
                }
            ),
            block_bytes=128,
        )

        no_evict.put([key0])
        no_evict.put([key1])

        self.assertEqual(no_evict.stats.admission_count, 1)
        self.assertEqual(no_evict.stats.admission_rejection_count, 1)

        never = MooncakeStore(
            parse_mooncake_config({"admission_policy": "never"}),
            block_bytes=128,
        )
        never.put([key0])
        self.assertEqual(never.stats.admission_count, 0)
        self.assertEqual(never.stats.admission_rejection_count, 1)


class MooncakeConnectorTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_factory_creates_mooncake_connectors(self):
        p2p = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="MooncakeConnector"),
            _CacheConfig(),
        )
        store = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="MooncakeStoreConnector"),
            _CacheConfig(),
        )

        self.assertIsInstance(p2p, MooncakeConnector)
        self.assertIsInstance(store, MooncakeStoreConnector)

    def test_mooncake_p2p_records_transfer(self):
        connector = MooncakeConnector(
            KVTransferConfig(kv_connector="MooncakeConnector"),
            _CacheConfig(),
        )
        req = Request(0, 16, 1, 16)
        req._physical_token_blocks = [object(), object()]
        plan = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=None,
        )
        connector.bind_connector_metadata(KVConnectorMetadata(connector_name="MooncakeConnector", loads=[plan]))

        latency = connector.start_load_kv()

        self.assertGreater(latency, 0)
        self.assertGreater(connector.stats.transfer_count, 0)
        self.assertGreater(connector.mooncake_stats.p2p_latency, 0)
        self.assertEqual(plan.extra["protocol"], "rdma")
        self.assertEqual(plan.extra["topology"], "cross_node")
        self.assertTrue(plan.keys)

    def test_multi_connector_delegates_p2p_transfer_plan(self):
        connector = KVConnectorFactory.create_connector(
            KVTransferConfig.from_file("./data/kv_transfer/mooncake_multi_connector.json"),
            _CacheConfig(),
        )
        req = Request(0, 16, 1, 16)
        req._physical_token_blocks = [object()]

        plan = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=None,
        )
        connector.bind_connector_metadata(
            KVConnectorMetadata(connector_name="MultiConnector", loads=[plan])
        )
        latency = connector.start_load_kv()

        self.assertIsInstance(connector, MultiConnector)
        self.assertGreater(latency, 0)
        self.assertEqual(plan.extra["connector"], "MooncakeConnector")
        self.assertGreater(connector.children[0].stats.transfer_count, 0)

    def test_store_connector_memory_and_disk_hits(self):
        memory_connector = MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 8,
                    "enable_offload": False,
                },
            ),
            _CacheConfig(),
        )
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        req1 = Request(1, 32, 1, 16, hash_ids=["a", "b"])
        meta0 = memory_connector.build_connector_meta(
            type("Output", (), {"scheduled": [req0], "preempted": []})()
        )
        memory_connector.bind_connector_metadata(meta0)
        memory_connector.wait_for_save()
        hit_tokens = memory_connector.get_num_new_matched_tokens(req1, 0)
        memory_connector.update_state_after_alloc(
            req1,
            [
                type("Block", (), {"block_number": 1})(),
                type("Block", (), {"block_number": 2})(),
            ],
            hit_tokens,
        )
        meta1 = memory_connector.build_connector_meta(
            type("Output", (), {"scheduled": [req1], "preempted": []})()
        )

        # Aligned mooncake semantics: a full-prompt hit leaves the trailing
        # block uncomputed-from-cache, so 2 hit blocks are capped to 1.
        self.assertEqual(hit_tokens, 16)
        self.assertEqual(meta1.loads[0].tier, "memory")
        self.assertEqual(meta1.loads[0].blocks, 1)
        self.assertTrue(meta1.loads[0].keys)
        self.assertEqual(
            meta1.loads[0].keys,
            [key.to_string() for key in memory_connector._keys_for_request(req1)][
                : meta1.loads[0].blocks
            ],
        )

        reset_mooncake_services()
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "memory_capacity_blocks": 1,
                "enable_offload": True,
                "ssd_capacity_blocks": 8,
            },
        )
        connector = MooncakeStoreConnector(config, _CacheConfig())
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        req1 = Request(1, 32, 1, 16, hash_ids=["a", "b"])
        req2 = Request(2, 32, 1, 16, hash_ids=["c", "d"])

        meta0 = connector.build_connector_meta(type("Output", (), {"scheduled": [req0], "preempted": []})())
        connector.bind_connector_metadata(meta0)
        connector.wait_for_save()
        hit_tokens = connector.get_num_new_matched_tokens(req1, 0)
        self.assertEqual(hit_tokens, 16)
        meta1 = connector.build_connector_meta(type("Output", (), {"scheduled": [req1], "preempted": []})())
        self.assertEqual(meta1.loads[0].tier, "ssd")
        self.assertEqual(meta1.loads[0].blocks, 1)
        self.assertTrue(meta1.loads[0].keys)
        connector.bind_connector_metadata(meta1)
        self.assertGreater(connector.start_load_kv(), 0)

        meta2 = connector.build_connector_meta(type("Output", (), {"scheduled": [req2], "preempted": []})())
        connector.bind_connector_metadata(meta2)
        connector.wait_for_save()
        hit_tokens = connector.get_num_new_matched_tokens(req1, 0)

        self.assertEqual(hit_tokens, 16)
        self.assertEqual(connector.service.store.lookup(connector._keys_for_request(req1)).tier, "ssd")
        self.assertGreater(connector.mooncake_stats.disk_tier_hit_count, 0)

    def test_ssd_saves_are_blocking(self):
        connector = MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 0,
                    "enable_offload": True,
                    "offload_tier": "ssd",
                    "ssd_capacity_blocks": 8,
                },
            ),
            _CacheConfig(),
        )
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req], "preempted": []})()
        )
        connector.bind_connector_metadata(meta)
        connector.set_simulation_time(5.0)

        latency = connector.wait_for_save()

        self.assertGreater(latency, 0.0)
        self.assertAlmostEqual(connector.stats.save_wait_time, latency)
        self.assertAlmostEqual(connector.mooncake_stats.save_wait_time, latency)

    def test_store_connector_delays_release_until_async_feedback(self):
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "load_async": True,
                "transfer_overlap": True,
            },
        )
        connector = MooncakeStoreConnector(config, _CacheConfig())
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=type(
                "Cache",
                (),
                {
                    "block_size": 16,
                    "num_gpu_blocks": 8,
                    "num_cpu_blocks": 8,
                    "model": "TestModel",
                },
            )(),
            max_parallem_sum=4,
            connector=connector,
        )
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        scheduler.add_requests([req])
        running, _ = scheduler.schedule()
        meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": running, "preempted": []})()
        )
        connector.bind_connector_metadata(meta)
        req.generation_idx = req.decode_len

        scheduler.free(req)

        self.assertEqual(req.status.name, "WAITING_FOR_CONNECTOR_FREE")
        self.assertEqual(scheduler.block_manager.block_table.get_num_blocks(req.id), 2)

        connector.wait_for_save()
        worker_meta = connector.build_connector_worker_meta()
        scheduler.update_connector_output(worker_meta)

        self.assertEqual(req.status.name, "FINISHED_STOPPED")
        self.assertEqual(scheduler.block_manager.block_table.get_num_blocks(req.id), 0)


class MooncakeAlignedSemanticsTest(unittest.TestCase):
    """Pin the reference-aligned save semantics (vLLM MooncakeStoreConnector).

    Reference: ref/vllm .../kv_connector/v1/mooncake/store/{scheduler,data,worker}.py
    - each request saves at most once, during prefill; decode steps never save
    - a step that loads from the store skips its save
    - the save existence-prefilter transfers only keys missing from the store
    """

    def setUp(self):
        reset_mooncake_services()

    def _connector(self, **extra):
        config = {"memory_capacity_blocks": 64, "enable_offload": False}
        config.update(extra)
        return MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config=config,
            ),
            _CacheConfig(),
        )

    @staticmethod
    def _meta(connector, requests):
        return connector.build_connector_meta(
            type("Output", (), {"scheduled": requests, "preempted": []})()
        )

    def test_save_happens_once_during_prefill_and_never_on_decode(self):
        connector = self._connector()
        req = Request(0, 32, 4, 16, hash_ids=["a", "b"])

        prefill_meta = self._meta(connector, [req])
        self.assertEqual(len(prefill_meta.saves), 1)
        self.assertEqual(prefill_meta.saves[0].blocks, 2)
        connector.bind_connector_metadata(prefill_meta)
        connector.wait_for_save()

        req.generation_idx = 1  # decode steps
        for _ in range(3):
            decode_meta = self._meta(connector, [req])
            self.assertEqual(decode_meta.saves, [])
        self.assertEqual(connector.mooncake_stats.put_count, 1)

    def test_load_step_precludes_save(self):
        connector = self._connector()
        req0 = Request(0, 48, 1, 16, hash_ids=["a", "b", "c"])
        connector.bind_connector_metadata(self._meta(connector, [req0]))
        connector.wait_for_save()

        req1 = Request(1, 48, 1, 16, hash_ids=["a", "b", "c"])
        hit_tokens = connector.get_num_new_matched_tokens(req1, 0)
        # Full-prompt hit capped to leave the trailing block uncomputed.
        self.assertEqual(hit_tokens, 32)
        meta = self._meta(connector, [req1])
        self.assertEqual(len(meta.loads), 1)
        self.assertEqual(meta.saves, [])
        # Decode steps do not retroactively save either.
        req1.generation_idx = 1
        self.assertEqual(self._meta(connector, [req1]).saves, [])

    def test_save_transfers_only_missing_blocks(self):
        connector = self._connector()
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        connector.bind_connector_metadata(self._meta(connector, [req0]))
        connector.wait_for_save()

        # Shares the (a, b) prefix chain; only (c, d) are missing.
        req1 = Request(1, 64, 1, 16, hash_ids=["a", "b", "c", "d"])
        meta = self._meta(connector, [req1])
        self.assertEqual(len(meta.saves), 1)
        self.assertEqual(meta.saves[0].blocks, 2)
        self.assertEqual(
            meta.saves[0].keys,
            connector._key_strings_for_request(req1)[2:4],
        )

        # A request whose prompt is fully present produces no save plan.
        req2 = Request(2, 32, 1, 16, hash_ids=["a", "b"])
        self.assertEqual(self._meta(connector, [req2]).saves, [])

    def test_every_step_legacy_policy_resaves_full_prompt(self):
        connector = self._connector(save_policy="every_step")
        req = Request(0, 32, 2, 16, hash_ids=["a", "b"])
        first = self._meta(connector, [req])
        self.assertEqual(len(first.saves), 1)
        self.assertEqual(first.saves[0].blocks, 2)
        req.generation_idx = 1
        second = self._meta(connector, [req])
        self.assertEqual(len(second.saves), 1)
        self.assertEqual(second.saves[0].blocks, 2)

    def test_on_request_released_clears_prefill_side_state(self):
        connector = self._connector()
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        connector.bind_connector_metadata(self._meta(connector, [req]))
        connector.wait_for_save()
        self.assertIn(req.id, connector._keys_cache)

        connector.on_request_released(req)

        self.assertNotIn(req.id, connector._keys_cache)
        self.assertNotIn(req.id, connector._request_keys)
        self.assertNotIn(req.id, connector._saved_request_ids)


class MooncakeEndToEndTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()
        reset_g_time()

    def test_engine_exports_mooncake_store_metrics(self):
        env = simpy.Environment()
        cluster = ClusterConfig(
            num_workers=1,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="TestGPU",
                    num_workers=1,
                    network="net1",
                )
            ],
        )
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.from_file(
                "./data/kv_transfer/mooncake_store_embedded.json"
            ),
            psla_config=_psla(),
            cluster_config=cluster,
            roofline=_Roofline(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
        requests = [
            Request(0, 32, 2, 16, inter_arrival_time=0.05, hash_ids=["a", "b"]),
            Request(1, 32, 2, 16, hash_ids=["a", "b"]),
        ]
        env.process(LLMSource(env, engine, requests, qps=1000, distribution="uniform"))

        def stop_when_done():
            while env.now < 1:
                if all(req.is_done for req in requests):
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return env.now
                yield env.timeout(1e-6)
            raise AssertionError("Mooncake requests did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        stats = get_mooncake_stats(engine)

        self.assertGreater(stats["mooncake_put_count"], 0)
        self.assertGreater(stats["mooncake_memory_tier_hit_count"], 0)
        self.assertGreater(stats["mooncake_transferred_bytes"], 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_result(
                args=argparse.Namespace(
                    qps=1,
                    batching="paged-attn",
                    results_path=tmpdir,
                    cluster="unused.json",
                ),
                g_time=g_time,
                engine=engine,
                model_config=_psla(),
                cluster=cluster,
                request_count=2,
                prefill_lens=[32, 32],
                decode_lens=[2, 2],
                requests=requests,
                notdone=[],
                duration=monitor.value,
                simulator_wall_time=0.1,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())
        self.assertGreater(result["mooncake_memory_tier_hit_count"], 0)
        self.assertGreater(result["mooncake_transferred_bytes"], 0)

    def test_non_mooncake_export_has_zero_defaults(self):
        env = simpy.Environment()
        cluster = ClusterConfig(
            num_workers=1,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="TestGPU",
                    num_workers=1,
                    network="net1",
                )
            ],
        )
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=_psla(),
            cluster_config=cluster,
            roofline=_Roofline(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
        request = Request(0, 32, 2, 16)
        request.arrive(env)
        request.step(env, 0.01, 1)
        request.step(env, 0.01, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_result(
                args=argparse.Namespace(
                    qps=1,
                    batching="paged-attn",
                    results_path=tmpdir,
                    cluster="unused.json",
                ),
                g_time=g_time,
                engine=engine,
                model_config=_psla(),
                cluster=cluster,
                request_count=1,
                prefill_lens=[32],
                decode_lens=[2],
                requests=[request],
                notdone=[],
                duration=0.1,
                simulator_wall_time=0.01,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())

        self.assertEqual(result["mooncake_get_count"], 0)
        self.assertEqual(result["mooncake_transferred_bytes"], 0)

    def test_multi_connector_store_and_p2p_metrics_aggregate(self):
        connector = KVConnectorFactory.create_connector(
            KVTransferConfig.from_file("./data/kv_transfer/mooncake_multi_connector.json"),
            _CacheConfig(),
        )
        store = connector.children[1]
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        req1 = Request(1, 32, 1, 16, hash_ids=["a", "b"])
        req0._physical_token_blocks = [object(), object()]
        req1._physical_token_blocks = [object(), object()]

        save_meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req0], "preempted": []})()
        )
        connector.bind_connector_metadata(save_meta)
        connector.wait_for_save()
        self.assertEqual(connector.get_num_new_matched_tokens(req1, 0), 16)
        load_meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req1], "preempted": []})()
        )
        connector.bind_connector_metadata(load_meta)
        connector.start_load_kv()
        p2p_plan = connector.build_transfer_plan(
            requests=[req1],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=None,
        )
        connector.bind_connector_metadata(
            type("Meta", (), {"loads": [p2p_plan], "saves": [], "preempted_request_ids": []})()
        )
        connector.start_load_kv()

        self.assertGreater(store.mooncake_stats.memory_tier_hit_count, 0)
        self.assertGreater(connector.children[0].mooncake_stats.p2p_latency, 0)


if __name__ == "__main__":
    unittest.main()
