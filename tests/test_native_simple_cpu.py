from __future__ import annotations

import unittest
from types import SimpleNamespace

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig
from TokenSim.block.hybrid_manager import HybridKVBlockManager
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import RequestStatus
from tests.test_hybrid_kv_cache import request


def engine(env, *, lockstep=True, extra=None, connector_name="SimpleCPUOffloadConnector"):
    cluster = ClusterConfig.from_file("data/clusters/8_b200/dep8.json")
    e = LLMEngine(
        env=env,
        block_size=256,
        batching="paged-attn",
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector_name,
            kv_connector_extra_config=extra or {"cpu_bytes_to_use_per_rank": 200000000},
        ),
        psla_config=PSLAConfig.from_file("data/psla/deepseek-v4-pro.json"),
        cluster_config=cluster,
        hardware=HardwareContext.load("data"),
        parallel_config=cluster.effective_parallel_config(),
        prefill_worker_pool_type="round_robin",
        decode_worker_pool_type="round_robin",
        max_parallem_sum=49,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.9,
        latency_backend_type="analytical",
        dp_lockstep=lockstep,
    )
    for w in e.workers:
        w.dynamic_batch = lambda requests: 0.001
    return e


class NativeSimpleCPUTest(unittest.TestCase):
    def test_generated_checkpoint_cpu_resume_in_both_native_backends(self):
        from tests.test_hybrid_kv_cache import NativeKVPoolTest
        for name, extra in (
            ("SimpleCPUOffloadConnector", {"cpu_bytes_to_use_per_rank": 4000000000}),
            ("MooncakeStoreConnector", {"mode": "standalone-store", "protocol": "rdma",
             "global_segment_size": 4000000000, "local_buffer_size": 1000000,
             "load_async": True, "transfer_overlap": True, "bandwidth_gbps": 50}),
        ):
            with self.subTest(connector=name):
                env = simpy.Environment()
                e = engine(env, extra=extra, connector_name=name)
                w = e.workers[0]
                manager = w.scheduler.block_manager
                r = request(75, 32256)
                r.decode_len = 2048
                r.record_context_admission(0, 0)
                manager.allocate(r)
                helper = NativeKVPoolTest()
                helper.compute(manager, r, 32256)
                r._append_tokens(1025)
                r.generation_idx = 1025
                helper.compute(manager, r, 1024)
                manager.release_request_blocks(r)
                r.prepare_recompute()
                # Manual cache production bypasses the worker's post-forward
                # save hook; run that hook before expecting CPU-ready pages.
                self.assertTrue(w.connector.pending_writes)
                w.connector.wait_for_save()
                env.run(until=0.1)
                self.assertEqual(w.connector.write_jobs, 0)
                for block in manager.blocks[1:]:
                    if block.block_hash is not None:
                        manager._evict(block)
                self.assertEqual(manager.plan_reuse(r).hit_tokens, 0)
                boundary, _ = w.connector.probe(r, manager.plan_reuse(r))
                self.assertEqual(boundary, 32768)
                r.dp_rank = 0
                r.arrive(env)
                e.add_requests_burst([r])
                env.run(until=r.completion_event)
                env.run(until=env.now + 0.1)
                e.raise_if_failed()
                self.assertGreaterEqual(r.reloaded_generated_cpu_tokens, 512)
                self.assertEqual(r.first_prefill_source.computed_tokens, 32256)
                self.assertLessEqual(r.cached_prefill_tokens, r.prefill_len)
                self.assertEqual(r.generation_idx, r.decode_len)
                self.assertFalse(manager.transfer_references)
                self.assertEqual(manager.get_num_allocated_blocks(), 1)
                self.assertTrue(manager.validate_conservation())
                self.assertTrue(w.connector.store.validate())

    def test_cache_metric_decomposition_does_not_fill_missing_legacy_cpu_observations(
        self,
    ):
        from util.results import get_agentx_metrics

        r = request(0, 1024)
        r.cached_prefill_tokens = 768
        r.external_cached_prefill_tokens = 512
        r.effective_prefill_tokens = 256
        args = SimpleNamespace(workload_type="agentx_weka", agentx_runtime_metadata={})
        self.assertNotIn("cpu_cache_hit_rate", get_agentx_metrics(args, [r], 1, 8))
        args.agentx_runtime_metadata = {"replay_protocol": "native"}
        r.record_prefill_admission(256, 512)
        metrics = get_agentx_metrics(args, [r], 1, 8)
        self.assertEqual(metrics["gpu_cache_hit_rate"], 0.25)
        self.assertEqual(metrics["cpu_cache_hit_rate"], 0.5)
        self.assertEqual(metrics["computed_input_tokens"], 256)

    def seeded(self, *, lockstep=True):
        env = simpy.Environment()
        e = engine(env, lockstep=lockstep)
        r = request(0, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        w = e.workers[0]
        self.assertTrue(w.connector.store.pages)
        self.assertFalse(w.scheduler.block_manager.transfer_references)
        # Remove local cache metadata, as GPU LRU pressure would, leaving CPU.
        for b in w.scheduler.block_manager.blocks[1:]:
            if b.block_hash is not None:
                self.assertEqual(b.ref_count, 0)
                w.scheduler.block_manager._evict(b)
        return env, e, w

    def test_full_engine_cpu_reload_recomputes_last_block_and_preserves_all_groups(
        self,
    ):
        env, e, w = self.seeded()
        r = request(1, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertEqual(r.gpu_cached_prefill_tokens, 0)
        self.assertEqual(r.external_cached_prefill_tokens, 3840)
        self.assertEqual(r.prefill_chunks, 1)
        self.assertEqual(r.generation_idx, 4)
        for worker in e.workers:
            self.assertTrue(worker.scheduler.block_manager.validate_conservation())
            self.assertTrue(worker.connector.store.validate())
            self.assertEqual(
                worker.scheduler.block_manager.get_num_allocated_blocks(), 1
            )

    def test_loading_request_does_not_stop_other_rank_or_compute_before_dma(self):
        env, e, w = self.seeded()
        w.connector.read_link.bandwidth_bytes_per_s = 10000000
        loading = request(1, 4096)
        loading.dp_rank = 0
        other = request(2, 256, scope="other")
        other.dp_rank = 1
        for r in (loading, other):
            r.arrive(env)
        e.add_requests_burst([loading, other])
        env.run(until=other.completion_event)
        e.raise_if_failed()
        self.assertEqual(loading.status, RequestStatus.WAITING_FOR_KV)
        self.assertEqual(loading.generation_idx, 0)
        self.assertEqual(loading.num_computed_tokens, 0)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())
        env.run(until=loading.completion_event)
        self.assertLess(other.completed_at, loading.completed_at)

    def test_gpu_prefix_and_cpu_suffix_are_counted_separately(self):
        env, e, w = self.seeded()
        short = request(1, 1024)
        short.dp_rank = 0
        short.arrive(env)
        e.add_requests_burst([short])
        env.run(until=short.completion_event)
        env.run(until=env.now + 0.01)
        long = request(2, 4096)
        local = w.scheduler.block_manager.plan_reuse(long)
        self.assertEqual(local.hit_tokens, 768)
        # An older CPU prefix can be evicted while its GPU copy remains.
        full = w.scheduler.block_manager.layout.full_group
        keys = w.scheduler.block_manager._input_keys(long)
        for i in range(3):
            key = (full.group_id, keys[i], 0)
            page = w.connector.store.pages.pop(key)
            self.assertEqual(page.references, 0)
            w.connector.store.lru.pop(key)
            w.connector.store.used_bytes -= page.bytes
        long.dp_rank = 0
        long.arrive(env)
        e.add_requests_burst([long])
        env.run(until=long.completion_event)
        e.raise_if_failed()
        self.assertEqual(long.gpu_cached_prefill_tokens, 768)
        self.assertEqual(long.external_cached_prefill_tokens, 3072)
        self.assertEqual(long.cached_prefill_tokens, 3840)
        self.assertEqual(long.effective_prefill_tokens, 256)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())

    def test_cpu_lookup_precedes_full_input_gate(self):
        env, e, w = self.seeded()
        old = w.scheduler.block_manager
        manager = HybridKVBlockManager(old.layout, 160, old.model)
        w.scheduler.block_manager = w.connector.manager = manager
        manager.publish_callbacks.append(w.connector._queue_write)
        w.scheduler.max_parallem_sum = 1
        w.connector.read_link.bandwidth_bytes_per_s = 10000000
        a = request(1, 4096)
        self.assertFalse(manager.can_allocate(a))
        boundary, hits = w.connector.probe(a, manager.plan_reuse(a))
        self.assertTrue(
            manager.can_allocate_external(a, manager.plan_reuse(a), boundary, hits)
        )
        a.dp_rank = 0
        a.arrive(env)
        e.add_requests_burst([a])
        env.run(until=env.now + 0.005)
        self.assertEqual(list(w.scheduler.loading), [a.id])
        self.assertEqual(a.generation_idx, 0)
        env.run(until=a.completion_event)
        e.raise_if_failed()
        self.assertTrue(a.is_done)
        self.assertTrue(manager.validate_conservation())

    def test_loading_does_not_use_running_quota_and_ready_kv_waits_for_a_slot(self):
        env, e, w = self.seeded()
        w.scheduler.max_parallem_sum = 1
        w.connector.read_link.bandwidth_bytes_per_s = 1000000000000
        a, b = request(1, 4096), request(2, 4096)
        for r in (a, b):
            r.dp_rank = 0
            r.arrive(env)
        e.add_requests_burst([a, b])
        env.run(until=env.now + 0.0005)
        self.assertEqual(w.connector.read_link.jobs, 2)
        self.assertFalse(w.connector.loads)
        self.assertFalse(w.scheduler.running)
        self.assertEqual(w.scheduler.loaded_waiting, [a, b])
        # Completed copies are admitted only after the current GPU pass ends.
        env.run(until=env.now + 0.0006)
        self.assertEqual(w.scheduler.running, [a])
        self.assertEqual(w.scheduler.loaded_waiting, [b])
        self.assertEqual(b.generation_idx, 0)
        env.run(until=b.completion_event)
        e.raise_if_failed()
        self.assertTrue(a.is_done and b.is_done)
        self.assertFalse(w.scheduler.loaded_waiting)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())

    def test_two_loads_share_direction_queue_and_release_duplicate_destinations(self):
        env, e, w = self.seeded()
        w.connector.read_link.bandwidth_bytes_per_s = 10000000
        a, b = request(1, 4096), request(2, 4096)
        for r in (a, b):
            r.dp_rank = 0
            r.arrive(env)
        e.add_requests_burst([a, b])
        env.run(until=b.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertLess(a.completed_at, b.completed_at)
        self.assertEqual(w.connector.read_link.jobs, 2)
        self.assertGreater(w.connector.read_link.queue_wait_seconds, 0)
        self.assertFalse(w.scheduler.block_manager.transfer_references)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())
        self.assertTrue(w.connector.store.validate())

    def test_cancel_loaded_request_while_waiting_for_running_slot(self):
        env, e, w = self.seeded()
        w.scheduler.max_parallem_sum = 1
        w.connector.read_link.bandwidth_bytes_per_s = 1000000000000
        a, b = request(1, 4096), request(2, 4096)
        for r in (a, b):
            r.dp_rank = 0
            r.arrive(env)
        e.add_requests_burst([a, b])
        env.run(until=env.now + 0.0011)
        self.assertEqual(w.scheduler.loaded_waiting, [b])
        ack = e.cancel_requests([b])[0]
        env.run(until=ack)
        env.run(until=a.completion_event)
        e.raise_if_failed()
        self.assertEqual(b.generation_idx, 0)
        self.assertFalse(w.scheduler.loaded_waiting)
        self.assertFalse(w.scheduler.loading)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())

    def test_cancelled_dma_is_pinned_until_completion_and_never_publishes(self):
        env, e, w = self.seeded()
        w.connector.read_link.bandwidth_bytes_per_s = 10000000
        r = request(1, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=env.now + 0.005)
        self.assertEqual(r.status, RequestStatus.WAITING_FOR_KV)
        event = e.cancel_requests([r])[0]
        env.run(until=event)
        manager = w.scheduler.block_manager
        self.assertTrue(manager.transfer_references)
        self.assertFalse(manager.tables)
        self.assertTrue(manager.validate_conservation())
        env.run(until=env.now + 10)
        e.raise_if_failed()
        self.assertFalse(manager.transfer_references)
        self.assertFalse(manager.cache)
        self.assertFalse(w.scheduler.running)
        self.assertEqual(r.generation_idx, 0)
        self.assertTrue(w.connector.store.validate())
        self.assertEqual(manager.get_num_allocated_blocks(), 1)

    def test_stores_pin_computed_pages_after_request_finish_and_are_invisible_until_copy(
        self,
    ):
        env = simpy.Environment()
        e = engine(env)
        w = e.workers[0]
        w.connector.write_link.bandwidth_bytes_per_s = 10000000
        r = request(0, 1024)
        r.decode_len = 1
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=r.completion_event)
        manager = w.scheduler.block_manager
        self.assertFalse(manager.tables)
        self.assertTrue(manager.transfer_references)
        self.assertTrue(manager.validate_conservation())
        self.assertTrue(any(not p.ready for p in w.connector.store.pages.values()))
        self.assertTrue(
            all(
                w.connector.store.lookup(p.key) is None
                for p in w.connector.store.pages.values()
                if not p.ready
            )
        )
        env.run(until=env.now + 10)
        self.assertFalse(manager.transfer_references)
        self.assertTrue(
            all(p.ready and p.references == 0 for p in w.connector.store.pages.values())
        )

    def test_non_lockstep_loading_worker_can_process_cancellation(self):
        env, e, w = self.seeded(lockstep=False)
        w.connector.read_link.bandwidth_bytes_per_s = 10000000
        r = request(1, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=env.now + 0.005)
        issued = env.now
        event = e.cancel_requests([r])[0]
        env.run(until=event)
        self.assertEqual(env.now, issued)
        self.assertTrue(w.connector.loads)
        env.run(until=env.now + 10)
        self.assertFalse(w.connector.loads)

    def test_server_wide_budget_divides_world_size_and_rank_override_is_explicit(self):
        for extra in (
            {"cpu_bytes_to_use": 1600000000},
            {"cpu_bytes_to_use": 8, "cpu_bytes_to_use_per_rank": 200000000},
        ):
            e = engine(simpy.Environment(), extra=extra)
            for w in e.workers:
                c = w.connector
                self.assertEqual(c.budget_bytes, 200000000)
                self.assertEqual(
                    c.store.capacity_bytes, (200000000 // 1228800 - 1) * 1228800
                )
            self.assertIsNot(e.workers[0].connector.store, e.workers[1].connector.store)
        with self.assertRaises(ConfigurationError):
            engine(
                simpy.Environment(),
                extra={
                    "cpu_bytes_to_use_per_rank": 200000000,
                    "host_available_bytes": 1000000000,
                },
            )


if __name__ == "__main__":
    unittest.main()
