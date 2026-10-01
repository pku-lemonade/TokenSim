"""Shared native Mooncake pages must remain bounded and asynchronously owned."""

import unittest

import simpy

from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import RequestStatus
from tests.test_hybrid_kv_cache import request
from tests.test_native_simple_cpu import engine


EXTRA = {
    "mode": "standalone-store", "protocol": "rdma",
    "global_segment_size": 200000000, "local_buffer_size": 1000000,
    "load_async": True, "transfer_overlap": True, "bandwidth_gbps": 50,
}


class NativeMooncakeTest(unittest.TestCase):
    def make_engine(self, env, **extra):
        return engine(env, connector_name="MooncakeStoreConnector", extra={**EXTRA, **extra})

    def seed(self, **extra):
        env = simpy.Environment()
        e = self.make_engine(env, **extra)
        r = request(0, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        return env, e

    def submit(self, env, e, r, rank):
        r.dp_rank = rank
        r.arrive(env)
        e.add_requests_burst([r])

    def assert_clean(self, e):
        for w in e.workers:
            self.assertTrue(w.scheduler.block_manager.validate_conservation())
            self.assertFalse(w.scheduler.block_manager.transfer_references)
            self.assertFalse(w.connector.loads)
            self.assertEqual(w.connector.write_jobs, 0)
            self.assertEqual(w.scheduler.block_manager.get_num_allocated_blocks(), 1)
        self.assertTrue(e.native_mooncake_service.store.validate())
        self.assertTrue(all(p.references == 0 for p in e.native_mooncake_service.store.pages.values()))

    def test_shared_cross_rank_hit_and_complete_group_recovery(self):
        env, e = self.seed()
        a, b = e.workers[0].connector, e.workers[1].connector
        self.assertIs(a.store.backing, b.store.backing)
        self.assertEqual(a.store.namespace, b.store.namespace)
        self.assertIs(a.read_link.network, b.read_link.network)
        self.assertIsNot(a.host_read, b.host_read)
        r = request(1, 4096)
        self.submit(env, e, r, 1)
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertEqual(r.first_prefill_source.gpu_hit_tokens, 0)
        self.assertEqual(r.first_prefill_source.cpu_hit_tokens, 3840)
        self.assertEqual(r.prefill_chunks, 1)
        self.assertGreater(b.stats.transfer_bytes, 0)
        self.assert_clean(e)

    def test_cache_salt_and_parent_chain_do_not_leak_across_requests(self):
        env, e = self.seed()
        w = e.workers[1]
        changed_salt = request(2, 4096, scope="other-play")
        changed_parent = request(3, 4096, raw=[999] + list(range(1, 64)))
        for r in (changed_salt, changed_parent):
            plan = w.scheduler.block_manager.plan_reuse(r)
            self.assertEqual(w.connector.probe(r, plan), (0, ()))
        self.assert_clean(e)

    def test_inflight_save_invisible_and_source_pinned_after_request_finishes(self):
        env = simpy.Environment()
        e = self.make_engine(env)
        e.native_mooncake_service.write.bandwidth_bytes_per_s = 1000000
        r = request(0, 1024)
        r.decode_len = 1
        self.submit(env, e, r, 0)
        env.run(until=r.completion_event)
        store = e.native_mooncake_service.store
        self.assertTrue(any(not p.ready for p in store.pages.values()))
        self.assertTrue(e.workers[0].scheduler.block_manager.transfer_references)
        other = request(1, 1024)
        manager = e.workers[1].scheduler.block_manager
        self.assertEqual(e.workers[1].connector.probe(other, manager.plan_reuse(other)), (0, ()))
        env.run(until=env.now + 40)
        self.assert_clean(e)

    def test_reads_contend_on_shared_queue_and_other_rank_computes(self):
        env, e = self.seed()
        link = e.native_mooncake_service.read
        link.bandwidth_bytes_per_s = 10000000
        a, b, other = request(1, 4096), request(2, 4096), request(3, 256, scope="new")
        self.submit(env, e, a, 1)
        self.submit(env, e, b, 2)
        self.submit(env, e, other, 3)
        env.run(until=other.completion_event)
        self.assertEqual(a.status, RequestStatus.WAITING_FOR_KV)
        self.assertEqual(b.status, RequestStatus.WAITING_FOR_KV)
        self.assertEqual(a.generation_idx, 0)
        self.assertEqual(b.generation_idx, 0)
        env.run(until=b.completion_event)
        env.run(until=env.now + 0.01)
        self.assertLess(a.completed_at, b.completed_at)
        self.assertEqual(link.jobs, 2)
        self.assertGreater(link.queue_wait_seconds, 0)
        self.assert_clean(e)

    def test_cancelled_load_does_not_publish_and_keeps_dma_pins_until_completion(self):
        env, e = self.seed()
        e.native_mooncake_service.read.bandwidth_bytes_per_s = 10000000
        r = request(1, 4096)
        self.submit(env, e, r, 1)
        env.run(until=env.now + 0.005)
        self.assertEqual(r.status, RequestStatus.WAITING_FOR_KV)
        event = e.cancel_requests([r])[0]
        env.run(until=event)
        manager = e.workers[1].scheduler.block_manager
        self.assertTrue(manager.transfer_references)
        self.assertTrue(manager.validate_conservation())
        env.run(until=env.now + 10)
        self.assertFalse(manager.cache)
        self.assertEqual(r.generation_idx, 0)
        self.assert_clean(e)

    def test_evicted_shared_page_cannot_survive_in_a_local_ready_cache(self):
        env, e = self.seed()
        service = e.native_mooncake_service
        consumer = e.workers[1].connector
        r = request(1, 4096)
        manager = e.workers[1].scheduler.block_manager
        self.assertEqual(consumer.probe(r, manager.plan_reuse(r))[0], 3840)
        # Put an unrelated object occupying the whole pool to evict all
        # unpinned objects after a previous consumer lookup.
        p = service.store.reserve_write(("other",), service.store.capacity_bytes)
        service.store.finish_write(p)
        self.assertEqual(consumer.probe(r, manager.plan_reuse(r)), (0, ()))
        self.assertTrue(service.store.validate())

    def test_budget_uses_shared_payload_bytes_and_simulations_are_isolated(self):
        env, e = self.seed()
        service = e.native_mooncake_service
        self.assertEqual(service.store.capacity_bytes, EXTRA["global_segment_size"] - 8 * EXTRA["local_buffer_size"])
        self.assertIsNone(service.store.allocation_unit)
        self.assertEqual(service.store.used_bytes, sum(p.bytes for p in service.store.pages.values()))
        second = self.make_engine(simpy.Environment())
        self.assertIsNot(service.store, second.native_mooncake_service.store)
        self.assertFalse(second.native_mooncake_service.store.pages)
        with self.assertRaises(ConfigurationError):
            self.make_engine(simpy.Environment(), host_available_bytes=100)
        with self.assertRaises(ConfigurationError):
            self.make_engine(simpy.Environment(), enable_offload=True)

    def test_reset_profile_preserves_service_and_has_no_duplicate_shared_counters(self):
        env, e = self.seed()
        service = e.native_mooncake_service
        used = service.store.used_bytes
        e.reset_profile_stats()
        self.assertEqual(service.store.used_bytes, used)
        self.assertIs(e.workers[0].connector.service, service)
        r = request(1, 4096)
        self.submit(env, e, r, 1)
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        from util.results import get_mooncake_stats
        self.assertEqual(get_mooncake_stats(e)["mooncake_memory_hit_tokens"], 3840)
        self.assert_clean(e)


if __name__ == "__main__":
    unittest.main()
