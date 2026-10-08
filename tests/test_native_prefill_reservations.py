"""Expectations from scheduler.py / single_type_kv_cache_manager.py 30f5c01d4ebb."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.block.hybrid_manager import HybridKVBlockManager
from tests import test_native_simple_cpu as cpu_tests
from tests.test_hybrid_kv_cache import request


class NativePrefillReservationTest(unittest.TestCase):
    def scheduler(self, *, mtp_tokens=None):
        env = simpy.Environment()
        e = cpu_tests.engine(env, mtp_num_speculative_tokens=mtp_tokens)
        scheduler = e.workers[0].scheduler
        # These tests isolate scheduler reservations. DMA/publication is
        # exercised below and by the native SimpleCPU engine suite.
        scheduler.block_manager.publish_callbacks.clear()
        return env, e, scheduler

    def test_partial_gpu_prefill_reservation_uses_virtual_window_lengths(self):
        env, e, s = self.scheduler(mtp_tokens=3)
        r = request(0, 4096)
        s.add_requests([r])
        output = s.schedule_output()
        self.assertEqual(output.running, [r])
        self.assertEqual(r.scheduled_tokens, 512)
        # With max_in_flight_tokens=1024 the window admission caps are:
        # 19, 19, 145, 259. Full pages need16. Explicit K3 makes the
        # physical end515: virtual ends9,9,65,129 and full3. Remaining
        # = (19-9)*2 + (145-65) + (259-129) + (16-3) =243, plus spec1.
        manager = s.block_manager
        self.assertTrue(s.mtp_enabled)
        self.assertEqual(manager.remaining_admission_slots(r), 243)
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 244)
        r.num_computed_tokens = 512
        manager.ensure_slots(r, 512, lookahead=3)
        # Next physical end1027 gives virtual ends17,17,129,257/full5.
        self.assertEqual(manager.remaining_admission_slots(r), 33)
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 34)
        # Removed old window positions are null entries in the source;
        # physical window length must not replace their absolute table end.
        self.assertLess(len(manager.tables[r.id][4]), 256)
        self.assertTrue(manager.validate_conservation())

    def test_no_future_reservation_for_new_last_chunk_and_decode_releases_old_one(self):
        env, e, s = self.scheduler()
        r = request(0, 512)
        s.add_requests([r])
        s.schedule_output()
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 0)
        r.num_computed_tokens = 512
        r.generation_idx = 1
        s.update([r])
        other = request(1, 4096, scope="other")
        s.add_requests([other])
        s.schedule_output()
        self.assertIn(other.id, s.inflight_prefills)
        s.block_manager.ensure_slots(
            other, other.context_len - other.num_computed_tokens
        )
        other.num_computed_tokens = 4096
        other.generation_idx = 1
        s.update([other])
        self.assertNotIn(other.id, s.inflight_prefills)

    def test_preemption_and_cancellation_drop_reservation_with_request_ownership(self):
        for operation in ("preempt", "cancel"):
            env, e, s = self.scheduler()
            r = request(0, 4096)
            s.add_requests([r])
            s.schedule_output()
            self.assertGreater(s._inflight_prefill_reserved_blocks(), 0)
            if operation == "preempt":
                s._preempt(r)
                self.assertIn(r, s.waiting)
            else:
                self.assertTrue(s.cancel_request(r))
            self.assertFalse(s.inflight_prefills)
            self.assertEqual(s._inflight_prefill_reserved_blocks(), 0)
            self.assertTrue(s.block_manager.validate_conservation())

    def test_async_import_cannot_consume_existing_prefill_commitment(self):
        helper = cpu_tests.NativeSimpleCPUTest()
        env, e, worker = helper.seeded(mtp_num_speculative_tokens=3)
        s = worker.scheduler
        old = s.block_manager
        manager = HybridKVBlockManager(old.layout, 460, old.model)
        s.block_manager = worker.connector.manager = manager
        manager.publish_callbacks.append(worker.connector._queue_write)
        # This miss passes full-input admission (458 slots). Explicit K3
        # allocates215 slots for the first chunk, leaving243 + spec1.
        miss = request(10, 4096, scope="new-tree")
        s.add_requests([miss])
        first = s.schedule_output()
        self.assertEqual(first.running, [miss])
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 244)
        hit = request(11, 4096)
        local = manager.plan_reuse(hit)
        boundary, pairs = worker.connector.probe(hit, local)
        self.assertEqual(boundary, 3840)
        self.assertTrue(manager.can_allocate_external(hit, local, boundary, pairs))
        self.assertFalse(s._can_admit_external(hit, local, (boundary, pairs)))
        self.assertEqual(s.stats["async_load_reservation_gate"], 1)
        # Releasing the older promise makes this same import admissible.
        s.cancel_request(miss)
        self.assertTrue(s._can_admit_external(hit, local, (boundary, pairs)))
        self.assertTrue(manager.validate_conservation())

    def test_only_initial_import_uses_other_reservations_not_its_full_input_envelope(
        self,
    ):
        env = simpy.Environment()
        e = cpu_tests.engine(env)
        producer = request(0, 512)
        producer.dp_rank = 0
        producer.arrive(env)
        e.add_requests_burst([producer])
        env.run(until=producer.completion_event)
        env.run(until=env.now + 0.01)
        worker = e.workers[0]
        old = worker.scheduler.block_manager
        manager = HybridKVBlockManager(old.layout, 460, old.model)
        r = request(1, 4096)
        local = manager.plan_reuse(r)
        boundary, hits = manager.plan_external_reuse(r, worker.connector.store)
        self.assertEqual(boundary, 256)
        # Enough for the 23 private DMA pages while leaving 430 slots, even
        # when its full-input gate plus that promise would exceed the pool.
        self.assertTrue(
            manager.can_allocate_external(r, local, boundary, hits, reserved_blocks=430)
        )
        self.assertFalse(
            manager.can_allocate_external(r, local, boundary, hits, reserved_blocks=440)
        )
        no_spec = replace(manager.layout, speculative_tokens=0)
        s = worker.scheduler
        s.block_manager = HybridKVBlockManager(no_spec, 460, old.model)
        self.assertEqual(s._spec_decode_step_blocks(), 0)

    def test_engine_defers_load_until_prefill_promise_shrinks_without_preemption(self):
        helper = cpu_tests.NativeSimpleCPUTest()
        env, e, worker = helper.seeded()
        s, old = worker.scheduler, worker.scheduler.block_manager
        manager = HybridKVBlockManager(old.layout, 460, old.model)
        s.block_manager = worker.connector.manager = manager
        manager.publish_callbacks.append(worker.connector._queue_write)
        s.long_prefill_token_threshold = 256
        worker.connector.read_link.bandwidth_bytes_per_s = 10000000
        miss = request(10, 4096, scope="new-tree")
        hit = request(11, 4096)
        for r in (miss, hit):
            r.dp_rank = 0
            r.arrive(env)
        e.add_requests_burst([miss, hit])
        env.run(until=env.now + 0.0005)
        self.assertEqual(s.waiting, [hit])
        self.assertFalse(worker.connector.loads)
        self.assertGreater(s.stats["async_load_reservation_gate"], 0)
        env.run(until=hit.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertTrue(miss.is_done and hit.is_done)
        self.assertEqual(worker.preempted_cnt, 0)
        self.assertFalse(s.inflight_prefills)
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 0)
        self.assertTrue(manager.validate_conservation())

    def test_loading_reservation_uses_reserved_destinations_and_cancellation_releases_it(
        self,
    ):
        helper = cpu_tests.NativeSimpleCPUTest()
        env, e, worker = helper.seeded(mtp_num_speculative_tokens=3)
        worker.connector.read_link.bandwidth_bytes_per_s = 10000000
        r = request(1, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=env.now + 0.0005)
        s = worker.scheduler
        self.assertEqual(r.num_computed_tokens, 0)
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 2)
        self.assertIn(r.id, s.loading)
        ack = e.cancel_requests([r])[0]
        env.run(until=ack)
        self.assertEqual(s._inflight_prefill_reserved_blocks(), 0)
        self.assertTrue(s.block_manager.transfer_references)
        self.assertTrue(s.block_manager.validate_conservation())
        env.run(until=env.now + 10)
        self.assertFalse(s.block_manager.transfer_references)


if __name__ == "__main__":
    unittest.main()
