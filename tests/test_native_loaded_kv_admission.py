"""Loaded KV returns through waiting admission (vLLM scheduler.py 30f5c01d4ebb)."""

import unittest

from TokenSim.block.hybrid_manager import HybridKVBlockManager
from TokenSim.llm.llm_request import RequestStatus
from tests import test_native_simple_cpu as cpu_tests
from tests.test_hybrid_kv_cache import request


class LoadedKVAdmissionTest(unittest.TestCase):
    def ready_with_pressure(self, *, lockstep=True):
        helper = cpu_tests.NativeSimpleCPUTest()
        env, e, w = helper.seeded(lockstep=lockstep)
        s, old = w.scheduler, w.scheduler.block_manager
        manager = HybridKVBlockManager(old.layout, 200, old.model)
        s.block_manager = w.connector.manager = manager
        manager.publish_callbacks.append(w.connector._queue_write)
        w.connector.read_link.bandwidth_bytes_per_s = 10000000
        r = request(1, 4096)
        r.dp_rank = 0
        r.arrive(env)
        e.add_requests_burst([r])
        env.run(until=env.now + 0.0005)
        self.assertIn(r.id, w.connector.loads)
        # Independent DMA ownership consumes capacity during the read. This
        # models late physical pressure without altering this request's data.
        pressure = []
        for _ in range(80):
            block = manager._allocate()
            manager.pin_transfer(block)
            manager._release(block)
            pressure.append(block)
        env.run(until=w.connector.loads[r.id])
        return env, e, w, r, pressure

    def test_query_capacity_failure_keeps_loaded_pages_and_does_not_reload(self):
        env, e, w, r, pressure = self.ready_with_pressure()
        env.run(until=env.now + 0.005)
        e.raise_if_failed()
        s, manager = w.scheduler, w.scheduler.block_manager
        self.assertEqual(r.status, RequestStatus.WAITING)
        self.assertEqual(s.loaded_waiting, [r])
        self.assertNotIn(r, s.running)
        self.assertEqual(r.generation_idx, 0)
        self.assertEqual(r.prefill_admission_count, 1)
        self.assertEqual(w.connector.read_link.jobs, 1)
        self.assertEqual(w.preempted_cnt, 0)
        self.assertGreater(s.stats["loaded_kv_capacity_gate"], 0)
        self.assertTrue(manager.block_table.get_blocks(r.id))
        self.assertTrue(manager.validate_conservation())
        for block in pressure:
            manager.unpin_transfer(block)
        w.connector.notify()
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertEqual(r.generation_idx, 4)
        self.assertEqual(r.prefill_admission_count, 1)
        self.assertEqual(w.connector.read_link.jobs, 1)
        self.assertEqual(w.preempted_cnt, 0)
        self.assertFalse(s.loading)
        self.assertFalse(s.loaded_waiting)
        self.assertEqual(manager.get_num_allocated_blocks(), 1)
        self.assertTrue(manager.validate_conservation())

    def test_cancel_while_query_capacity_blocked_releases_request_without_resurrection(
        self,
    ):
        env, e, w, r, pressure = self.ready_with_pressure()
        env.run(until=env.now + 0.005)
        ack = e.cancel_requests([r])[0]
        env.run(until=ack)
        self.assertFalse(w.scheduler.loaded_waiting)
        self.assertFalse(w.scheduler.block_manager.tables)
        self.assertEqual(r.generation_idx, 0)
        for block in pressure:
            w.scheduler.block_manager.unpin_transfer(block)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertEqual(r.generation_idx, 0)
        self.assertEqual(w.connector.read_link.jobs, 1)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())

    def test_loaded_request_obeys_decode_budget_and_prefill_cadence(self):
        env, e, w, loaded, pressure = self.ready_with_pressure()
        manager, s = w.scheduler.block_manager, w.scheduler
        for block in pressure:
            manager.unpin_transfer(block)
        decode = request(2, 256, scope="decoder")
        manager.allocate(decode)
        manager.ensure_slots(decode, 256)
        decode.num_computed_tokens, decode.generation_idx = 256, 1
        decode.status = RequestStatus.RUNNING
        s.running.append(decode)
        s.prefill_capacity_bound = False
        # No prefill work may enter on a deferred step even though its KV is
        # ready; then the decoder may exhaust a one-token budget on release.
        output = s.schedule_output(throttle_prefills=True)
        self.assertEqual(output.running, [decode])
        self.assertEqual(s.loaded_waiting, [loaded])
        s.max_num_batched_tokens = 1
        output = s.schedule_output(throttle_prefills=False)
        self.assertEqual(output.running, [decode])
        self.assertEqual(s.loaded_waiting, [loaded])
        s.max_num_batched_tokens = 512
        output = s.schedule_output(throttle_prefills=False)
        self.assertEqual(output.running, [decode, loaded])
        self.assertEqual(loaded.scheduled_tokens, 256)
        self.assertFalse(s.loaded_waiting)
        self.assertEqual(loaded.prefill_admission_count, 1)
        self.assertTrue(manager.validate_conservation())

    def test_non_lockstep_capacity_wait_wakes_after_memory_is_released(self):
        env, e, w, r, pressure = self.ready_with_pressure(lockstep=False)
        env.run(until=env.now + 0.005)
        self.assertEqual(w.scheduler.loaded_waiting, [r])
        self.assertEqual(r.generation_idx, 0)
        for block in pressure:
            w.scheduler.block_manager.unpin_transfer(block)
        w.connector.notify()
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertEqual(r.prefill_admission_count, 1)
        self.assertEqual(w.preempted_cnt, 0)
        self.assertTrue(w.scheduler.block_manager.validate_conservation())


if __name__ == "__main__":
    unittest.main()
