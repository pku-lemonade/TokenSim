"""Grouped cache recovery, allocation conservation and cancellation."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.block.hybrid_manager import HybridKVBlockManager
from TokenSim.config.kv_layout import CacheGroup, CacheLayerPage, HybridKVLayout
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler


def layout(eagle=False):
    return HybridKVLayout(
        groups=(
            CacheGroup(0, (CacheLayerPage("full", "full", 16, 8, alignment=1),)),
            CacheGroup(
                1,
                (
                    CacheLayerPage(
                        "window", "window", 4, 16, window_tokens=8, alignment=1
                    ),
                ),
                eagle=eagle,
            ),
        ),
        logical_block_tokens=16,
        physical_slot_bytes=128,
        retention_interval=32,
        speculative_tokens=1 if eagle else 0,
        max_in_flight_tokens=8,
        max_model_len=256,
    )


class GroupedCacheTest(unittest.TestCase):
    def test_same_public_id_in_separate_engines_cannot_share_unknown_tokens(self):
        managers = [HybridKVBlockManager(layout(), 20, "any-model") for _ in range(2)]
        requests = [Request(0, 64, 2, 16, hash_ids=[1, 2]) for _ in managers]
        keys = [manager._input_keys(req) for manager, req in zip(managers, requests)]
        self.assertEqual(keys[0][:2], keys[1][:2])
        self.assertNotEqual(keys[0][2:], keys[1][2:])
        self.assertEqual(managers[0]._input_keys(requests[0]), keys[0])

    def test_recovery_requires_all_groups_and_leaves_a_writable_tail(self):
        manager = HybridKVBlockManager(layout(eagle=True), 20, "any-model")
        first = Request(1, 64, 2, 16, hash_ids=[101, 202, 303, 404])
        manager.allocate(first)
        for _ in range(8):
            self.assertTrue(manager.can_ensure_slots(first, 8))
            manager.ensure_slots(first, 8)
            first.num_computed_tokens += 8
            manager.cache_computed(first)
            manager.validate_conservation()
        manager.free(first)
        second = Request(2, 64, 2, 16, hash_ids=[101, 202, 303, 404])
        plan = manager.plan_reuse(second)
        self.assertGreater(plan.hit_tokens, 0)
        self.assertLess(plan.hit_tokens, second.prefill_len)
        manager.allocate(second)
        second.start_context_build(plan.hit_tokens)
        manager.ensure_slots(second, 8)
        manager.validate_conservation()
        manager.release_request_blocks(second)
        self.assertEqual(manager.get_num_allocated_blocks(), 1)
        self.assertTrue(all(block.ref_count == 0 for block in manager.blocks[1:]))

    def test_cancellation_cannot_publish_uncomputed_pages(self):
        manager = HybridKVBlockManager(layout(), 12, "any-model")
        request = Request(1, 32, 2, 16, hash_ids=[101, 202])
        manager.allocate(request)
        manager.ensure_slots(request, 8)
        manager.release_request_blocks(request)
        manager.validate_conservation()
        self.assertEqual(manager.cache, {})
        self.assertEqual(manager.get_num_allocated_blocks(), 1)

    def test_failed_first_chunk_allocation_rolls_back_the_whole_admission(self):
        cache = SimpleNamespace(
            hybrid_layout=layout(), num_gpu_blocks=40, model="any-model", block_size=16
        )
        scheduler = LLMPagedAttnScheduler(0, cache, max_num_seqs=4)
        previous = Request(1, 64, 2, 16, hash_ids=[101, 202, 303, 404])
        manager = scheduler.block_manager
        manager.allocate(previous)
        manager.ensure_slots(previous, 64)
        previous.num_computed_tokens = 64
        manager.free(previous)
        request = Request(2, 64, 2, 16, hash_ids=[101, 202, 303, 404])
        scheduler.add_requests([request])
        before = request.admission_state()
        with patch.object(manager, "can_ensure_slots", return_value=False):
            running, preempted = scheduler.schedule()
        # vLLM stops admitting instead of preempting a request that never ran.
        self.assertEqual((running, preempted), ([], []))
        self.assertEqual(request.admission_state(), before)
        self.assertEqual(request.status, RequestStatus.WAITING)
        self.assertIs(scheduler.waiting[0], request)
        self.assertEqual(manager.get_num_allocated_blocks(), 1)
        manager.validate_conservation()
        running, _ = scheduler.schedule()
        self.assertEqual(running, [request])
        self.assertEqual(request.prefill_admission_count, 1)
        self.assertGreater(request.first_prefill_source.gpu_hit_tokens, 0)
