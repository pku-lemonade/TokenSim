"""Grouped cache recovery, allocation conservation and cancellation."""

import unittest

from TokenSim.block.hybrid_manager import HybridKVBlockManager
from TokenSim.config.kv_layout import CacheGroup, CacheLayerPage, HybridKVLayout
from TokenSim.llm.llm_request import Request


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
