import unittest
from TokenSim.block.hybrid_manager import HybridKVBlockManager
from TokenSim.kv_transfer.native_storage import NativePageStore
from tests import test_hybrid_kv_cache as hybrid_tests

request = hybrid_tests.request


class ExternalImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hybrid_tests.NativeKVPoolTest.setUpClass()
        cls.model, cls.layout = (
            hybrid_tests.NativeKVPoolTest.model,
            hybrid_tests.NativeKVPoolTest.layout,
        )

    def cached_store(self):
        helper = hybrid_tests.NativeKVPoolTest()
        manager = helper.manager()
        req = request(0, 4096)
        manager.allocate(req)
        helper.compute(manager, req, 4096)
        store = NativePageStore(
            100000000, allocation_unit=self.layout.physical_slot_bytes
        )
        for key, block in manager.cache.items():
            page = store.reserve_write(key, self.layout.physical_slot_bytes)
            store.finish_write(page)
        return store

    def test_complete_group_import_reserves_private_destinations_before_publication(
        self,
    ):
        store = self.cached_store()
        manager = HybridKVBlockManager(self.layout, 1000, self.model.model_id)
        req = request(1, 4096)
        boundary, hits = manager.plan_external_reuse(req, store)
        self.assertEqual(boundary, 3840)
        manager.allocate(req)
        for _, pairs in hits:
            for _, page in pairs:
                store.acquire(page)
        targets = manager.allocate_external_destinations(req, hits)
        self.assertTrue(targets)
        self.assertFalse(any(block.cached for _, _, _, block in targets))
        manager.finish_external_import(req, targets, boundary)
        for _, _, page, _ in targets:
            store.release(page)
        self.assertTrue(all(block.cached for _, _, _, block in targets))
        manager.release_request_blocks(req)
        manager.validate_conservation()
        store.validate()

    def test_cpu_suffix_can_match_when_early_full_pages_only_exist_on_gpu(self):
        store = self.cached_store()
        manager = HybridKVBlockManager(self.layout, 1000, self.model.model_id)
        req = request(1, 4096)
        keys = manager._input_keys(req)
        full = self.layout.full_group
        for i in range(4):
            key = (full.group_id, keys[i], 0)
            page = store.pages.pop(key)
            store.lru.pop(key)
            store.used_bytes -= page.bytes
        boundary, hits = manager.plan_external_reuse(req, store, local_tokens=1024)
        self.assertEqual(boundary, 3840)
        full_pairs = dict(hits)[full.group_id]
        self.assertEqual([i for i, _ in full_pairs], list(range(4, 15)))

    def test_missing_window_state_prevents_full_prefix_import(self):
        store = self.cached_store()
        manager = HybridKVBlockManager(self.layout, 1000, self.model.model_id)
        for key in list(store.pages):
            if key[0] == self.layout.groups[0].group_id:
                page = store.pages.pop(key)
                store.lru.pop(key)
                store.used_bytes -= page.bytes
        self.assertEqual(manager.plan_external_reuse(request(1, 4096), store)[0], 0)


if __name__ == "__main__":
    unittest.main()
