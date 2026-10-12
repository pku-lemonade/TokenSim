import unittest

from TokenSim.block.block_manager import BlockManager
from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.llm.llm_request import Request


BLOCK_SIZE = 16


def make_request(
    request_id: int,
    prefill_len: int,
    decode_len: int = 1,
    hash_ids: list[str | int] | None = None,
    output_hash_ids: list[str | int] | None = None,
    cache_salt: str | None = None,
    reuse_group: str | None = "tenant-a",
) -> Request:
    return Request(
        id=request_id,
        prefill_len=prefill_len,
        decode_len=decode_len,
        block_size=BLOCK_SIZE,
        hash_ids=hash_ids,
        output_hash_ids=output_hash_ids,
        cache_salt=cache_salt,
        reuse_group=reuse_group,
    )


def allocate_and_commit(manager: BlockManager, req: Request) -> list[int]:
    manager.allocate(req)
    manager.commit_input_cache(req)
    return [block.block_number for block in manager.block_table.get_blocks(req.id)]


def block_ids(manager: BlockManager, req: Request) -> list[int]:
    return [block.block_number for block in manager.block_table.get_blocks(req.id)]


class PrefixCacheKeyTest(unittest.TestCase):
    def test_parent_hash_scopes_later_blocks(self):
        first_path = build_prefix_keys(["a", "b"], model="m")
        second_path = build_prefix_keys(["x", "b"], model="m")

        self.assertNotEqual(first_path[1], second_path[1])
        self.assertEqual(first_path[1].block_signature, second_path[1].block_signature)


class AdaptedVllmPrefixCacheTest(unittest.TestCase):
    def test_prefill_full_blocks_cached_partial_block_private(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=8,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req = make_request(0, prefill_len=55, hash_ids=["a", "b", "c"])

        block_ids = allocate_and_commit(manager, req)
        blocks = manager.block_table.get_blocks(req.id)

        self.assertEqual(len(block_ids), 4)
        self.assertEqual(req.reuse_hit_blocks, 0)
        self.assertEqual(req.reuse_miss_blocks, 3)
        self.assertEqual(req.effective_prefill_tokens, 55)
        for block in blocks[:3]:
            self.assertTrue(block.cached)
            self.assertTrue(block.is_full)
            self.assertIsNotNone(block.block_hash)
            self.assertEqual(block.ref_count, 1)
        self.assertFalse(blocks[3].cached)
        self.assertFalse(blocks[3].is_full)
        self.assertIsNone(blocks[3].block_hash)

    def test_longest_contiguous_prefix_hit_stops_after_first_miss(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=12,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req0 = make_request(0, prefill_len=48, hash_ids=["a", "b", "c"])
        ids0 = allocate_and_commit(manager, req0)
        self.assertEqual(ids0, [0, 1, 2])

        req1 = make_request(1, prefill_len=48, hash_ids=["a", "b", "x"])
        manager.allocate(req1)
        ids1 = [block.block_number for block in manager.block_table.get_blocks(req1.id)]
        self.assertEqual(ids1[:2], ids0[:2])
        self.assertNotEqual(ids1[2], ids0[2])
        self.assertEqual(req1.reuse_hit_blocks, 2)
        self.assertEqual(req1.reuse_miss_blocks, 1)
        self.assertEqual(req1.cached_prefill_tokens, 32)
        self.assertEqual(req1.effective_prefill_tokens, 16)

        req2 = make_request(2, prefill_len=48, hash_ids=["a", "z", "c"])
        manager.allocate(req2)
        ids2 = [block.block_number for block in manager.block_table.get_blocks(req2.id)]
        self.assertEqual(ids2[0], ids0[0])
        self.assertNotEqual(ids2[1], ids0[1])
        self.assertNotEqual(ids2[2], ids0[2])
        self.assertEqual(req2.reuse_hit_blocks, 1)
        self.assertEqual(req2.reuse_miss_blocks, 2)
        self.assertEqual(req2.cached_prefill_tokens, 16)
        self.assertEqual(req2.effective_prefill_tokens, 32)

    def test_cache_hit_while_original_request_still_running_increments_ref_count(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=8,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req0 = make_request(0, prefill_len=32, hash_ids=["a", "b"])
        allocate_and_commit(manager, req0)

        req1 = make_request(1, prefill_len=37, hash_ids=["a", "b"])
        manager.allocate(req1)
        shared_blocks = manager.block_table.get_blocks(req1.id)[:2]

        self.assertEqual(req1.reuse_hit_blocks, 2)
        self.assertEqual(req1.effective_prefill_tokens, 5)
        self.assertTrue(all(block.ref_count == 2 for block in shared_blocks))

        manager.free(req1)
        self.assertTrue(all(block.ref_count == 1 for block in shared_blocks))
        manager.free(req0)
        self.assertTrue(all(block.ref_count == 0 for block in shared_blocks))
        self.assertTrue(all(block.cached for block in shared_blocks))

    def test_zero_ref_cached_block_removed_from_free_queue_on_hit(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=8,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req0 = make_request(0, prefill_len=32, hash_ids=["a", "b"])
        allocate_and_commit(manager, req0)
        cached_blocks = list(manager.block_table.get_blocks(req0.id))
        manager.free(req0)

        self.assertTrue(
            all(block in manager.gpu_allocator.free_blocks for block in cached_blocks)
        )
        req1 = make_request(1, prefill_len=32, hash_ids=["a", "b"])
        manager.allocate(req1)

        self.assertEqual(req1.reuse_hit_blocks, 2)
        self.assertTrue(all(block.ref_count == 1 for block in cached_blocks))
        self.assertTrue(
            all(
                block not in manager.gpu_allocator.free_blocks
                for block in cached_blocks
            )
        )

    def test_different_salt_group_or_model_prevents_hit(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=12,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req0 = make_request(
            0,
            prefill_len=32,
            hash_ids=["a", "b"],
            cache_salt="salt-a",
            reuse_group="tenant-a",
        )
        allocate_and_commit(manager, req0)

        different_salt = make_request(
            1,
            prefill_len=32,
            hash_ids=["a", "b"],
            cache_salt="salt-b",
            reuse_group="tenant-a",
        )
        manager.allocate(different_salt)
        self.assertEqual(different_salt.reuse_hit_blocks, 0)

        different_group = make_request(
            2,
            prefill_len=32,
            hash_ids=["a", "b"],
            cache_salt="salt-a",
            reuse_group="tenant-b",
        )
        manager.allocate(different_group)
        self.assertEqual(different_group.reuse_hit_blocks, 0)
        self.assertNotEqual(req0.input_cache_keys, different_group.input_cache_keys)

        other_model = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=8,
            num_cpu_blocks=8,
            model="qwen",
            watermark=0,
        )
        other_model_req = make_request(
            3,
            prefill_len=32,
            hash_ids=["a", "b"],
            cache_salt="salt-a",
            reuse_group="tenant-a",
        )
        other_model.allocate(other_model_req)
        self.assertEqual(other_model_req.reuse_hit_blocks, 0)
        self.assertNotEqual(req0.input_cache_keys, other_model_req.input_cache_keys)

    def test_touched_blocks_survive_eviction(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=4,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req_a = make_request(0, prefill_len=32, hash_ids=["a0", "a1"])
        req_b = make_request(1, prefill_len=32, hash_ids=["b0", "b1"])
        allocate_and_commit(manager, req_a)
        allocate_and_commit(manager, req_b)
        blocks_a = list(manager.block_table.get_blocks(req_a.id))
        blocks_b = list(manager.block_table.get_blocks(req_b.id))
        manager.free(req_a)
        manager.free(req_b)

        # Touch A by reusing it, then free it again so B becomes LRU.
        req_a_hit = make_request(2, prefill_len=32, hash_ids=["a0", "a1"])
        manager.allocate(req_a_hit)
        manager.free(req_a_hit)

        req_c = make_request(3, prefill_len=32, hash_ids=["c0", "c1"])
        manager.allocate(req_c)

        self.assertTrue(all(block.cached for block in blocks_a))
        self.assertTrue(any(not block.cached for block in blocks_b))
        old_prefix = make_request(4, prefill_len=32, hash_ids=["b0", "b1"])
        self.assertEqual(
            manager.kv_cache_manager.plan_reuse(old_prefix).hit_block_count, 0
        )

    def test_output_hash_ids_register_full_decode_blocks(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=8,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req0 = make_request(
            0,
            prefill_len=16,
            decode_len=16,
            hash_ids=["p"],
            output_hash_ids=["o"],
        )
        allocate_and_commit(manager, req0)
        req0.generation_idx = 15
        req0._append_tokens(15)
        manager.append_slot(req0)
        req0.generation_idx = 16
        req0._append_tokens(1)
        manager.free(req0)

        req1 = make_request(1, prefill_len=32, hash_ids=["p", "o"])
        manager.allocate(req1)
        self.assertEqual(req1.reuse_hit_blocks, 2)
        self.assertEqual(req1.effective_prefill_tokens, 0)


class StrongTreePrefixCacheTest(unittest.TestCase):
    def test_deep_prefix_tree_reuses_distinct_branch_depths(self):
        root = [f"root-{i}" for i in range(128)]
        left = [f"left-{i}" for i in range(48)]
        right = [f"right-{i}" for i in range(48)]
        left_leaf_a = [f"left-a-{i}" for i in range(32)]
        left_leaf_b = [f"left-b-{i}" for i in range(32)]
        right_leaf = [f"right-a-{i}" for i in range(32)]
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=512,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )

        req_left_a = make_request(
            0,
            prefill_len=(len(root) + len(left) + len(left_leaf_a)) * BLOCK_SIZE,
            hash_ids=root + left + left_leaf_a,
        )
        ids_left_a = allocate_and_commit(manager, req_left_a)

        req_right = make_request(
            1,
            prefill_len=(len(root) + len(right) + len(right_leaf)) * BLOCK_SIZE,
            hash_ids=root + right + right_leaf,
        )
        manager.allocate(req_right)
        ids_right = block_ids(manager, req_right)
        self.assertEqual(req_right.reuse_hit_blocks, len(root))
        self.assertEqual(ids_right[: len(root)], ids_left_a[: len(root)])
        self.assertNotEqual(ids_right[len(root)], ids_left_a[len(root)])
        manager.commit_input_cache(req_right)

        req_left_b = make_request(
            2,
            prefill_len=(len(root) + len(left) + len(left_leaf_b)) * BLOCK_SIZE,
            hash_ids=root + left + left_leaf_b,
        )
        manager.allocate(req_left_b)
        ids_left_b = block_ids(manager, req_left_b)
        expected_left_hit = len(root) + len(left)
        self.assertEqual(req_left_b.reuse_hit_blocks, expected_left_hit)
        self.assertEqual(ids_left_b[:expected_left_hit], ids_left_a[:expected_left_hit])
        self.assertNotEqual(
            ids_left_b[expected_left_hit], ids_left_a[expected_left_hit]
        )
        shared_root_blocks = manager.block_table.get_blocks(req_left_a.id)[: len(root)]
        self.assertTrue(all(block.ref_count == 3 for block in shared_root_blocks))

        req_right_again = make_request(
            3,
            prefill_len=(len(root) + len(right) + len(right_leaf)) * BLOCK_SIZE,
            hash_ids=root + right + right_leaf,
        )
        manager.allocate(req_right_again)
        expected_right_hit = len(root) + len(right) + len(right_leaf)
        self.assertEqual(req_right_again.reuse_hit_blocks, expected_right_hit)
        self.assertEqual(block_ids(manager, req_right_again), ids_right)

    def test_long_tree_no_hash_path_keeps_baseline_metrics(self):
        manager = BlockManager(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=400,
            num_cpu_blocks=8,
            model="llama",
            watermark=0,
        )
        req = make_request(0, prefill_len=320 * BLOCK_SIZE, hash_ids=None)
        manager.allocate(req)

        self.assertEqual(req.reuse_hit_blocks, 0)
        self.assertEqual(req.reuse_miss_blocks, 0)
        self.assertEqual(req.cached_prefill_tokens, 0)
        self.assertEqual(req.effective_prefill_tokens, 320 * BLOCK_SIZE)
        self.assertEqual(len(manager.block_table.get_blocks(req.id)), 320)


if __name__ == "__main__":
    unittest.main()
