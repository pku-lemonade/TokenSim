"""Af replica kv."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from TokenSim.af.runtime.topology import AFCacheConfig
from TokenSim.block.block import Device
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.config.config import KVTransferConfig
from TokenSim.errors import OutOfBlocksError, SimulationStateError
from TokenSim.kv_transfer import P2PConnector
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from tests.af.support.parallel import _cluster
from tests.af.support.runtime import _request, _runtime_plan


class ReplicaKVBlockManagerTest(unittest.TestCase):
    def _manager(self, gpu=(8, 8), cpu=(8, 8)):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        return ReplicaKVBlockManager(
            block_size=4,
            model="TestModel",
            replica=replica,
            gpu_blocks_by_rank=dict(zip(replica.rank_keys, gpu)),
            cpu_blocks_by_rank=dict(zip(replica.rank_keys, cpu)),
        )

    def test_allocate_append_and_free_map_one_shard_per_rank(self):
        manager = self._manager()
        req = _request(1)
        manager.allocate(req)

        mappings = manager.mappings_for_request(req.id)
        self.assertEqual(len(mappings), 2)
        self.assertTrue(all(len(item.shards) == 2 for item in mappings))
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )

        req.generation_idx = 1
        req._append_tokens(1)
        manager.append_slot(req)
        self.assertEqual(len(manager.mappings_for_request(req.id)), 3)
        manager.free(req)
        self.assertEqual(manager.mappings_for_request(req.id), ())
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_partial_rank_allocation_rolls_back_every_rank_and_request(self):
        manager = self._manager()
        req = _request(2)
        failing_rank = manager.rank_keys[1]

        def fail(_request):
            raise OutOfBlocksError("injected follower exhaustion")

        manager.rank_managers[failing_rank].allocate = fail
        with self.assertRaisesRegex(OutOfBlocksError, "injected"):
            manager.allocate(req)

        self.assertEqual(req.num_physical_token_blocks, 0)
        self.assertEqual(manager.mappings_for_request(req.id), ())
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_fork_reference_counts_and_last_free_are_group_consistent(self):
        manager = self._manager()
        parent = _request(3, prefill_len=4)
        child = _request(4, prefill_len=4)
        manager.allocate(parent)
        manager.fork(parent, child)

        for rank_key in manager.rank_keys:
            block = manager.rank_managers[rank_key].block_table.get_blocks(parent.id)[0]
            self.assertEqual(block.ref_count, 2)
        manager.release_request_blocks(parent)
        for rank_key in manager.rank_keys:
            block = manager.rank_managers[rank_key].block_table.get_blocks(child.id)[0]
            self.assertEqual(block.ref_count, 1)
        manager.release_request_blocks(child)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_swap_and_reservation_update_all_rank_pools(self):
        manager = self._manager()
        req = _request(5)
        manager.allocate(req)
        manager.swap_out(req)
        self.assertTrue(
            all(
                item.device == Device.CPU
                for item in manager.mappings_for_request(req.id)
            )
        )
        self.assertEqual(
            [manager.rank_status(key)["cpu_used"] for key in manager.rank_keys], [2, 2]
        )
        manager.swap_in(req)
        self.assertTrue(
            all(
                item.device == Device.GPU
                for item in manager.mappings_for_request(req.id)
            )
        )

        manager.reserve_gpu_blocks(1)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [3, 3]
        )
        manager.release_reserved_blocks(1)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )

    def test_partial_swap_failure_restores_every_rank_and_mapping(self):
        manager = self._manager()
        req = _request(13)
        manager.allocate(req)
        failing_rank = manager.rank_keys[1]

        def fail(_request):
            raise OutOfBlocksError("injected swap exhaustion")

        manager.rank_managers[failing_rank].swap_out = fail
        with self.assertRaisesRegex(OutOfBlocksError, "injected swap"):
            manager.swap_out(req)

        self.assertTrue(
            all(
                mapping.device == Device.GPU
                for mapping in manager.mappings_for_request(req.id)
            )
        )
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )
        self.assertEqual(
            [manager.rank_status(key)["cpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_logical_capacity_uses_smallest_rank(self):
        manager = self._manager(gpu=(100, 80), cpu=(40, 30))
        self.assertEqual(manager.num_total_gpu_blocks, 80)
        self.assertEqual(manager.num_total_cpu_blocks, 30)
        self.assertEqual(manager.get_gpu_status(), (80, 0, 80))

    def test_dynamic_invariants_report_missing_shard_and_owner_conflict(self):
        manager = self._manager()
        req = _request(6)
        manager.allocate(req)
        manager.logical_mappings.pop((req.id, 0))
        with self.assertRaisesRegex(SimulationStateError, "orphan KV shards"):
            manager.validate_invariants(req)

        manager._rebuild_request_mappings(req.id)
        req.af_a_owners[manager.domain] = "another-replica"
        with self.assertRaisesRegex(SimulationStateError, "owner conflict"):
            manager.validate_invariants(req)

    def test_scheduler_preemption_releases_all_rank_shards(self):
        manager = self._manager(gpu=(2, 2))
        cache = SimpleNamespace(
            block_size=4,
            num_gpu_blocks=2,
            num_cpu_blocks=8,
            model="TestModel",
        )
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=cache,
            max_parallem_sum=8,
            max_occupy_ratio=1,
            block_manager=manager,
        )
        req0 = _request(7, prefill_len=4, decode_len=4)
        req1 = _request(8, prefill_len=4, decode_len=4)
        scheduler.add_requests([req0, req1])
        scheduler.schedule_output()
        for req in (req0, req1):
            req.generation_idx = 4
            req._append_tokens(4)
        output = scheduler.schedule_output()
        self.assertTrue(output.preempted)
        victim = output.preempted[0]
        self.assertEqual(manager.mappings_for_request(victim.id), ())

    def test_prefix_reuse_keeps_logical_hits_and_rank_refcounts_aligned(self):
        manager = self._manager()
        original = Request(
            10,
            8,
            1,
            block_size=4,
            hash_ids=["a", "b"],
        )
        original.arrive(SimpleNamespace(now=0.0))
        manager.allocate(original)
        manager.commit_input_cache(original)
        manager.release_request_blocks(original)

        reused = Request(
            11,
            8,
            1,
            block_size=4,
            hash_ids=["a", "b"],
        )
        reused.arrive(SimpleNamespace(now=1.0))
        manager.allocate(reused)

        self.assertEqual(reused.reuse_hit_blocks, 2)
        self.assertEqual(len(manager.mappings_for_request(reused.id)), 2)
        for rank_key in manager.rank_keys:
            blocks = manager.rank_managers[rank_key].block_table.get_blocks(reused.id)
            self.assertTrue(all(block.ref_count == 1 for block in blocks))

    def test_connector_counts_logical_blocks_once_and_release_clears_all_shards(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        cache = AFCacheConfig.from_plan(plan, replica)
        connector = P2PConnector(KVTransferConfig(kv_connector="P2PConnector"), cache)
        req = _request(12)
        manager.allocate(req)

        transfer = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=0.1,
            request_block_counts=manager.get_request_block_counts([req]),
        )
        self.assertEqual(transfer.blocks, 2)
        self.assertEqual(transfer.request_block_counts, {req.id: 2})
        self.assertEqual(len(manager.mappings_for_request(req.id)), 2)
        manager.release_request_blocks(req)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys],
            [0, 0],
        )
