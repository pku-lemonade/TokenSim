from __future__ import annotations

import unittest
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.block.hybrid_manager import HybridKVBlockManager
from TokenSim.block.prefix_cache import RequestLocalPrefixKey
from TokenSim.config.kv_layout import hybrid_kv_layout
from TokenSim.config.model_config import ModelCatalog
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.memory_budget import hybrid_memory_budget
from TokenSim.hardware.context import HardwareContext
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.workload.agentx_native.replay import engine_hashes
from TokenSim.workload.agentx_native.logging import NativeReplayLogs
from util.results import get_cache_capacity_stats
from TokenSim.errors import ConfigurationError


def request(i, tokens, *, scope="tree", raw=None):
    raw = list(range(tokens // 64)) if raw is None else raw
    req = Request(
        i,
        tokens,
        4,
        256,
        cache_salt=scope,
        hash_ids=engine_hashes(
            raw, trace_block_size=64, engine_block_size=256, input_tokens=tokens
        ),
    )
    req.native_trace_hash_ids, req.native_trace_block_size = raw, 64
    return req


class NativeKVLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")

    def test_pages_are_single_latent_compressed_and_aligned(self):
        layout = hybrid_kv_layout(self.model)
        pages = {p.name: p for g in layout.groups for p in g.layers}
        self.assertEqual(pages["layers.2.compressed"].page_bytes, 32768)
        self.assertEqual(pages["layers.0.compressed"].page_bytes, 1024)
        self.assertEqual(pages["layers.2.indexer"].payload_bytes, 4352)
        self.assertEqual(pages["layers.2.indexer"].page_bytes, 4608)
        self.assertEqual(pages["layers.2.indexer_state"].page_bytes, 8192)
        self.assertEqual(pages["layers.2.compressor_state"].page_bytes, 32768)
        self.assertEqual(pages["layers.0.compressor_state"].page_bytes, 32768)
        self.assertEqual(pages["mtp.61.swa"].page_bytes, 32768)
        self.assertNotIn("mtp.61.compressed", pages)

    def test_groups_alias_one_stride_and_partition_all_layer_pages(self):
        layout = hybrid_kv_layout(self.model)
        self.assertEqual(layout.physical_slot_bytes, 1228800)
        self.assertEqual(
            sorted(
                (g.block_tokens, len(g.layers), g.group_bytes) for g in layout.groups
            ),
            [
                (4, 60, 1228800),
                (8, 31, 1015808),
                (64, 31, 1015808),
                (64, 31, 1015808),
                (256, 91, 1153024),
            ],
        )
        self.assertEqual(sum(g.eagle for g in layout.groups), 1)
        self.assertEqual(
            sum(len(g.layers) for g in layout.groups),
            len({p.name for g in layout.groups for p in g.layers}),
        )

    def test_log_equivalent_reconstruction_depends_on_token_budget_and_is_not_capacity_input(
        self,
    ):
        for budget, slots, upper, equivalent in (
            (8192, 28194, 10778, 2742953),
            (16384, 90064, 17434, 5416940),
        ):
            layout = hybrid_kv_layout(self.model, max_num_batched_tokens=budget)
            self.assertEqual(layout.max_request_slots(), upper)
            self.assertEqual(layout.equivalent_tokens(slots), equivalent)
        layout = hybrid_kv_layout(self.model, max_num_batched_tokens=16384)
        self.assertEqual(layout.equivalent_tokens(90062), 5416820)

    def test_hbm_budget_is_calculated_and_every_deduction_is_positive_once(self):
        for sku, budget, seqs, util in (
            ("b200", 8192, 49, 0.9),
            ("b300", 16384, 96, 0.92),
        ):
            parallel = ClusterConfig.from_file(
                f"data/clusters/8_{sku}/dep8.json"
            ).effective_parallel_config()
            cache = CacheConfig(
                256,
                self.hardware.device(sku),
                self.model,
                parallel,
                usable_memory_fraction=util,
                max_num_batched_tokens=budget,
                max_num_seqs=seqs,
            )
            memory = cache.memory_budget
            self.assertEqual(
                memory["hardware_budget_bytes"],
                int(self.hardware.device(sku).memory_capacity_bytes.value * util),
            )
            self.assertTrue(all(value > 0 for value in memory["deductions"].values()))
            self.assertEqual(
                memory["available_kv_bytes"],
                memory["hardware_budget_bytes"] - sum(memory["deductions"].values()),
            )
            self.assertEqual(
                cache.num_gpu_blocks, memory["available_kv_bytes"] // 1228800
            )
            self.assertGreater(
                cache.num_gpu_blocks, cache.hybrid_layout.max_request_slots(996579)
            )
            self.assertEqual(cache.num_cpu_blocks, 0)

    def test_mtp_disabled_removes_draft_group_marker_and_preserves_ordinary_kv(self):
        layout = hybrid_kv_layout(
            self.model, include_mtp=False, speculative_tokens=0
        )
        self.assertFalse(any(g.eagle for g in layout.groups))
        self.assertFalse(any(p.draft for g in layout.groups for p in g.layers))


class NativeKVPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ModelCatalog.load("data/models").get("DeepSeek-V4-Pro")
        cls.layout = hybrid_kv_layout(cls.model)

    def manager(self, slots=20000):
        return HybridKVBlockManager(
            self.layout, slots, self.model.model_id, watermark=0
        )

    def compute(self, manager, req, tokens, chunk=512):
        for size in range(0, tokens, chunk):
            amount = min(chunk, tokens - size)
            manager.ensure_slots(req, amount)
            req.num_computed_tokens += amount
            manager.cache_computed(req)
            manager.validate_conservation()

    def test_admission_checks_full_input_but_physical_allocation_advances_by_chunk(
        self,
    ):
        manager = self.manager()
        req = request(0, 996352)
        self.assertTrue(manager.can_allocate(req))
        manager.allocate(req)
        self.assertEqual(manager.block_table.get_num_blocks(req.id), 0)
        manager.ensure_slots(req, 512)
        # 2 full + 2*8 SWA + 64 HCA state + 128 CSA state slots.
        self.assertEqual(manager.block_table.get_num_blocks(req.id), 210)
        manager.validate_conservation()

    def test_complete_input_admission_uses_free_slots_without_virtual_reservations(
        self,
    ):
        manager = self.manager(15000)
        first = request(0, 996352)
        second = request(1, 996352)
        self.assertTrue(manager.can_allocate(first))
        manager.allocate(first)
        self.assertTrue(manager.can_allocate(second))
        manager.ensure_slots(first, 16384)
        self.assertFalse(manager.can_allocate(second))
        manager.release_request_blocks(first)
        self.assertTrue(manager.can_allocate(second))

    def test_watermark_only_protects_admission_alongside_scheduled_requests(self):
        manager = HybridKVBlockManager(
            self.layout, 250, self.model.model_id, watermark=0.2
        )
        req = request(0, 512)
        self.assertTrue(manager.can_allocate(req))
        self.assertFalse(manager.can_allocate(req, has_scheduled_requests=True))
        manager.allocate(req)
        self.compute(manager, req, 512)
        self.assertTrue(manager.can_ensure_slots(req, 512))
        manager.ensure_slots(req, 512)
        manager.validate_conservation()

    def generated_checkpoint(self):
        manager = self.manager()
        req = request(71, 32256)
        req.decode_len = 2048
        manager.allocate(req)
        req.record_context_admission(0, 0)
        self.compute(manager, req, req.prefill_len)
        req._append_tokens(1025)
        req.generation_idx = 1025
        self.compute(manager, req, 1024)
        manager.release_request_blocks(req)
        req.prepare_recompute()
        return manager, req

    def test_generated_retention_checkpoint_recovers_only_the_same_request(self):
        manager, req = self.generated_checkpoint()
        plan = manager.plan_reuse(req)
        self.assertEqual(plan.hit_tokens, 32768)
        self.assertIsInstance(plan.input_keys[127], RequestLocalPrefixKey)
        self.assertEqual(plan.effective_prefill_tokens, 0)
        other = request(72, 32256)
        other._append_tokens(1025)
        other.generation_idx = 1025
        self.assertLessEqual(manager.plan_reuse(other).hit_tokens, 32000)
        manager.allocate(req)
        req.record_context_admission(plan.hit_tokens, 0)
        req.start_context_build(plan.hit_tokens)
        self.assertEqual(req.remaining_context_tokens, 513)
        self.assertEqual(req.reloaded_generated_gpu_tokens, 512)
        self.assertEqual(req.cached_prefill_tokens, req.prefill_len)
        self.assertTrue(manager.validate_conservation())
        manager.release_request_blocks(req)

    def test_missing_generated_eagle_peek_forces_all_group_recovery_back(self):
        manager, req = self.generated_checkpoint()
        self.assertEqual(manager.plan_reuse(req).hit_tokens, 32768)
        eagle = next(group for group in self.layout.groups if group.eagle)
        key = manager._slot_key(eagle, 32768 // eagle.block_tokens,
                                manager._input_keys(req), req)
        self.assertIsInstance(key[1], RequestLocalPrefixKey)
        manager._evict(manager.cache[key])
        self.assertLessEqual(manager.plan_reuse(req).hit_tokens, 32000)
        self.assertTrue(manager.validate_conservation())

    def test_full_group_context_limit_leaves_last_real_token_and_updates_keys(self):
        # Isolate the full-group n-1 contract; mixed groups may lower the hit.
        layout = replace(self.layout, groups=(self.layout.full_group,))
        manager = HybridKVBlockManager(layout, 32, self.model.model_id)
        req = request(73, 256)
        req.decode_len = 1024
        manager.allocate(req)
        self.compute(manager, req, 256)
        self.assertEqual(len(manager._input_keys(req)), 1)
        req._append_tokens(513)
        req.generation_idx = 513
        self.compute(manager, req, 512)
        keys = manager._input_keys(req)
        self.assertEqual(len(keys), 3)
        self.assertTrue(all(isinstance(key, RequestLocalPrefixKey) for key in keys[1:]))
        manager.release_request_blocks(req)
        req.prepare_recompute()
        plan = manager.plan_reuse(req)
        self.assertEqual(plan.hit_tokens, 768)
        self.assertEqual(plan.reuse_miss_blocks, 0)
        req.start_context_build(plan.hit_tokens)
        self.assertEqual(req.remaining_context_tokens, 1)
        self.assertTrue(manager.validate_conservation())

    def test_context_metrics_preserve_first_input_and_separate_generated_tiers(self):
        req = request(74, 256)
        req.record_context_admission(0, 0)
        first = req.first_prefill_source
        req._append_tokens(513)
        req.generation_idx = 513
        req.record_context_admission(512, 256)
        self.assertIs(req.first_prefill_source, first)
        self.assertEqual(first.computed_tokens, 256)
        self.assertEqual(req.gpu_cached_context_tokens, 512)
        self.assertEqual(req.external_cached_context_tokens, 256)
        self.assertEqual(req.reloaded_gpu_tokens, 256)
        self.assertEqual(req.reloaded_cpu_tokens, 0)
        self.assertEqual(req.reloaded_generated_gpu_tokens, 256)
        self.assertEqual(req.reloaded_generated_cpu_tokens, 256)
        self.assertEqual(req.cached_prefill_tokens, 256)
        self.assertEqual(req.external_cached_prefill_tokens, 0)
        self.assertEqual(req.effective_prefill_tokens, 0)

    def test_all_groups_required_for_prefix_recovery_and_hit_is_at_most_n_minus_one(
        self,
    ):
        manager = self.manager()
        first = request(0, 4096)
        manager.allocate(first)
        self.compute(manager, first, 4096)
        manager.free(first)
        second = request(1, 4096)
        plan = manager.plan_reuse(second)
        self.assertEqual(plan.hit_tokens, 3840)
        self.assertLess(plan.hit_tokens, second.prefill_len)
        window = next(g for g in self.layout.groups if g.window_tokens and not g.eagle)
        for key in list(manager.cache):
            if key[0] == window.group_id:
                del manager.cache[key]
        self.assertEqual(manager.plan_reuse(second).hit_tokens, 0)

    def test_checkpoint_states_are_cached_but_released_from_request_references(self):
        manager = self.manager()
        req = request(0, 66560)
        manager.allocate(req)
        self.compute(manager, req, 66560)
        self.assertTrue(manager.validate_conservation())
        full = self.layout.full_group
        live = sum(len(t) for t in manager.tables[req.id].values())
        # The last 512-token chunk plus the preceding 127/7-token windows.
        self.assertEqual(live - len(manager.tables[req.id][full.group_id]), 230)
        checkpoint = request(1, 33024)
        plan = manager.plan_reuse(checkpoint)
        self.assertEqual(plan.hit_tokens, 32768)

    def test_eagle_peek_is_checked_then_dropped_before_writing(self):
        manager = self.manager()
        first = request(0, 4096)
        manager.allocate(first)
        self.compute(manager, first, 4096)
        eagle = next(g for g in self.layout.groups if g.eagle)
        second = request(1, 4096)
        plan = manager.plan_reuse(second)
        self.assertEqual(plan.hit_tokens, 3840)
        recovered = dict(plan.group_hits)[eagle.group_id]
        self.assertEqual([i for i, _ in recovered], [58, 59])
        peek = manager.cache[manager._slot_key(eagle, 60, plan.input_keys, second)]
        manager.allocate(second)
        second.start_context_build(plan.hit_tokens)
        manager.ensure_slots(second, 256)
        self.assertIsNot(manager.tables[second.id][eagle.group_id][60], peek)
        manager.validate_conservation()

    def test_window_reclaim_precedes_new_group_allocations_in_a_tight_pool(self):
        manager = self.manager(237)
        req = request(0, 4096)
        manager.allocate(req)
        self.compute(manager, req, 512)
        manager.ensure_slots(req, 512)
        req.num_computed_tokens += 512
        manager.cache_computed(req)
        self.assertTrue(manager.can_ensure_slots(req, 512))
        manager.ensure_slots(req, 512)
        manager.validate_conservation()

    def test_prefix_sharing_and_release_order_preserve_reference_counts(self):
        manager = self.manager()
        first = request(0, 4096)
        manager.allocate(first)
        self.compute(manager, first, 4096)
        first.generation_idx = 1
        second = request(1, 4096)
        manager.allocate(second)
        self.assertTrue(
            any(b.ref_count == 2 for b in manager.block_table.get_blocks(first.id))
        )
        manager.release_request_blocks(first)
        manager.validate_conservation()
        manager.release_request_blocks(second)
        manager.validate_conservation()
        self.assertEqual(manager.get_num_allocated_blocks(), 1)

    def test_eviction_and_scope_isolation_do_not_leave_stale_prefix_entries(self):
        manager = self.manager(1000)
        req = request(0, 4096)
        manager.allocate(req)
        self.compute(manager, req, 4096)
        manager.free(req)
        self.assertEqual(
            manager.plan_reuse(request(1, 4096, scope="different")).hit_tokens, 0
        )
        blocks = [manager._allocate() for _ in range(999)]
        self.assertEqual(manager.plan_reuse(request(2, 4096)).hit_tokens, 0)
        for block in blocks:
            manager._release(block)
        self.assertEqual(manager.get_gpu_status(), (999, 1, 1000))

    def test_concurrent_duplicate_cache_survives_primary_copy_eviction(self):
        manager = self.manager()
        first, second = request(0, 4096), request(1, 4096)
        manager.allocate(first)
        manager.allocate(second)
        self.compute(manager, first, 4096)
        self.compute(manager, second, 4096)
        full = self.layout.full_group
        original = manager.tables[first.id][full.group_id][0]
        duplicate = manager.tables[second.id][full.group_id][0]
        key = original.block_hash
        self.assertIs(manager.cache[key], original)
        self.assertTrue(duplicate.cached)
        manager.free(first)
        manager._evict(original)
        self.assertIs(manager.cache[key], duplicate)
        self.assertEqual(manager.plan_reuse(request(2, 4096)).hit_tokens, 3840)
        manager.validate_conservation()

    def test_uncached_window_recycling_preserves_cached_prefix_under_churn(self):
        layout = hybrid_kv_layout(self.model, max_num_batched_tokens=512)
        manager = HybridKVBlockManager(layout, 1000, self.model.model_id)
        first = request(0, 4096)
        self.assertTrue(manager.can_allocate(first))
        manager.allocate(first)
        self.compute(manager, first, 4096)
        manager.free(first)
        other = request(1, 66560, scope="other")
        self.assertTrue(manager.can_allocate(other))
        manager.allocate(other)
        self.compute(manager, other, 66560)
        self.assertGreater(manager.allocation_count, manager.num_total_gpu_blocks * 20)
        self.assertEqual(manager.plan_reuse(request(2, 4096)).hit_tokens, 3840)
        manager.validate_conservation()

    def test_free_prioritizes_uncached_pages_and_evicts_cached_tail_before_prefix(self):
        manager = self.manager(1000)
        req = request(0, 512)
        manager.allocate(req)
        self.compute(manager, req, 512)
        full = manager.layout.full_group
        head, tail = [manager.tables[req.id][full.group_id][i] for i in (0, 1)]
        manager.free(req)
        order = list(manager.free_slots)
        self.assertLess(order.index(tail.block_number), order.index(head.block_number))
        uncached = manager._allocate()
        self.assertFalse(uncached.cached)
        manager._release(uncached)
        self.assertIs(manager._allocate(), uncached)

    def test_completed_full_prefix_is_published_once_as_decode_advances(self):
        manager = self.manager()
        req = request(0, 4096)
        manager.allocate(req)
        self.compute(manager, req, 4096)
        req.generation_idx = 1
        full = self.layout.full_group
        from unittest.mock import patch

        with patch.object(manager, "_publish", wraps=manager._publish) as publish:
            for _ in range(20):
                manager.append_slot(req)
                req.num_computed_tokens += 1
                manager.cache_computed(req)
            self.assertFalse(
                any(
                    call.args[0].group_id == full.group_id
                    for call in publish.call_args_list
                )
            )
        manager.validate_conservation()

    def test_cancellation_does_not_publish_uncomputed_blocks(self):
        manager = self.manager()
        req = request(0, 4096)
        manager.allocate(req)
        manager.ensure_slots(req, 512)
        manager.release_request_blocks(req)
        self.assertEqual(len(manager.cache), 0)
        manager.validate_conservation()

    def test_decode_lookahead_allocates_each_groups_token_width(self):
        manager = self.manager()
        req = request(0, 512)
        manager.allocate(req)
        self.compute(manager, req, 512)
        req.generation_idx = 1
        manager.append_slot(req)
        self.assertEqual(max(manager.tables[req.id][4]), 128)
        req.num_computed_tokens = 516
        manager.append_slot(req)
        self.assertEqual(max(manager.tables[req.id][4]), 129)
        self.assertEqual(max(manager.tables[req.id][3]), 64)
        manager.validate_conservation()

    def test_scheduler_reserves_verify_queries_and_lookahead_at_page_boundary(self):
        manager = self.manager()
        req = request(0, 250)
        manager.allocate(req)
        self.compute(manager, req, 250)
        req.generation_idx = 1
        scheduler = object.__new__(LLMPagedAttnScheduler)
        scheduler.block_manager = manager
        scheduler.mtp_enabled = True
        scheduler.cache_config = SimpleNamespace(hybrid_layout=self.layout)
        self.assertTrue(scheduler._reserve_decode_slot(req, [], [], 2, 4))
        # accepted=2 + K=3 stops at 255 and misses this page. The target
        # forward plus lookahead covers token end 250 + 4 + 3 = 257.
        full = self.layout.full_group
        self.assertEqual(max(manager.tables[req.id][full.group_id]), 1)
        for group in self.layout.groups:
            self.assertGreaterEqual(
                (max(manager.tables[req.id][group.group_id]) + 1) * group.block_tokens, 257
            )
        manager.validate_conservation()
        manager.free(req)
        manager.validate_conservation()
        self.assertEqual(sum(block.ref_count for block in manager.blocks), 1)

    def test_scheduler_uses_native_pool_and_prefill_chunks(self):
        hardware = HardwareContext.load("data")
        parallel = ClusterConfig.from_file(
            "data/clusters/8_b200/dep8.json"
        ).effective_parallel_config()
        cache = CacheConfig(
            256,
            hardware.device("b200"),
            self.model,
            parallel,
            max_num_batched_tokens=512,
            max_num_seqs=49,
        )
        scheduler = LLMPagedAttnScheduler(
            0, cache, max_parallem_sum=49, max_num_batched_tokens=512
        )
        env = simpy.Environment()
        req = request(0, 4096)
        req.arrive(env)
        scheduler.add_requests([req])
        output = scheduler.schedule_output()
        self.assertEqual(req.scheduled_tokens, 512)
        self.assertEqual(
            scheduler.block_manager.block_table.get_num_blocks(req.id), 210
        )
        self.assertTrue(scheduler.block_manager.validate_conservation())


class NativeEngineCacheIntegrationTest(unittest.TestCase):
    def engine(self, env):
        hardware = HardwareContext.load("data")
        cluster = ClusterConfig.from_file("data/clusters/8_b200/dep8.json")
        engine = LLMEngine(
            env=env,
            block_size=256,
            batching="paged-attn",
            kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=PSLAConfig.from_file("data/psla/deepseek-v4-pro.json"),
            cluster_config=cluster,
            hardware=hardware,
            parallel_config=cluster.effective_parallel_config(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=49,
            max_num_batched_tokens=512,
            gpu_memory_utilization=0.9,
            latency_backend_type="analytical",
        )
        return engine

    def test_full_native_engine_completes_chunks_and_releases_real_pool(self):
        env = simpy.Environment()
        engine = self.engine(env)
        for worker in engine.workers:
            worker.dynamic_batch = lambda requests: 0.001
        req = request(0, 4096)
        req.arrive(env)
        engine.add_requests_burst([req])
        env.run(until=req.completion_event)
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertEqual(req.prefill_chunks, 8)
        for worker in engine.workers:
            manager = worker.scheduler.block_manager
            self.assertTrue(manager.validate_conservation())
            self.assertEqual(manager.get_num_allocated_blocks(), 1)
        stats = get_cache_capacity_stats(engine)
        self.assertEqual(
            stats["kv_cache_capacity_tokens_per_dp_rank"],
            engine.workers[0].cache_config.hybrid_layout.equivalent_tokens(
                engine.workers[0].cache_config.num_gpu_blocks
            ),
        )
        self.assertIn("log-equivalent", stats["kv_cache_capacity_token_kind"])
        # Exercise the export schema as well as the capacity helper: a full
        # hour must not fail only when its result dataclass is constructed.
        from TokenSim.config.psla_config import LLMResult, MetricData
        from dataclasses import asdict

        result = LLMResult(
            cluster=ClusterConfig.from_file("data/clusters/8_b200/dep8.json"),
            qps=1,
            request_count=1,
            prefill_lens=[4096],
            decode_lens=[4],
            batching="paged-attn",
            request_time=MetricData.from_list([1]),
            prefill_time=MetricData.from_list([1]),
            decode_time=MetricData.from_list([1]),
            duration=1,
            output_qps=1,
            output_token_ps=4,
            notdone=[],
            **stats,
        )
        self.assertEqual(len(asdict(result)["worker_cache_budgets"]), 8)

    def test_native_benchmark_rejects_cacheless_or_unchunked_scheduling_before_loading(
        self,
    ):
        from benchmark import main

        for batching, chunked in (("dynamic", True), ("paged-attn", False)):
            args = SimpleNamespace(
                latency_backend="analytical",
                data_root="data",
                cluster="data/clusters/8_b200/dep8.json",
                kv_transfer_config=None,
                model_config_path="data/psla/deepseek-v4-pro.json",
                verbose="none",
                program_id=0,
                workload_type="agentx_weka",
                dataset_path="does-not-exist",
                batching=batching,
                chunked_prefill=chunked,
            )
            with self.assertRaisesRegex(ConfigurationError, "paged-attn and chunked"):
                main(args)

    def test_cancelling_unadmitted_prefix_probe_forgets_metadata(self):
        env = simpy.Environment()
        engine = self.engine(env)
        scheduler = engine.workers[0].scheduler
        req = request(0, 4096)
        req.arrive(env)
        scheduler.add_requests([req])
        manager = scheduler.block_manager
        manager.can_allocate(req)
        self.assertIn(req.id, manager.keys)
        self.assertTrue(scheduler.cancel_request(req))
        self.assertNotIn(req.id, manager.keys)
        self.assertNotIn(req.id, manager.fine_keys)

    def test_native_kernel_cancellation_releases_without_publishing_and_logs_capacity(
        self,
    ):
        env = simpy.Environment()
        engine = self.engine(env)
        for worker in engine.workers:
            worker.dynamic_batch = lambda requests: 5
        req = request(0, 4096)
        req.arrive(env)
        engine.add_requests_burst([req])
        env.run(until=1)
        ack = engine.cancel_requests([req])[0]
        self.assertFalse(ack.triggered)
        env.run(until=ack)
        engine.raise_if_failed()
        with tempfile.TemporaryDirectory() as directory:
            logs = NativeReplayLogs(directory, engine)
            logs.snapshot(env, SimpleNamespace(phase="cancelled", phase_counts={}))
            logs.close()
            record = json.loads((Path(directory) / "cache_windows.jsonl").read_text())
            self.assertTrue(all(w["used_slots"] == 1 for w in record["workers"]))
        for worker in engine.workers:
            manager = worker.scheduler.block_manager
            self.assertEqual(len(manager.cache), 0)
            self.assertTrue(manager.validate_conservation())


if __name__ == "__main__":
    unittest.main()
