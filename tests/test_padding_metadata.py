"""Serving-derived query padding, metadata roles and model-length edges."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import simpy

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.config.kv_layout import deepseek_v4_kv_layout
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.latency.deepseek_v4_moe import MoEPass, RankShape
import tests.test_full_mtp as full_mtp


class PaddingMetadataTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")
        full_mtp.FullMTPTest.setUpClass()

    def scheduler(self, budget=4, *, limit=1048576):
        layout = deepseek_v4_kv_layout(self.model)
        layout = replace(layout, groups=(layout.full_group,), max_model_len=limit)
        cache = SimpleNamespace(block_size=256, num_gpu_blocks=128,
                                num_cpu_blocks=0, model=self.model.model_id, hybrid_layout=layout)
        return LLMPagedAttnScheduler(0, cache, max_occupy_ratio=1,
                                     max_num_batched_tokens=budget, mtp_acceptance_length=2.49)

    def cached_tail(self, scheduler):
        manager = scheduler.block_manager
        producer = Request(0, 256, 1, 256, hash_ids=["shared"])
        manager.allocate(producer)
        manager.ensure_slots(producer, 256)
        producer.num_computed_tokens = 256
        manager.cache_computed(producer)
        manager.free(producer)
        return Request(1, 257, 5, 256, hash_ids=["shared"])

    def test_waiting_tail_verifies_four_rows_but_advances_one_and_samples_one(self):
        scheduler = self.scheduler()
        req = self.cached_tail(scheduler)
        env = simpy.Environment()
        req.arrive(env)
        scheduler.add_requests([req])
        out = scheduler.schedule_output()
        self.assertEqual(out.running, [req])
        self.assertEqual((req.step_tokens, req.query_tokens, req.scheduled_placeholder_tokens), (1, 4, 3))
        self.assertEqual(req.logits_tokens, 4)
        self.assertTrue(req.decode_graph_eligible)
        self.assertTrue(req.is_context_build)
        self.assertEqual(scheduler.stats["scheduled_context_tokens"], 1)
        self.assertEqual(scheduler.stats["scheduled_context_query_tokens"], 4)
        req.advance(env, 0.1, 1)
        self.assertEqual((req.num_computed_tokens, req.generation_idx), (257, 1))
        self.assertEqual(req.scheduled_placeholder_tokens, 0)

    def test_insufficient_padded_budget_keeps_fresh_waiting_prefix_unacquired(self):
        scheduler = self.scheduler(budget=3)
        req = self.cached_tail(scheduler)
        scheduler.add_requests([req])
        free = scheduler.block_manager.get_num_free_blocks()
        self.assertEqual(scheduler.schedule_output().running, [])
        self.assertEqual(scheduler.waiting, [req])
        self.assertNotIn(req.id, scheduler.block_manager.tables)
        self.assertEqual(scheduler.block_manager.get_num_free_blocks(), free)

    def test_loaded_tail_keeps_its_pages_until_complete_verify_budget_fits(self):
        scheduler = self.scheduler(budget=3)
        req = self.cached_tail(scheduler)
        scheduler.block_manager.allocate(req)
        req.start_context_build(256)
        req.status = RequestStatus.WAITING
        scheduler.loaded_waiting.append(req)
        scheduler.loading[req.id] = req
        blocks = scheduler.block_manager.block_table.get_blocks(req.id)
        refs = [block.ref_count for block in blocks]
        self.assertEqual(scheduler.schedule_output().running, [])
        self.assertEqual([block.ref_count for block in blocks], refs)
        self.assertEqual(scheduler.loaded_waiting, [req])
        scheduler.max_num_batched_tokens = 4
        self.assertEqual(scheduler.schedule_output().running, [req])
        self.assertEqual((req.step_tokens, req.query_tokens), (1, 4))

    def test_running_prefill_flag_suppresses_waiting_tail_padding(self):
        scheduler = self.scheduler(budget=516)
        tail = self.cached_tail(scheduler)
        old = Request(2, 1024, 5, 256)
        scheduler.block_manager.allocate(old)
        scheduler.block_manager.ensure_slots(old, 512)
        old.num_computed_tokens = 512
        old.prefill_chunk_pending = True
        old.status = RequestStatus.RUNNING
        scheduler.running.append(old)
        scheduler.add_requests([tail])
        out = scheduler.schedule_output()
        self.assertEqual(out.running, [old, tail])
        self.assertEqual((tail.step_tokens, tail.query_tokens, tail.logits_tokens), (1, 1, 1))

    def test_new_waiting_prefill_does_not_mutate_running_prefill_flag(self):
        scheduler = self.scheduler(budget=516)
        tail = self.cached_tail(scheduler)
        fresh = Request(3, 512, 5, 256)
        scheduler.add_requests([fresh, tail])
        out = scheduler.schedule_output()
        self.assertEqual(out.running, [fresh, tail])
        self.assertEqual((tail.step_tokens, tail.query_tokens), (1, 4))

    def test_model_length_disables_tail_padding_without_room_for_bonus(self):
        scheduler = self.scheduler(limit=260)
        req = self.cached_tail(scheduler)
        scheduler.add_requests([req])
        self.assertEqual(scheduler.schedule_output().running, [req])
        self.assertEqual((req.query_tokens, req.scheduled_placeholder_tokens), (1, 0))

    def test_prompt_at_model_limit_is_rejected_before_allocation(self):
        scheduler = self.scheduler(limit=16)
        with self.assertRaises(ConfigurationError):
            scheduler.add_requests([Request(4, 16, 1, 256)])
        self.assertEqual(scheduler.waiting, [])
        self.assertEqual(scheduler.block_manager.get_num_allocated_blocks(), 1)

    def test_length_cap_clips_query_and_output_and_releases_request(self):
        scheduler = self.scheduler(budget=16, limit=16)
        req = Request(5, 8, 20, 256)
        env = simpy.Environment()
        req.arrive(env)
        scheduler.add_requests([req])
        widths = []
        for _ in range(12):
            out = scheduler.schedule_output()
            if not out.running:
                break
            self.assertEqual(out.running, [req])
            widths.append(req.query_tokens)
            self.assertLessEqual(req.num_computed_tokens + req.query_tokens + 1, 16)
            req.advance(env, 0.1, 1)
            scheduler.update(out.running)
            if req.is_done:
                break
        self.assertTrue(req.is_done)
        self.assertEqual(req.context_len, 16)
        self.assertEqual(req.generation_idx, 8)
        self.assertEqual(req.decode_len, 20)  # original requested length retained
        self.assertEqual(req.status, RequestStatus.FINISHED_LENGTH_CAPPED)
        self.assertTrue(any(width < 4 for width in widths[1:]))
        self.assertTrue(req.completion_event.triggered)
        self.assertEqual(scheduler.block_manager.get_num_allocated_blocks(), 1)

    def test_short_context_uses_decode_attention_but_one_bonus_logit(self):
        backend = full_mtp.FullMTPTest().backend()
        req = Request(6, 8192, 20, 256)
        req.num_computed_tokens = 8190
        req.scheduled_tokens = req.scheduled_query_tokens = 2
        with patch.object(backend.native_attention, "estimate_decode_batch", return_value=0) as decode, patch.object(
            backend, "_attention_prefill", return_value=0
        ) as prefill, patch.object(backend, "_compose", side_effect=lambda **step: step["requests"]):
            rows = backend.estimate_step_latency([req])
        decode.assert_called_once()
        prefill.assert_not_called()
        self.assertEqual(rows, 1)
        self.assertFalse(req.decode_graph_eligible)

    def test_recompute_graph_role_and_placeholder_logits_are_independent(self):
        backend = full_mtp.FullMTPTest().backend()
        req = Request(7, 256, 1024, 256)
        req.generation_idx = 260
        req.needs_recompute = True
        req.num_computed_tokens = 512
        req.scheduled_tokens = req.scheduled_query_tokens = 4
        self.assertTrue(req.is_context_build)
        self.assertTrue(req.decode_graph_eligible)
        self.assertEqual(req.logits_tokens, 1)
        backend.set_moe_step(MoEPass({rank: RankShape(4, 4, False, 1) for rank in range(8)}))
        backend.estimate_step_latency([req])
        self.assertEqual(backend.target_logits_counts["rows"], 1)
        self.assertEqual(backend.dsv4_hc_head.rows, 4)
        self.assertEqual(backend.native_mtp.counts["graph_passes"], 3)
