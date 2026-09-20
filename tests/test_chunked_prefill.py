"""Chunked prefill: token-budgeted steps that mix decode tokens and prefill chunks."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.config.config import ClusterConfig, WorkerGroupConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.errors import ConfigurationError
from TokenSim.latency import OperatorTableLatencyBackend
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, RequestStatus, g_time, reset_g_time
from TokenSim.llm.llm_scheduler import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    LLMPagedAttnScheduler,
    SchedulePhase,
)
from TokenSim.moe import ExpertRouting
from tests.hardware_fixtures import moe_test_model, test_device, test_hardware, test_model
from util.request import LLMSource
from util.results import export_result

BLOCK = 16


def _env(now: float = 0.0):
    return SimpleNamespace(now=now)


def _request(request_id: int, prefill_len: int, decode_len: int = 2, arrived_at: float | None = None) -> Request:
    req = Request(id=request_id, prefill_len=prefill_len, decode_len=decode_len, block_size=BLOCK)
    req.arrive(_env(float(request_id) if arrived_at is None else arrived_at))
    return req


def _cache_config(num_gpu_blocks: int, num_cpu_blocks: int = 64):
    return SimpleNamespace(block_size=BLOCK, num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks, model="test")


def _scheduler(budget: int | None, num_gpu_blocks: int = 4096, **kwargs) -> LLMPagedAttnScheduler:
    return LLMPagedAttnScheduler(
        id=0,
        cache_config=_cache_config(num_gpu_blocks),
        max_parallem_sum=kwargs.pop("max_parallem_sum", None),
        max_occupy_ratio=1,
        max_num_batched_tokens=budget,
        **kwargs,
    )


def _run_step(scheduler: LLMPagedAttnScheduler, now: float, latency: float = 0.01) -> list[Request]:
    """One worker step: schedule, apply, update. Returns the step's batch."""
    output = scheduler.schedule_output()
    batch = list(output.running)
    for req in batch:
        req.advance(_env(now), latency, len(batch))
    scheduler.update(batch)
    return batch


class RequestStepStateTest(unittest.TestCase):
    def test_partial_chunks_accumulate_service_time_and_only_the_last_samples(self):
        req = _request(0, prefill_len=100, decode_len=2, arrived_at=0.0)
        req.start_context_build(cached_tokens=0)

        req.scheduled_tokens = 40
        self.assertIsNone(req.advance(_env(1.0), 0.5, 1))
        self.assertEqual((req.num_computed_tokens, req.remaining_context_tokens), (40, 60))
        self.assertTrue(req.is_prefill)
        req.scheduled_tokens = 60
        self.assertIsNone(req.advance(_env(2.0), 0.7, 3))

        self.assertEqual(req.generation_idx, 1)
        self.assertEqual(req.prefill_chunks, 2)
        self.assertAlmostEqual(req.prefill_latency, 2.0)  # arrival -> first token, both chunks
        self.assertAlmostEqual(req.prefill_service_time, 1.2)
        self.assertEqual(req.prefill_batch_size, 3)
        # decode: exactly one token to compute per step
        self.assertEqual(req.remaining_context_tokens, 1)
        req.scheduled_tokens = 1
        req.advance(_env(2.1), 0.1, 1)
        self.assertTrue(req.is_done)
        self.assertAlmostEqual(req.decode_service_time_sum, 0.1)

    def test_full_prefix_hit_still_computes_the_last_token(self):
        req = _request(0, prefill_len=64)
        req.start_context_build(cached_tokens=64)
        self.assertEqual(req.remaining_context_tokens, 1)

    def test_recompute_completion_samples_and_books_recompute_service(self):
        req = _request(0, prefill_len=32, decode_len=4)
        req.start_context_build(0)
        req.scheduled_tokens = 32
        req.advance(_env(1.0), 0.2, 1)
        req.scheduled_tokens = 1
        req.advance(_env(1.1), 0.05, 1)
        self.assertEqual(req.generation_idx, 2)

        req.prepare_recompute()
        self.assertEqual((req.num_computed_tokens, req.recompute_tokens), (0, 34))
        req.start_context_build(0)
        req.scheduled_tokens = 20
        req.advance(_env(2.0), 0.3, 1)
        self.assertTrue(req.needs_recompute)
        req.scheduled_tokens = 14
        req.advance(_env(2.5), 0.3, 1)

        self.assertFalse(req.needs_recompute)
        self.assertEqual(req.generation_idx, 3)  # the restoring pass yields the next token
        self.assertEqual((req.recomputation_count, req.recomputed_tokens_total), (1, 34))
        self.assertAlmostEqual(req.recompute_service_time, 0.6)
        self.assertAlmostEqual(req.decode_service_time_sum, 0.05)  # not double booked
        self.assertAlmostEqual(req.decode_time_max, 2.5 - 1.1)  # the preemption stall shows in ITL


class ChunkedSchedulerTest(unittest.TestCase):
    def test_long_prompt_is_prefilled_over_budget_sized_chunks(self):
        scheduler = _scheduler(budget=1000)
        req = _request(0, prefill_len=2500, decode_len=2)
        scheduler.add_requests([req])

        chunks = []
        while req.is_prefill:
            batch = _run_step(scheduler, now=len(chunks) + 1.0)
            self.assertEqual(batch, [req])
            chunks.append(batch[0].num_computed_tokens)

        self.assertEqual(chunks, [1000, 2000, 2500])
        self.assertEqual(req.prefill_chunks, 3)
        self.assertEqual(req.generation_idx, 1)
        # decode step: one token, no chunk
        batch = _run_step(scheduler, now=5.0)
        self.assertEqual([r.scheduled_tokens for r in batch], [0])  # consumed by advance
        self.assertTrue(req.is_done)

    def test_decodes_are_served_before_prefill_chunks_share_the_budget(self):
        scheduler = _scheduler(budget=100)
        old = _request(0, prefill_len=50, decode_len=5, arrived_at=0.0)
        scheduler.add_requests([old])
        _run_step(scheduler, now=1.0)  # old finishes its prefill
        self.assertEqual(old.generation_idx, 1)

        young = _request(1, prefill_len=250, decode_len=2, arrived_at=1.0)
        scheduler.add_requests([young])

        output = scheduler.schedule_output()
        self.assertEqual(output.running, [old, young])
        self.assertEqual(output.phase, SchedulePhase.MIXED)
        self.assertEqual([r.scheduled_tokens for r in output.running], [1, 99])
        self.assertEqual(output.scheduled, [young])  # admissions of this step
        for req in output.running:
            req.advance(_env(2.0), 0.01, 2)
        scheduler.update(output.running)
        self.assertEqual(old.generation_idx, 2)
        self.assertEqual(young.num_computed_tokens, 99)

        # every following step keeps the decode flowing while the prompt continues
        steps = 0
        while young.is_prefill:
            output = scheduler.schedule_output()
            self.assertEqual(output.running[0], old)
            self.assertEqual(output.running[0].scheduled_tokens, 1)
            self.assertLessEqual(sum(r.scheduled_tokens for r in output.running), 100)
            for req in output.running:
                req.advance(_env(3.0 + steps), 0.01, len(output.running))
            scheduler.update(output.running)
            steps += 1
        self.assertEqual(steps, 2)  # 99 + 100 + 51 = 250
        self.assertEqual(young.prefill_chunks, 3)
        self.assertEqual(old.generation_idx, 4)

    def test_running_request_beyond_the_budget_waits_for_the_next_step(self):
        scheduler = _scheduler(budget=100)
        first = _request(0, prefill_len=300, decode_len=2, arrived_at=0.0)
        second = _request(1, prefill_len=300, decode_len=2, arrived_at=0.5)
        scheduler.add_requests([first, second])

        output = scheduler.schedule_output()
        # the whole budget goes to the first prompt; the second is not admitted yet
        self.assertEqual(output.running, [first])
        self.assertEqual(first.scheduled_tokens, 100)
        self.assertEqual(scheduler.waiting, [second])

    def test_admission_stops_at_budget_and_resumes_next_step(self):
        scheduler = _scheduler(budget=120)
        a = _request(0, prefill_len=100, decode_len=2, arrived_at=0.0)
        b = _request(1, prefill_len=100, decode_len=2, arrived_at=0.1)
        scheduler.add_requests([a, b])

        output = scheduler.schedule_output()
        self.assertEqual(output.running, [a, b])
        self.assertEqual([r.scheduled_tokens for r in output.running], [100, 20])
        for req in output.running:
            req.advance(_env(1.0), 0.01, 2)
        scheduler.update(output.running)

        output = scheduler.schedule_output()
        # a decodes (1 token), b takes the remaining 80 prompt tokens
        self.assertEqual([r.scheduled_tokens for r in output.running], [1, 80])

    def test_disabled_budget_prefills_whole_prompt_in_one_step(self):
        scheduler = _scheduler(budget=None)
        req = _request(0, prefill_len=50_000, decode_len=1)
        scheduler.add_requests([req])

        output = scheduler.schedule_output()

        self.assertEqual(output.phase, SchedulePhase.PREFILL)
        self.assertEqual(req.scheduled_tokens, 50_000)
        req.advance(_env(1.0), 1.0, 1)
        self.assertTrue(req.is_done)
        self.assertEqual(req.prefill_chunks, 1)

    def test_budget_must_be_positive(self):
        with self.assertRaises(ConfigurationError):
            _scheduler(budget=0)

    def test_preemption_of_a_partial_prefill_restarts_it_as_recompute(self):
        # 3 blocks of 16: a decoding request (1 block, growing) and a 32-token
        # prompt (2 blocks) whose prefill is chunked 16 tokens at a time.
        scheduler = _scheduler(budget=16, num_gpu_blocks=3)
        decoding = _request(0, prefill_len=15, decode_len=4, arrived_at=0.0)
        scheduler.add_requests([decoding])
        _run_step(scheduler, now=1.0)  # 15 prompt tokens + token 16 fill block 0
        self.assertEqual(decoding.generation_idx, 1)
        partial = _request(1, prefill_len=32, decode_len=1, arrived_at=1.0)
        scheduler.add_requests([partial])

        batch = _run_step(scheduler, now=2.0)  # decode token + first 15-token chunk
        self.assertEqual(batch, [decoding, partial])
        self.assertEqual(partial.num_computed_tokens, 15)

        # token 17 needs a new block; none is free, so the youngest running
        # request (the partial prefill) is preempted and loses its chunks.
        output = scheduler.schedule_output()
        self.assertEqual(output.preempted, [partial])
        self.assertEqual(output.running, [decoding])
        self.assertTrue(partial.needs_recompute)
        self.assertEqual((partial.num_computed_tokens, partial.recompute_tokens), (0, 32))
        self.assertEqual(scheduler.waiting, [partial])
        for req in output.running:
            req.advance(_env(3.0), 0.01, 1)
        scheduler.update(output.running)

        now = 4.0
        while not (decoding.is_done and partial.is_done):
            _run_step(scheduler, now=now)
            now += 1.0
            self.assertLess(now, 20.0, "requests did not finish")

        self.assertEqual((partial.recomputation_count, partial.recomputed_tokens_total), (1, 32))
        self.assertEqual(partial.prefill_chunks, 3)  # 15 tokens, then two 16-token recompute chunks
        self.assertEqual(partial.status, RequestStatus.FINISHED_STOPPED)


class MixedStepMoERoutingTest(unittest.TestCase):
    def test_chunk_routes_its_share_of_the_prompt_histogram(self):
        model = moe_test_model()
        routing = ExpertRouting(model.moe, seed=0)
        req = Request(0, prefill_len=8, decode_len=1, block_size=BLOCK, prefill_expert_histogram={0: 8, 1: 8})
        req.start_context_build(0)
        req.scheduled_tokens = 2
        self.assertEqual(routing.histogram_for_step([req]), {0: 2, 1: 2})
        req.scheduled_tokens = 8
        self.assertEqual(routing.histogram_for_step([req]), {0: 8, 1: 8})
        # unscheduled request: whole prompt, as before
        req.scheduled_tokens = 0
        self.assertEqual(routing.histogram_for_step([req]), {0: 8, 1: 8})

    def test_synthetic_histogram_follows_scheduled_tokens(self):
        model = moe_test_model()
        routing = ExpertRouting(model.moe, seed=0)
        req = Request(0, prefill_len=64, decode_len=1, block_size=BLOCK)
        req.start_context_build(0)
        req.scheduled_tokens = 4
        self.assertEqual(sum(routing.histogram_for_step([req]).values()), 4 * model.moe.num_experts_per_tok)


class ChunkedPrefillEngineTest(unittest.TestCase):
    def _run(self, budget: int | None, prefill_len: int, tmpdir: str):
        reset_g_time()
        env = simpy.Environment()
        device = test_device("TestGPU", memory_gib=2.0)
        model = test_model("TestModel", hidden_size=64, intermediate_size=128, num_layers=2, num_attention_heads=4)
        cluster = ClusterConfig(
            num_workers=1,
            networks={"net1": "ethernet-test"},
            worker_groups=[WorkerGroupConfig(role="hybrid", hardware="TestGPU", num_workers=1, network="net1")],
        )
        psla = PSLAConfig(
            name="chunk-test", model="TestModel", distribution="burst", prefill_mean_len=prefill_len, prefill_range_len=0,
            decode_mean_len=4, decode_range_len=0, decode_len_distribution="uniform",
            first_token_latency=MetricData(0, 0, 0), decode_token_latency=MetricData(0, 0, 0), qps=1,
        )
        engine = LLMEngine(
            env=env, block_size=BLOCK, batching="paged-attn", kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=psla, cluster_config=cluster, hardware=test_hardware(device, models=[model]),
            prefill_worker_pool_type="round_robin", decode_worker_pool_type="round_robin", max_parallem_sum=64,
            max_occupy_ratio=1, latency_backend_type="analytical", max_num_batched_tokens=budget,
        )
        short = Request(0, 64, 12, block_size=BLOCK)   # decoding while the long prompt is prefilled
        long = Request(1, prefill_len, 4, block_size=BLOCK)
        requests = [short, long]
        env.process(LLMSource(env, engine, requests, qps=1, distribution="burst"))

        def stop_when_done():
            while env.now < 60:
                if all(req.is_done for req in requests):
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return env.now
                yield env.timeout(1e-4)
            raise AssertionError("requests did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        engine.raise_if_failed()
        args = argparse.Namespace(qps=1, batching="paged-attn", results_path=tmpdir, cluster="unused.json")
        export_result(
            args=args, g_time=g_time, engine=engine, model_config=psla, cluster=cluster, request_count=2,
            prefill_lens=[64, prefill_len], decode_lens=[12, 4], requests=requests, notdone=[], duration=monitor.value,
        )
        result = json.loads((Path(tmpdir) / "result_1.json").read_text())
        return engine, short, long, result

    def test_chunking_bounds_context_queries_and_keeps_decode_flowing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, short, long, result = self._run(budget=512, prefill_len=2000, tmpdir=tmpdir)

        # both arrive together: step 1 holds the short prompt (64) plus 448 tokens of
        # the long one, then 511 + 511 + 511 alongside the short request's decodes, then 19
        self.assertEqual(long.prefill_chunks, 5)
        self.assertEqual(result["prefill_chunk_count"], 1 + 5)
        self.assertEqual(result["max_num_batched_tokens"], 512)
        backend = engine.workers[0].latency_backend
        context_queries = [dict(key[1]) for key in backend._cache if key[0] == "context_attention"]
        self.assertTrue(context_queries)
        self.assertLessEqual(max(q["input_seq_len"] for q in context_queries), 512)
        # the short request kept decoding during the long prefill: its longest
        # inter-token gap is one mixed step, not the whole 2000-token prefill
        self.assertGreater(short.decode_time_sum, 0)
        self.assertLess(short.decode_time_max, long.prefill_service_time)

    def test_disabled_chunking_prefills_in_one_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, short, long, result = self._run(budget=None, prefill_len=2000, tmpdir=tmpdir)

        self.assertEqual(long.prefill_chunks, 1)
        self.assertIsNone(result["max_num_batched_tokens"])
        backend = engine.workers[0].latency_backend
        context_queries = [dict(key[1]) for key in backend._cache if key[0] == "context_attention"]
        self.assertEqual(max(q["input_seq_len"] for q in context_queries), 2000)

    def test_default_budget_is_the_vllm_online_serving_default(self):
        self.assertEqual(DEFAULT_MAX_NUM_BATCHED_TOKENS, 8192)
        self.assertEqual(_scheduler(budget=DEFAULT_MAX_NUM_BATCHED_TOKENS).max_num_batched_tokens, 8192)


if __name__ == "__main__":
    unittest.main()
