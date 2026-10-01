"""Source expectations: scheduler.py:1034 and metrics/stats.py:PrefillStats."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.errors import SimulationStateError
from TokenSim.llm.llm_request import Request
from tests import test_native_simple_cpu as cpu_tests
from tests.test_hybrid_kv_cache import request
from util.results import get_agentx_metrics


class NativePrefillSourceStatsTest(unittest.TestCase):
    def test_first_admission_is_immutable_while_reloads_accumulate_separately(self):
        r = request(0, 4096)
        r.record_prefill_admission(1024, 2048)
        original = r.first_prefill_source
        r.record_prefill_admission(3840, 0)
        r.record_prefill_admission(0, 3840)
        self.assertIs(r.first_prefill_source, original)
        self.assertEqual(
            (
                original.gpu_hit_tokens,
                original.cpu_hit_tokens,
                original.computed_tokens,
            ),
            (1024, 2048, 1024),
        )
        self.assertEqual(r.prefill_admission_count, 3)
        self.assertEqual((r.reloaded_gpu_tokens, r.reloaded_cpu_tokens), (3840, 3840))
        with self.assertRaises(FrozenInstanceError):
            original.gpu_hit_tokens = 0
        with self.assertRaises(SimulationStateError):
            r.record_prefill_admission(-1, 0)
        with self.assertRaises(SimulationStateError):
            r.record_prefill_admission(4096, 1)

    def test_first_miss_stays_a_miss_after_recovery_uses_computed_gpu_cache(self):
        env = simpy.Environment()
        e = cpu_tests.engine(env)
        s = e.workers[0].scheduler
        manager = s.block_manager
        manager.publish_callbacks.clear()
        r = request(0, 4096)
        s._admit(r)
        self.assertEqual(r.first_prefill_source.computed_tokens, 4096)
        manager.ensure_slots(r, 4096)
        r.num_computed_tokens, r.generation_idx = 4096, 1
        manager.cache_computed(r)
        s._preempt(r)
        s.waiting.remove(r)
        s._admit(r)
        self.assertEqual(r.cached_prefill_tokens, 3840)
        self.assertEqual(r.first_prefill_source.gpu_hit_tokens, 0)
        self.assertEqual(r.first_prefill_source.computed_tokens, 4096)
        self.assertEqual(r.reloaded_gpu_tokens, 3840)
        self.assertTrue(manager.validate_conservation())

    def test_engine_cpu_first_hit_is_not_reclassified_when_gpu_recovers_it(self):
        helper = cpu_tests.NativeSimpleCPUTest()
        env, e, worker = helper.seeded()
        r = request(1, 4096)
        r.decode_len, r.dp_rank = 200, 0
        r.arrive(env)
        e.add_requests_burst([r])
        while r.generation_idx == 0:
            env.step()
            e.raise_if_failed()
        self.assertFalse(e.dp_coordinator.busy)
        self.assertEqual(r.first_prefill_source.cpu_hit_tokens, 3840)
        s = worker.scheduler
        # Force one recovery at a settled GPU boundary, as the scheduler's
        # FCFS victim path would do. The first source observation must survive.
        s.running.remove(r)
        s._preempt(r)
        env.run(until=r.completion_event)
        env.run(until=env.now + 0.01)
        e.raise_if_failed()
        self.assertEqual(r.prefill_admission_count, 2)
        self.assertEqual(r.first_prefill_source.gpu_hit_tokens, 0)
        self.assertEqual(r.first_prefill_source.cpu_hit_tokens, 3840)
        self.assertEqual(r.reloaded_gpu_tokens, 3840)
        self.assertEqual(r.reloaded_cpu_tokens, 0)
        self.assertEqual(r.generation_idx, 200)
        self.assertTrue(s.block_manager.validate_conservation())

    def test_native_metrics_use_first_sources_and_preserve_latest_diagnostics(self):
        r = request(0, 1024)
        r.record_prefill_admission(256, 512)
        r.cached_prefill_tokens, r.external_cached_prefill_tokens = 768, 0
        r.effective_prefill_tokens = 256
        r.record_prefill_admission(768, 0)
        r.recomputed_tokens_total = 512
        args = SimpleNamespace(
            workload_type="agentx_weka",
            agentx_runtime_metadata={"replay_protocol": "native"},
        )
        m = get_agentx_metrics(args, [r], 1, 8)
        self.assertEqual(m["cache_source_observation"], "first-admission")
        self.assertEqual(
            (
                m["gpu_cached_input_tokens"],
                m["cpu_cached_input_tokens"],
                m["computed_input_tokens"],
            ),
            (256, 512, 256),
        )
        self.assertEqual(
            (m["latest_gpu_hit_tokens"], m["latest_cpu_hit_tokens"]), (768, 0)
        )
        self.assertEqual(
            (m["readmission_count"], m["reloaded_gpu_tokens"], m["recomputed_tokens"]),
            (1, 768, 512),
        )

    def test_unobserved_native_source_is_missing_instead_of_zero_hit(self):
        r = request(0, 1024)
        args = SimpleNamespace(
            workload_type="agentx_weka",
            agentx_runtime_metadata={"replay_protocol": "native"},
        )
        self.assertIsNone(r.first_prefill_source)
        with self.assertRaises(ValueError):
            get_agentx_metrics(args, [r], 1, 8)


if __name__ == "__main__":
    unittest.main()
