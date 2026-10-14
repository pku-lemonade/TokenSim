"""Model boundaries, layer selection and request geometry."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.factory import simulate_decode
from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.config.cluster.core import ClusterConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from scripts.deepseek_v3_runtime import configuration
from tests.af.support.config import _cluster_dict
from tests.af.support.hardware import af_test_hardware
from tests.af.support.queries import _moe


class ModelLayersTest(unittest.TestCase):
    def test_implicit_prefill_and_decode_execute_each_layer_kind(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                cluster = ClusterConfig(**_cluster_dict())
                cluster.af.latency_provider = "operator_table"
                moe = replace(_moe(), first_k_dense_replace=1, num_shared_experts=1)
                hardware = af_test_hardware(layers=2, moe=moe, capacity_gib=0.1)
                env = simpy.Environment()
                engine = LLMEngine(
                    env=env,
                    kv_transfer_config=KVTransferConfig.default(),
                    model_spec=hardware.model("TestModel"),
                    cluster_config=cluster,
                    hardware=hardware,
                    config=SimulationConfig(latency_fallback="analytical_only"),
                )
                req = Request(1, 8, 3, 4)
                req.generation_idx = int(phase == "decode")
                executor = engine.af_executor
                iteration = executor.submit(0, [req], phase)
                env.run(until=env.process(executor.run_iteration(iteration)))
                stages = {
                    layer: {
                        event.stage
                        for event in executor.trace.events
                        if event.layer_id == layer and event.event_type == "compute"
                    }
                    for layer in (0, 1)
                }
                self.assertIn("dense_ffn", stages[0])
                self.assertNotIn("router", stages[0])
                self.assertNotIn("routed_expert", stages[0])
                self.assertTrue(
                    {"router", "routed_expert", "shared_expert", "weighted_combine"}
                    <= stages[1]
                )
                executor.trace.validate_invariants(iteration.iteration_id)


class ModelBoundaryTest(unittest.TestCase):
    def engine(self):
        cluster = ClusterConfig(**_cluster_dict())
        cluster.af.latency_provider = "operator_table"
        hardware = af_test_hardware(capacity_gib=0.1)
        env = simpy.Environment()
        return env, LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(latency_fallback="analytical_only"),
        )

    def test_ordinary_serving_includes_embedding_head_and_sampling(self):
        env, engine = self.engine()
        self.assertIsInstance(engine.af_executor, ModelAFExecutor)
        req = Request(1, 8, 3, 4)
        req.generation_idx = 1
        iteration = engine.af_executor.submit(0, [req], "decode")
        env.run(until=env.process(engine.af_executor.run_iteration(iteration)))
        boundaries = [
            e
            for e in engine.af_executor.trace.events
            if e.layer_id == -1 and e.event_type == "compute"
        ]
        self.assertEqual(
            [e.operation for e in boundaries],
            ["embedding_lookup", "final_norm", "lm_head", "token_selection"],
        )
        self.assertTrue(all(e.service_duration > 0 for e in boundaries))
        self.assertGreater(
            iteration.makespan, sum(e.service_duration for e in boundaries)
        )

    def test_partial_prefill_has_no_logits_until_its_prompt_is_complete(self):
        env, engine = self.engine()
        req = Request(1, 16, 3, 4)
        req.scheduled_tokens = 8
        first = engine.af_executor.submit(0, [req], "prefill")
        env.run(until=env.process(engine.af_executor.run_iteration(first)))
        self.assertFalse(
            any(
                e.operation == "lm_head"
                for e in engine.af_executor.trace.events_for(first.iteration_id)
            )
        )
        req.num_computed_tokens = 8
        second = engine.af_executor.submit(0, [req], "prefill")
        env.run(until=env.process(engine.af_executor.run_iteration(second)))
        head = [
            e
            for e in engine.af_executor.trace.events_for(second.iteration_id)
            if e.operation == "lm_head"
        ]
        self.assertEqual(len(head), 1)
        self.assertEqual(head[0].local_work["token_count"], 1)


class QueryGeometryTest(unittest.TestCase):
    def test_initial_chunk_does_not_read_uncomputed_prompt_keys(self):
        req = Request(1, 64, 3, 4)
        req.scheduled_tokens = 8
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        block = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(block.context_lengths, (8,))
        self.assertEqual(block.query_lengths, (8,))
        self.assertEqual(block.query_offsets, (0,))
        self.assertEqual(block.logits_rows, 0)
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(block),
            tuple((1, i) for i in range(8)),
        )

    def test_cached_suffix_uses_actual_query_positions_and_full_kv(self):
        req = Request(1, 64, 3, 4)
        req.start_context_build(56)
        req.scheduled_tokens = 8
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        block = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(block.context_lengths, (64,))
        self.assertEqual(block.query_offsets, (56,))
        self.assertEqual(block.logits_rows, 1)
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(block),
            tuple((1, i) for i in range(56, 64)),
        )


class MLAPrefillValidationTest(unittest.TestCase):
    def test_partial_prefill_does_not_submit_boundaries_or_change_ownership(self):
        runtime = configuration(af_batch_size=2)
        executor, _, _, _ = simulate_decode(runtime, 1, 16)
        for computed, scheduled in ((0, 8), (8, 56)):
            with self.subTest(computed=computed, scheduled=scheduled):
                req = Request(7, 64, 3, 16)
                req.num_computed_tokens = computed
                req.scheduled_tokens = scheduled
                before = len(executor.trace.events)
                with self.assertRaisesRegex(ConfigurationError, "unchunked prefill"):
                    executor.submit(0, [req], "prefill")
                self.assertEqual(len(executor.trace.events), before)
                self.assertEqual(req.af_a_owners, {})

    def test_whole_equal_length_prefill_is_accepted(self):
        runtime = configuration(af_batch_size=2)
        executor, _, _, _ = simulate_decode(runtime, 1, 16)
        requests = [Request(i, 16, 3, 16) for i in (1, 2)]
        for req in requests:
            req.scheduled_tokens = 16
        iteration = executor.submit(0, requests, "prefill")
        self.assertEqual(iteration.blocks[0].token_count, 32)
