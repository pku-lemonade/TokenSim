"""Serving latency includes model boundaries for ordinary MHA and GQA."""

import unittest

import simpy

from TokenSim.af.model_executor import ModelAFExecutor
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from tests.test_af_executor import af_test_hardware, _cluster_dict


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
