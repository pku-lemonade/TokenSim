"""Implicit domains preserve dense, routed, and shared model components."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from tests.test_af_executor import af_test_hardware, _cluster_dict
from tests.test_af_operator_provider import _moe


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
