"""Native 1A1F protocol contracts, with no GPU or fitted timings required."""
import copy
from tests.test_af_executor import af_test_resources
import unittest
from dataclasses import replace
from types import SimpleNamespace

import simpy

from scripts.deepseek_v3_runtime import configuration
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af import AFExecutor, AFServiceEstimate
from TokenSim.af.readiness import TransferReadiness
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError


class NativeProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = configuration(a_tp=1, f_tp=1, af_batch_size=8)
        cls.model = ModelSpec.from_mapping(dict(
            model_id="qwen3-moe-tiny-random", hidden_size=256, intermediate_size=512,
            num_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            vocab_size=128, dtype="bf16", kv_cache_dtype="bf16", attention_type="gqa",
            moe=dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=256,
                     num_shared_experts=0, num_moe_layers=2)))

    def document(self, **overrides):
        doc = copy.deepcopy(self.base.doc)
        doc["af"].update(execution_protocol="vllm_afd", phases=["decode"],
                         communication_timing=dict(enabled=True))
        doc["af"].update(overrides)
        for entry in doc["af"]["links"] + doc["af"]["execution_domains"]:
            entry["phases"] = ["decode"]
        return doc

    def run_case(self, top_k=2, cache=True):
        model = replace(self.model, moe=replace(self.model.moe, num_experts_per_tok=top_k))
        runtime = build_model_runtime(self.document(service_estimate_cache=cache,
            dag_template_cache=cache), hardware=self.base.hardware, model=model)
        executor, rows, _, _ = simulate_decode(runtime, 8, 129, steps=2, fallback="analytical_only")
        for row in rows:
            executor.trace.validate_invariants(row["iteration_id"])
        return executor, rows

    def test_hidden_return_and_f_combine_are_independent_of_top_k(self):
        for k in (1, 2):
            executor, rows = self.run_case(k)
            for row in rows:
                for layer_id in (0, 1):
                    events = [e for e in executor.trace.events
                              if e.iteration_id == row["iteration_id"] and e.layer_id == layer_id]
                    transfers = [e for e in events if e.event_type == "transfer"]
                    self.assertEqual(len(transfers), 2)
                    self.assertEqual([e.payload_bytes for e in transfers], [4096, 4096])
                    combines = [e for e in events if e.stage == "weighted_combine"]
                    self.assertEqual(len(combines), 1)
                    combine = combines[0]
                    self.assertEqual(combine.physical_endpoints, ("f-pool:0",))
                    self.assertEqual(combine.input_edge_count, 8*k)
                    self.assertEqual(combine.output_token_count, 8)
                    returned = next(e for e in transfers if e.direction == "f_to_a")
                    self.assertGreaterEqual(returned.start_at, combine.end_at)
                    residual = next(e for e in events if e.stage == "residual")
                    self.assertEqual(residual.physical_endpoints, ("a-pool:0",))
                    self.assertGreaterEqual(residual.start_at, returned.end_at)
                    expert = next(e for e in events if e.stage == "routed_expert")
                    self.assertEqual(expert.assignment_count, 8*k)
                    # Use distinct expert/token values to detect a lost edge,
                    # changed weight, duplicate assignment or wrong token owner.
                    router = next(e for e in events if e.stage == "router")
                    def weighted_outputs(edges):
                        result = {}
                        for edge in edges:
                            token = (edge.request_id, edge.token_id)
                            value = (edge.expert_id + 1) * 101 + edge.request_id
                            result[token] = result.get(token, 0) + value * edge.routing_weight
                        return result
                    self.assertEqual(weighted_outputs(router.routing_edges),
                                     weighted_outputs(combine.routing_edges))
                    self.assertEqual(len(weighted_outputs(combine.routing_edges)), 8)
                    components = expert.details["components"]
                    self.assertEqual([c["table"] for c in components], ["routed_experts"])
                    attention = next(e for e in events if e.stage == "attention")
                    names = {c["name"] for c in attention.details["components"]}
                    self.assertTrue({"q_norm", "k_norm"}.issubset(names))

    def test_cache_preserves_timeline_and_invocation_identity(self):
        cached, _ = self.run_case(cache=True)
        uncached, _ = self.run_case(cache=False)
        signature = lambda ex: [(e.node_id, e.ready_at, e.start_at, e.end_at, e.payload_bytes)
                                for e in ex.trace.events]
        self.assertEqual(signature(cached), signature(uncached))
        self.assertEqual(len({e.node_id for e in cached.trace.events}), len(cached.trace.events))

    def test_router_is_selected_by_protocol(self):
        runtime = build_model_runtime(self.document(router_location="a"),
                                      hardware=self.base.hardware, model=self.model)
        self.assertEqual(runtime.cluster.af.router_location, "f")

    def test_unsupported_phase_and_execution_fail_early(self):
        for override in (dict(phases=["prefill", "decode"]),
                         dict(communication_timing=dict(enabled=False)),
                         dict(transport_router_logits=True),
                         dict(routed_expert_execution="per_expert")):
            with self.subTest(override=override), self.assertRaises((ValueError, ConfigurationError)):
                build_model_runtime(self.document(**override), hardware=self.base.hardware, model=self.model)

    def test_shared_experts_and_dtype_are_rejected(self):
        shared = replace(self.model, moe=replace(self.model.moe, num_shared_experts=1))
        with self.assertRaisesRegex(ConfigurationError, "without shared"):
            build_model_runtime(self.document(), hardware=self.base.hardware, model=shared)
        with self.assertRaisesRegex(ConfigurationError, "BF16"):
            build_model_runtime(self.document(activation_dtype="fp16"),
                                hardware=self.base.hardware, model=self.model)

    def test_stream_wait_is_not_transport_service_or_fabric_queue(self):
        class Provider:
            def estimate(self, query):
                payload = getattr(query, "payload_bytes", None)
                return AFServiceEstimate(5e-6 if payload else 0, "oracle",
                    payload_bytes=payload, details={"timing_scope": "transport_service"})

        runtime = build_model_runtime(self.document(
            communication_timing=dict(enabled=True, host_submit_us=0)),
            hardware=self.base.hardware, model=self.model)
        env = simpy.Environment()
        seen = []

        def readiness(block, layer, step):
            seen.append(step.node_id)
            return TransferReadiness(receiver_stream=env.timeout(10e-6))

        executor = AFExecutor(env, runtime.cluster, runtime.workers, Provider(), Provider(),
            self.model.model_id, "bf16", 2, 256, moe_config=self.model.moe,
            runtime_plan=runtime.plan, transfer_readiness=readiness, resources=af_test_resources(env, runtime.cluster, runtime.workers))
        for _ in range(2):
            iteration = executor.submit(0, [SimpleNamespace(id=0, is_prefill=False,
                prefill_len=128, context_len=129, generation_idx=0)], "decode")
            env.process(executor.run_iteration(iteration))
            env.run()
            executor.trace.validate_invariants(iteration.iteration_id)
        transfers = [e for e in executor.trace.events if e.event_type == "transfer"]
        self.assertEqual(len(seen), len(transfers))
        self.assertEqual(len(set(seen)), len(seen))
        for event in transfers:
            self.assertAlmostEqual(event.details["readiness_wait_duration"], 10e-6)
            self.assertAlmostEqual(event.queue_duration, 0)
            self.assertAlmostEqual(event.service_duration, 5e-6)

    def test_readiness_rejects_events_from_another_simulation(self):
        from TokenSim.errors import SimulationStateError
        with self.assertRaises(SimulationStateError):
            TransferReadiness(receiver_stream=simpy.Environment().event()).events(simpy.Environment())


if __name__ == "__main__":
    unittest.main()
