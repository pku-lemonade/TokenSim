"""Router placement changes ownership and dependency order, not routed work."""
import unittest
from tests.test_af_executor import af_test_resources
from dataclasses import replace
from types import SimpleNamespace

import simpy

from tests.test_af_parallel_planning import _cluster
from tests.test_af_runtime_state import _runtime_plan, _moe_config
from TokenSim.af import AFExecutor, AFServiceEstimate


class ConstantProvider:
    def estimate(self, query):
        return AFServiceEstimate(0.001, "placement-test",
            payload_bytes=getattr(query, "payload_bytes", None))


class RouterPlacementTest(unittest.TestCase):
    def run_case(self, side):
        base = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster = replace(base, af=replace(base.af, router_location=side,
            af_block_size=2, transport_router_logits=True))
        moe = replace(_moe_config(), num_shared_experts=0)
        plan = _runtime_plan(cluster, moe_config=moe, dtype="bf16")
        env = simpy.Environment()
        executor = AFExecutor(env, cluster, cluster.workers(), ConstantProvider(),
            ConstantProvider(), "TestModel", "bf16", total_layers=2, hidden_size=16,
            moe_config=moe, random_seed=42, runtime_plan=plan, resources=af_test_resources(env, cluster, cluster.workers()))
        requests = [SimpleNamespace(id=i, is_prefill=False, prefill_len=8,
            context_len=9, generation_idx=1) for i in range(2)]
        iteration = executor.submit(0, requests, "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        executor.trace.validate_invariants(iteration.iteration_id)
        return plan, executor.trace.events

    def test_ownership_dependencies_and_work_are_preserved(self):
        by_side = {}
        for side in ("a", "f"):
            plan, events = self.run_case(side)
            by_side[side] = events
            routers = [e for e in events if e.stage == "router"]
            self.assertTrue(routers)
            for event in routers:
                self.assertEqual(event.physical_endpoints, (f"{side}-pool:0",))
                layer = [e for e in events if e.layer_id == event.layer_id]
                hidden = next(e for e in layer if e.operation == "af_a_to_f")
                experts = [e for e in layer if e.stage == "routed_expert"]
                if side == "f":
                    self.assertGreaterEqual(event.start_at, hidden.end_at)
                else:
                    self.assertGreaterEqual(hidden.start_at, event.end_at)
                self.assertTrue(all(e.start_at >= event.end_at for e in experts))
            for state in plan.physical_states:
                router_bytes = sum(c.bytes for c in state.components if c.category == "router")
                self.assertEqual(router_bytes > 0, state.physical_identity == f"{side}-pool:0")
            logits = [e for e in events if e.operation == "af_router_logits"]
            if side == "a":
                self.assertEqual(len(logits), len(routers))
                for event in logits:
                    self.assertEqual(event.payload_bytes, 2 * 4 * 2)
                    experts = [e for e in events if e.layer_id == event.layer_id and e.stage == "routed_expert"]
                    self.assertTrue(all(e.start_at >= event.end_at for e in experts))
            else:
                self.assertEqual(logits, [])
        workload = lambda es: sorted((e.layer_id, e.token_count, str(e.routing_edges))
            for e in es if e.stage == "routed_expert")
        self.assertEqual(workload(by_side["a"]), workload(by_side["f"]))


if __name__ == "__main__":
    unittest.main()
