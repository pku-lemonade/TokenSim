"""Router residency and token ownership across continuous batches."""

from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe.routing import RoutingAssignment
from tests.af.support.config import _cluster
from tests.af.support.execution import af_test_hardware, af_test_resources
from tests.af.support.model import _moe, _runtime_plan


class ConstantProvider:
    def estimate(self, query):
        return AFServiceEstimate(
            0.001, "placement-test", payload_bytes=getattr(query, "payload_bytes", None)
        )


class RouterPlacementTest(unittest.TestCase):
    def run_case(self, side):
        base = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster = replace(
            base,
            af=replace(
                base.af,
                router_location=side,
                af_block_size=2,
                transport_router_logits=True,
            ),
        )
        moe = replace(_moe(top_k=1, shared=1), num_shared_experts=0)
        plan = _runtime_plan(cluster, moe_config=moe, dtype="bf16")
        env = simpy.Environment()
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            ConstantProvider(),
            ConstantProvider(),
            "TestModel",
            "bf16",
            total_layers=2,
            hidden_size=16,
            moe_config=moe,
            random_seed=42,
            runtime_plan=plan,
            resources=af_test_resources(env, cluster, cluster.workers()),
        )
        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, prefill_len=8, context_len=9, generation_idx=1
            )
            for i in range(2)
        ]
        iteration = executor.submit(0, requests)
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
                router_bytes = sum(
                    c.bytes for c in state.components if c.category == "router"
                )
                self.assertEqual(
                    router_bytes > 0, state.physical_identity == f"{side}-pool:0"
                )
            logits = [e for e in events if e.operation == "af_router_logits"]
            if side == "a":
                self.assertEqual(len(logits), len(routers))
                for event in logits:
                    self.assertEqual(event.payload_bytes, 2 * 4 * 2)
                    experts = [
                        e
                        for e in events
                        if e.layer_id == event.layer_id and e.stage == "routed_expert"
                    ]
                    self.assertTrue(all(e.start_at >= event.end_at for e in experts))
            else:
                self.assertEqual(logits, [])
                metadata = [e for e in events if e.operation == "af_router_metadata"]
                self.assertEqual(len(metadata), len(routers))
                for event in metadata:
                    self.assertEqual(
                        event.payload_bytes,
                        2 * plan.layer_plan(event.layer_id).top_k * 8,
                    )
                    combines = [
                        e
                        for e in events
                        if e.layer_id == event.layer_id
                        and e.stage == "weighted_combine"
                    ]
                    self.assertTrue(combines)
                    for combine in combines:
                        self.assertIn(event.node_id, combine.predecessor_ids)
                        self.assertGreaterEqual(combine.start_at, event.end_at)
        workload = lambda es: sorted(
            (e.layer_id, e.token_count, str(e.routing_edges))
            for e in es
            if e.stage == "routed_expert"
        )
        self.assertEqual(workload(by_side["a"]), workload(by_side["f"]))


if __name__ == "__main__":
    unittest.main()


class ContinuousRoutingTest(unittest.TestCase):
    def make_executor(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster.af.af_block_size = 8
        hardware = af_test_hardware(layers=1, moe=_moe())
        plan = _runtime_plan(cluster, hardware=hardware, moe_config=_moe())
        env = simpy.Environment()
        return env, AFExecutor(
            env,
            cluster,
            cluster.workers(),
            ConstantProvider(),
            ConstantProvider(),
            "TestModel",
            "fp16",
            1,
            16,
            moe_config=_moe(),
            runtime_plan=plan,
            resources=af_test_resources(env, cluster, cluster.workers()),
        )

    def requests(self):
        result = [Request(i, 8, 8, 4) for i in (11, 22)]
        for req, step in zip(result, (1, 3)):
            req.generation_idx = step
        return result

    def test_synthetic_routes_preserve_each_request_position_in_one_block(self):
        env, executor = self.make_executor()
        iteration = executor.submit(0, self.requests())
        self.assertEqual(len(iteration.blocks), 1)
        snapshot = iteration.blocks[0].routing_snapshot_for_layer(0)
        self.assertEqual(dict(snapshot.request_decode_steps), {11: 1, 22: 3})
        self.assertEqual(
            {a.request_id: a.decode_step for a in snapshot.assignments}, {11: 1, 22: 3}
        )
        self.assertEqual(set(snapshot.original_tokens), {(11, 8), (22, 10)})
        env.run(until=env.process(executor.run_iteration(iteration)))
        executor.trace.validate_invariants(iteration.iteration_id)

    def test_external_routes_are_selected_by_each_request_step(self):
        _, executor = self.make_executor()
        requests = self.requests()
        for req in requests:
            req.decode_routing_assignments = [
                asdict(
                    RoutingAssignment(
                        0,
                        "decode",
                        req.generation_idx,
                        0,
                        req.id,
                        req.context_len - 1,
                        k,
                        k,
                        weight,
                        0,
                        0,
                    )
                )
                for k, weight in enumerate((0.7, 0.3))
            ]
        iteration = executor.submit(0, requests)
        snapshot = iteration.blocks[0].routing_snapshot_for_layer(0)
        self.assertEqual(snapshot.assignment_count, 4)
        wrong = replace(snapshot.assignments[0], decode_step=99)
        with self.assertRaisesRegex(WorkloadValidationError, "different.*step"):
            replace(snapshot, assignments=(wrong, *snapshot.assignments[1:]))
