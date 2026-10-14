"""An AF block can contain requests at independent decode positions."""

from dataclasses import asdict, replace
import unittest

import simpy

from TokenSim.af import AFExecutor
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe.routing import RoutingAssignment
from tests.test_af_executor import af_test_hardware, af_test_resources
from tests.test_af_operator_provider import _moe
from tests.test_af_parallel_planning import _cluster
from tests.test_af_router_placement import ConstantProvider
from tests.test_af_runtime_state import _runtime_plan


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
        iteration = executor.submit(0, self.requests(), "decode")
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
        iteration = executor.submit(0, requests, "decode")
        snapshot = iteration.blocks[0].routing_snapshot_for_layer(0)
        self.assertEqual(snapshot.assignment_count, 4)
        wrong = replace(snapshot.assignments[0], decode_step=99)
        with self.assertRaisesRegex(WorkloadValidationError, "different.*step"):
            replace(snapshot, assignments=(wrong, *snapshot.assignments[1:]))
