"""Expert and tensor partitioning reuse the stage's actual physical ranks."""

import unittest
from types import SimpleNamespace

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.config import _cluster
from tests.af.support.model import _runtime_hardware, _runtime_plan, _request


class ExpertShardTest(unittest.TestCase):
    def test_expert_owners_memory_and_local_tensor_axes(self):
        cluster = _cluster(
            a_tp=1, a_dp=1, f_tp=4, f_dp=2, f_ep=4, placement_strategy="dedicated"
        )
        plan = _runtime_plan(cluster)
        self.assertEqual(plan.domain_runtime("f-domain").expert_tensor_parallel_size, 2)
        for expert in range(4):
            bindings = plan.expert_execution("f-domain", expert, 0)
            self.assertEqual(
                tuple(b.domain_rank for b in bindings), (2 * expert, 2 * expert + 1)
            )
            for offset, binding in enumerate(bindings):
                shape = plan.local_shape(
                    "f-domain", binding.domain_rank, 0, "routed_expert"
                )
                self.assertEqual(shape["expert_tp_rank"], offset)
                self.assertEqual(shape["intermediate_size"], 16)
                self.assertEqual(
                    plan.expert_tp_group(binding), (2 * expert, 2 * expert + 1)
                )
        routed = [
            component
            for state in plan.physical_states
            for component in state.components
            if component.category == "routed_expert"
        ]
        self.assertEqual(
            sum(component.bytes for component in routed), 4 * 3 * 16 * 32 * 2
        )

    def test_native_dag_executes_expert_collectives_on_the_owned_shards(self):
        cluster = _cluster(
            a_tp=1, a_dp=1, f_tp=4, f_dp=2, f_ep=4, placement_strategy="dedicated"
        )
        hardware = _runtime_hardware()
        plan = _runtime_plan(cluster, hardware=hardware)
        model = hardware.model("TestModel")
        # The shared runtime fixture gives this model a MoE layer.
        from dataclasses import replace
        from tests.af.support.model import _moe

        model = replace(model, moe=_moe(top_k=1, shared=1))
        workers = cluster.workers()
        placement = build_topology_placement(cluster, workers, hardware)
        compute, transfer = build_operator_table_af_providers(
            hardware=hardware,
            model=model,
            cluster_config=cluster,
            worker_configs=workers,
            placement=placement,
            fallback="analytical_only",
        )
        env = simpy.Environment()
        executor = ModelAFExecutor(
            env,
            cluster,
            workers,
            compute,
            transfer,
            model.model_id,
            model.dtype,
            model.num_layers,
            model.hidden_size,
            runtime_plan=plan,
            model=model,
            moe_config=model.moe,
            trace_recorder=AFTraceRecorder(True),
        )
        request = SimpleNamespace(
            id=1, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
        )
        iteration = executor.submit(0, [request], "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        routed = [e for e in executor.trace.events if e.stage == "routed_expert"]
        self.assertEqual(len(routed), 2)
        self.assertEqual(len({e.resource_id for e in routed}), 2)
        self.assertGreater(iteration.makespan, 0)
