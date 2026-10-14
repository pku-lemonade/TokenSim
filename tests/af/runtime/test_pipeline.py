"""Pipeline stages own their model/cache shards and execute concurrently."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.execution import FixedComputeProvider, FixedTransferProvider
from tests.af.support.model import _runtime_hardware, _moe
from tests.af.support.pipeline import pipeline_cluster, decode_request


class PipelineRuntimeTest(unittest.TestCase):
    def runtime(self, a_pp=2, f_pp=2):
        cluster = pipeline_cluster(a_pp, f_pp)
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), moe=_moe(top_k=1, shared=1))
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        return cluster, model, plan

    def test_stage_weights_cache_and_collectives_use_physical_ranks(self):
        cluster, model, plan = self.runtime()
        _, _, single = self.runtime(1, 1)
        self.assertEqual(
            sum(state.model_bytes for state in plan.physical_states),
            sum(state.model_bytes for state in single.physical_states),
        )
        self.assertEqual(
            sum(
                b.kv_partition.bytes_per_token
                for b in plan.rank_bindings
                if b.kv_partition
            ),
            sum(
                b.kv_partition.bytes_per_token
                for b in single.rank_bindings
                if b.kv_partition
            ),
        )
        for binding in plan.rank_bindings:
            state = plan.physical_state(binding.physical_identity)
            self.assertTrue(
                all(
                    component.layer_id in (None, binding.pp_rank)
                    for component in state.components
                )
            )
            self.assertEqual(
                binding.tp_group, (2 * binding.pp_rank, 2 * binding.pp_rank + 1)
            )
        self.assertEqual(plan.replica("a-domain", 0).leader_rank_key, ("a-domain", 0))
        self.assertEqual(
            plan.collective_plan("a-domain", 0, "residual", 1).participant_ranks, (2, 3)
        )
        with self.assertRaises(ConfigurationError):
            self.runtime(3, 2)

    def test_pipeline_handoff_uses_the_production_transport_provider(self):
        cluster, model, plan = self.runtime()
        hardware = _runtime_hardware()
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
        trace = AFTraceRecorder(True)
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
            trace_recorder=trace,
        )
        iteration = executor.submit(0, [decode_request(1)], "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        handoffs = [
            event
            for event in trace.events
            if event.operation == "af_pipeline_handoff"
            and event.event_type == "transfer"
        ]
        self.assertTrue(handoffs)
        self.assertTrue(all(event.match_type == "analytical" for event in handoffs))

    def simulate(self, parallel):
        cluster, model, plan = self.runtime()
        env = simpy.Environment()
        trace = AFTraceRecorder(True)
        transfer = FixedTransferProvider(0, 0)
        transfer.operation_durations["af_pipeline_handoff"] = 0.1
        transfer.placement = build_topology_placement(
            cluster, cluster.workers(), _runtime_hardware()
        )
        executor = ModelAFExecutor(
            env,
            cluster,
            cluster.workers(),
            FixedComputeProvider(1, 1),
            transfer,
            model.model_id,
            "bf16",
            2,
            model.hidden_size,
            runtime_plan=plan,
            model=model,
            moe_config=model.moe,
            trace_recorder=trace,
        )

        def run():
            processes = []
            for request_id in (1, 2, 3, 4):
                iteration = executor.submit(0, [decode_request(request_id)], "decode")
                process = env.process(executor.run_iteration(iteration))
                if parallel:
                    processes.append(process)
                else:
                    yield process
            if processes:
                yield env.all_of(processes)

        env.process(run())
        env.run()
        return env.now, trace

    def test_microbatches_overlap_stages_and_conserve_handoff_bytes(self):
        serial, _ = self.simulate(False)
        parallel, trace = self.simulate(True)
        self.assertLess(parallel, serial)
        transfers = [
            event
            for event in trace.events
            if event.operation == "af_pipeline_handoff"
            and event.event_type == "transfer"
        ]
        self.assertEqual(
            sum(event.payload_bytes for event in transfers), 4 * 2 * 16 * 2
        )
        attention = [
            event
            for event in trace.events
            if event.event_type == "compute" and event.service_duration > 0
        ]
        overlapping = [
            (left, right)
            for left in attention
            for right in attention
            if left.iteration_id != right.iteration_id
            and left.layer_id != right.layer_id
            and left.start_at < right.end_at
            and right.start_at < left.end_at
        ]
        self.assertTrue(overlapping)
