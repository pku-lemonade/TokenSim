"""Independent request streams use native PP resources and resident KV leases."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.af.serving.metrics import steady_metrics
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from tests.af.support.pipeline import pipeline_cluster
from tests.af.support.model import _runtime_hardware, _moe
from tests.af.support.execution import FixedComputeProvider, FixedTransferProvider
from TokenSim.hardware.placement import build_topology_placement


class CohortServingTest(unittest.TestCase):
    def test_production_provider_reserves_endpoint_injection_atomically(self):
        cluster = pipeline_cluster(resource_scope="endpoint")
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), moe=_moe(top_k=1, shared=1))
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        placement = build_topology_placement(cluster, cluster.workers(), hardware)
        runtime = SimpleNamespace(
            hardware=hardware,
            model=model,
            cluster=cluster,
            workers=cluster.workers(),
            plan=plan,
            placement=placement,
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        sessions = [
            service.admit(i, 0, maximum_context=16, current_context=8) for i in range(2)
        ]
        for session in sessions:
            env.process(service.forward([session]))
        env.run()
        self.assertFalse(service.executor.resources.endpoint_resources.scheduler.owners)
        self.assertFalse(
            service.executor.resources.endpoint_resources.scheduler.waiters
        )
        self.assertTrue(
            any(event.event_type == "collective" for event in service.trace.events)
        )
        self.assertTrue(
            any("link:endpoint:" in event.resource_id for event in service.trace.events)
        )

    def test_independent_cohorts_overlap_and_report_utilization(self):
        cluster = pipeline_cluster()
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), moe=_moe(top_k=1, shared=1))
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        placement = build_topology_placement(cluster, cluster.workers(), hardware)
        runtime = SimpleNamespace(
            hardware=hardware,
            model=model,
            cluster=cluster,
            workers=cluster.workers(),
            plan=plan,
            placement=placement,
        )
        env = simpy.Environment()
        transfer = FixedTransferProvider(0, 0)
        transfer.operation_durations["af_pipeline_handoff"] = 0.1
        transfer.placement = placement
        service = AFService(
            env, runtime, providers=(FixedComputeProvider(1, 1), transfer)
        )
        events = []

        def stream(session):
            for _ in range(3):
                yield from service.forward([session])
                session.generated += 1
                events.append(dict(time=env.now, tokens=1, phase="decode"))

        sessions = [
            service.admit(i, 0, maximum_context=20, current_context=8) for i in range(4)
        ]
        for session in sessions:
            env.process(stream(session))
        env.run()
        metrics = steady_metrics(
            events,
            service.trace.events,
            start=1,
            end=env.now + 1e-6,
            capacities=service.executor.resources.resource_capacities(),
        )
        self.assertEqual(metrics["decode_tokens"], 12)
        self.assertTrue(
            all(
                value <= 1 + 1e-12 for value in metrics["resource_utilization"].values()
            )
        )
        self.assertGreater(sum(metrics["resource_queue_s"].values()), 0)
        self.assertEqual(service.summary()["passes"], 12)
        for session in sessions:
            service.release(session)
        self.assertFalse(service.memory.leases)
