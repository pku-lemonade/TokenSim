"""Warm KV transfer moves missing ranges and retains bounded windows."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.serving.service import AFService
from TokenSim.af.serving.handoff import plan_handoff, execute_handoff
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.pipeline import pipeline_cluster
from tests.af.support.model import _runtime_hardware


class HandoffTest(unittest.TestCase):
    def services(self, kv_heads=None):
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), attention_windows=(4, 0))
        if kv_heads is not None:
            model = replace(model, num_key_value_heads=kv_heads)
        env = simpy.Environment()
        runtimes = []
        for namespace, pp in (("source", 1), ("target", 2)):
            cluster = pipeline_cluster(
                pp, pp, resource_scope="endpoint", namespace=namespace
            )
            workers = cluster.workers()
            plan = compile_af_runtime_plan(
                cluster_config=cluster,
                worker_configs=workers,
                hardware=hardware,
                model=model,
                block_size=4,
            )
            runtimes.append(
                SimpleNamespace(
                    hardware=hardware,
                    model=model,
                    cluster=cluster,
                    workers=workers,
                    plan=plan,
                    placement=build_topology_placement(cluster, workers, hardware),
                )
            )
        identities = [
            worker.physical_worker_ref.identity
            for runtime in runtimes
            for worker in runtime.workers
        ]
        resources = EndpointResources(
            env, {"af-net": "test", "cross": "test", "replication": "test"}, identities
        )
        source, target = [
            AFService(env, runtime, communication=resources) for runtime in runtimes
        ]
        return env, hardware, resources, source, target

    def test_warm_suffix_bytes_and_commit_after_transport(self):
        env, hardware, resources, source, target = self.services()
        before = source.admit(1, 0, maximum_context=24, current_context=20)
        after = target.admit(1, 0, maximum_context=24, current_context=16)
        incremental = plan_handoff(
            source,
            before,
            target,
            after,
            context=20,
            mode="incremental",
            cross_network="cross",
            replication_network="replication",
        )
        full = plan_handoff(
            source,
            before,
            target,
            after,
            context=20,
            mode="full",
            cross_network="cross",
            replication_network="replication",
        )
        bytes_per_token = (
            source.runtime.model.kv_elements_per_token()
            * source.runtime.plan.kv_dtype_size_bytes
        )
        self.assertEqual(
            sum(copy.bytes for copy in incremental), 2 * 4 * bytes_per_token
        )
        self.assertEqual(sum(copy.bytes for copy in full), (4 + 20) * bytes_per_token)
        self.assertTrue(all(copy.first_token == 16 for copy in incremental))
        events = []
        env.process(
            execute_handoff(
                env,
                incremental,
                hardware=hardware,
                resources=resources,
                identity=(1,),
                target=target,
                target_session=after,
                context=20,
                events=events,
            )
        )
        self.assertEqual(after.cached_context, 16)
        env.run()
        self.assertEqual(after.cached_context, 20)
        self.assertEqual(len(events), len(incremental))
        self.assertFalse(resources.scheduler.owners)

    def test_replication_waits_for_the_cross_pool_copy(self):
        env, hardware, resources, source, target = self.services(kv_heads=1)
        before = source.admit(1, 0, maximum_context=24, current_context=20)
        after = target.admit(1, 0, maximum_context=24, current_context=16)
        copies = plan_handoff(
            source,
            before,
            target,
            after,
            context=20,
            mode="incremental",
            cross_network="cross",
            replication_network="replication",
        )
        self.assertEqual(sum(copy.replication for copy in copies), 2)
        events = []
        env.process(
            execute_handoff(
                env,
                copies,
                hardware=hardware,
                resources=resources,
                identity=(1,),
                target=target,
                target_session=after,
                context=20,
                events=events,
            )
        )
        env.run()
        for layer in (0, 1):
            cross = next(
                event
                for event in events
                if event["layer"] == layer and event["network"] == "cross"
            )
            local = next(
                event
                for event in events
                if event["layer"] == layer and event["network"] == "replication"
            )
            self.assertGreaterEqual(local["start_s"], cross["end_s"])

    def test_cancelled_handoff_does_not_commit_or_hold_network_endpoints(self):
        env, hardware, resources, source, target = self.services()
        before = source.admit(1, 0, maximum_context=24, current_context=20)
        after = target.admit(1, 0, maximum_context=24, current_context=16)
        copies = plan_handoff(
            source,
            before,
            target,
            after,
            context=20,
            mode="incremental",
            cross_network="cross",
            replication_network="replication",
        )
        process = env.process(
            execute_handoff(
                env,
                copies,
                hardware=hardware,
                resources=resources,
                identity=(1,),
                target=target,
                target_session=after,
                context=20,
                events=[],
            )
        )

        def cancel():
            yield env.timeout(1e-9)
            process.interrupt("cancel")
            try:
                yield process
            except simpy.Interrupt:
                pass

        env.process(cancel())
        env.run()
        self.assertEqual(after.cached_context, 16)
        self.assertFalse(resources.scheduler.owners)
        self.assertFalse(resources.scheduler.waiters)
        with self.assertRaises(ConfigurationError):
            plan_handoff(
                source,
                before,
                target,
                after,
                context=25,
                mode="full",
                cross_network="cross",
                replication_network="replication",
            )
