"""Host/transport timing, physical resources and fabric contention."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.communication.resources import AFResourceRegistry
from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.af.communication import AFCommunicationTimingConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.support.config import _cluster
from tests.af.support.execution import af_test_resources, execution_plan


class ZeroCompute:
    def estimate(self, query):
        return AFServiceEstimate(0.0, "test:zero-compute")


class WireOnly:
    def __init__(self, duration_us=3.0, scope="transport_service"):
        self.duration_us, self.scope = duration_us, scope

    def estimate(self, query):
        return AFServiceEstimate(
            self.duration_us / 1e6,
            "test:wire",
            payload_bytes=query.payload_bytes,
            details={"timing_scope": self.scope},
        )


def run_case(
    submit_us=5.0,
    complete_us=0.0,
    wire_us=3.0,
    scope="per_worker",
    enabled=True,
    provider_scope="transport_service",
    duplex="full",
):
    cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1)
    cluster.af.communication_timing = AFCommunicationTimingConfig(
        enabled=enabled,
        host_submit_us=submit_us,
        host_completion_us=complete_us,
        host_resource_scope=scope,
    )
    cluster.af.transfer_topology_mode = "flat_link"
    # Construct from normalized JSON to recompute the actual fabric configuration.
    if duplex == "half":
        from pydantic import TypeAdapter
        from TokenSim.config.cluster_config import ClusterConfig

        doc = TypeAdapter(ClusterConfig).dump_python(
            cluster, mode="json", exclude_none=True
        )
        doc["af"]["links"][0].update(duplex="half", half_duplex_capacity=1)
        doc["af"]["links"][0].pop("a_to_f_capacity", None)
        doc["af"]["links"][0].pop("f_to_a_capacity", None)
        doc["af"]["communication_fabrics"] = []
        cluster = ClusterConfig(**doc)
    env = simpy.Environment()
    executor = AFExecutor(
        env,
        cluster,
        cluster.workers(),
        ZeroCompute(),
        WireOnly(wire_us, provider_scope),
        "test",
        "bf16",
        total_layers=1,
        hidden_size=16,
        runtime_plan=execution_plan(cluster),
        resources=af_test_resources(env, cluster, cluster.workers()),
    )
    requests = [
        SimpleNamespace(
            id=i, is_prefill=True, prefill_len=1, prefill_compute_len=1, context_len=1
        )
        for i in range(2)
    ]
    iteration = executor.submit(0, requests, "prefill")
    env.process(executor.run_iteration(iteration))
    env.run()
    executor.trace.validate_invariants(iteration.iteration_id)
    return executor, iteration


class CommunicationTimingTest(unittest.TestCase):
    def test_host_submission_and_wire_overlap_without_double_counting(self):
        executor, iteration = run_case()
        sends = [
            e
            for e in executor.trace.events
            if e.event_type == "transfer" and e.direction == "a_to_f"
        ]
        submits = [
            e
            for e in executor.trace.events
            if e.event_type == "host_submit" and e.direction == "a_to_f"
        ]
        self.assertEqual(len(sends), 2)
        self.assertEqual(len(submits), 2)
        self.assertAlmostEqual(submits[0].start_at, 0)
        self.assertAlmostEqual(submits[0].end_at, 5e-6)
        self.assertAlmostEqual(submits[1].start_at, 5e-6)
        self.assertAlmostEqual(submits[1].queue_duration, 5e-6)
        self.assertAlmostEqual(sends[0].start_at, 5e-6)
        self.assertAlmostEqual(sends[0].end_at, 8e-6)
        self.assertAlmostEqual(sends[1].start_at, 10e-6)
        self.assertAlmostEqual(iteration.makespan, 21e-6)
        syncs = [e for e in executor.trace.events if e.event_type == "synchronize"]
        self.assertEqual(len(syncs), 4)
        self.assertTrue(all(e.service_duration == 0 for e in syncs))
        self.assertAlmostEqual(sum(e.service_duration for e in sends), 6e-6)

    def test_wire_queue_changes_wait_not_fixed_cost(self):
        executor, _ = run_case(submit_us=1.0, wire_us=10.0)
        sends = [
            e
            for e in executor.trace.events
            if e.event_type == "transfer" and e.direction == "a_to_f"
        ]
        self.assertAlmostEqual(sends[0].start_at, 1e-6)
        self.assertAlmostEqual(sends[1].start_at, 11e-6)
        self.assertAlmostEqual(sends[1].queue_duration, 9e-6)
        for event in sends:
            self.assertAlmostEqual(event.service_duration, 10e-6)
        submits = [e for e in executor.trace.events if e.event_type == "host_submit"]
        self.assertTrue(all(abs(e.service_duration - 1e-6) < 1e-12 for e in submits))
        sync = next(
            e
            for e in executor.trace.events
            if e.node_id == sends[1].node_id.removesuffix(":transport")
        )
        self.assertAlmostEqual(sync.details["dependency_wait_duration"], 19e-6)
        self.assertEqual(sync.service_duration, 0.0)

    def test_completion_cost_occupies_host_not_wire(self):
        executor, _ = run_case(complete_us=2.0)
        syncs = [e for e in executor.trace.events if e.event_type == "synchronize"]
        self.assertTrue(all(abs(e.service_duration - 2e-6) < 1e-12 for e in syncs))
        self.assertTrue(all(e.resource_id.startswith("host:") for e in syncs))
        self.assertTrue(
            all(
                abs(e.service_duration - 3e-6) < 1e-12
                for e in executor.trace.events
                if e.event_type == "transfer"
            )
        )

    def test_shared_host_serializes_submissions_from_both_endpoints(self):
        per_worker, per_iteration = run_case()
        shared, shared_iteration = run_case(scope="shared")
        self.assertGreater(shared_iteration.makespan, per_iteration.makespan)
        self.assertEqual(
            {
                e.resource_id
                for e in shared.trace.events
                if e.event_type == "host_submit"
            },
            {"host:shared"},
        )

    def test_legacy_behavior_and_zero_host_cost_are_preserved(self):
        legacy, old = run_case(enabled=False, provider_scope="legacy_unspecified")
        separated, new = run_case(submit_us=0.0)
        self.assertAlmostEqual(old.makespan, new.makespan)
        self.assertFalse(
            any(e.event_type == "host_submit" for e in legacy.trace.events)
        )
        self.assertTrue(
            any(e.event_type == "host_submit" for e in separated.trace.events)
        )

    def test_half_duplex_still_shares_only_the_wire(self):
        executor, _ = run_case(submit_us=1.0, wire_us=10.0, duplex="half")
        wires = [e for e in executor.trace.events if e.event_type == "transfer"]
        self.assertEqual({e.resource_id for e in wires}, {"link:af-fabric:shared"})

    def test_ambiguous_measurements_are_rejected_in_separated_mode(self):
        for scope in ("legacy_unspecified", "eager_interval"):
            with (
                self.subTest(scope=scope),
                self.assertRaisesRegex(SimulationStateError, "timing_scope"),
            ):
                run_case(provider_scope=scope)

    def test_invalid_fixed_costs_are_rejected(self):
        for value in (-1.0, float("nan"), float("inf"), True, "5"):
            with (
                self.subTest(value=value),
                self.assertRaises((ConfigurationError, ValueError)),
            ):
                AFCommunicationTimingConfig(host_submit_us=value)

    def test_formal_mla_collectives_keep_completion_dependencies(self):
        from scripts.run_deepseek_mla import configuration

        config = configuration(a_tp=2, f_tp=2)
        cluster, shape, moe, plan = (
            config.cluster,
            config.model,
            config.model.moe,
            config.plan,
        )
        cluster.af.communication_timing = AFCommunicationTimingConfig(
            enabled=True, host_submit_us=5.0
        )
        env = simpy.Environment()
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            ZeroCompute(),
            WireOnly(),
            shape.model_id,
            "bf16",
            total_layers=1,
            hidden_size=shape.hidden_size,
            moe_config=moe,
            runtime_plan=plan,
            resources=af_test_resources(env, cluster, cluster.workers()),
        )
        request = SimpleNamespace(
            id=0, is_prefill=False, context_len=4096, generation_idx=0
        )
        iteration = executor.submit(0, [request], "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        executor.trace.validate_invariants(iteration.iteration_id)
        events = {e.node_id: e for e in executor.trace.events}
        collectives = [e for e in events.values() if e.event_type == "collective"]
        self.assertTrue(any(e.operation == "all_gather" for e in collectives))
        for wire in collectives:
            complete = events[wire.node_id.removesuffix(":transport")]
            self.assertEqual(complete.event_type, "synchronize")
            self.assertGreaterEqual(complete.start_at, wire.end_at)


class AFResourceRegistryTest(unittest.TestCase):
    @staticmethod
    def _registry():
        root = Path(__file__).resolve().parents[3]
        cluster = ClusterConfig.from_file(
            root / "data" / "clusters" / "af" / "decode_tp2_dp2_ep4_rank_reuse.json"
        )
        cluster.af.transfer_topology_mode = "flat_link"
        return cluster, AFResourceRegistry(
            simpy.Environment(), cluster, cluster.workers()
        )

    def test_transfer_direction_resolves_its_link_lane(self):
        cluster, registry = self._registry()
        link = cluster.phase_to_af_link["decode"]

        a_to_f = registry.communication_resources_for_step(link, "a_to_f")
        f_to_a = registry.communication_resources_for_step(link, "f_to_a")

        self.assertEqual(a_to_f[0][0], (link.resource_group, "a_to_f"))
        self.assertEqual(f_to_a[0][0], (link.resource_group, "f_to_a"))
        self.assertIsNot(a_to_f[0][1], f_to_a[0][1])

    def test_collective_uses_only_its_explicit_lane(self):
        cluster, registry = self._registry()
        link = cluster.phase_to_af_link["decode"]
        a_domain, _ = cluster.phase_to_af_domains["decode"]
        plan = a_domain.communication_plan[0]

        collective = registry.communication_resources_for_step(
            link,
            plan.operation,
            plan.resource_group,
            plan.resource_lane,
        )

        self.assertEqual(collective[0][0], (plan.resource_group, plan.resource_lane))
        self.assertIsNot(
            collective[0][1],
            registry.communication_resources_for_step(link, "a_to_f")[0][1],
        )

    def test_collective_lane_is_never_inferred(self):
        cluster, registry = self._registry()
        link = cluster.phase_to_af_link["decode"]

        with self.assertRaisesRegex(ConfigurationError, "explicit resource_lane"):
            registry.communication_resources_for_step(
                link, "all_reduce", "collective-fabric"
            )


if __name__ == "__main__":
    unittest.main()
