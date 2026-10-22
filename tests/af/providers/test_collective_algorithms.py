"""Collective policies survive config, compilation, boundaries and caches."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.providers.operator_table import TopologyAFTransferProvider
from TokenSim.af.serving.service import AFService
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.errors import ConfigurationError
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration
from tests.af.support.config import _cluster_dict
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _transfer_query
from tests import test_lpx_native
from tests.af.support.pipeline import cluster_document


class CollectivePolicyTest(unittest.TestCase):
    def provider(self, mode="flat_link"):
        doc = _cluster_dict(attention_workers=2, ffn_workers=2)
        doc["af"]["transfer_topology_mode"] = mode
        cluster = ClusterConfig(**doc)
        hardware = af_test_hardware()
        from TokenSim.hardware.placement import build_topology_placement

        workers = cluster.workers()
        return TopologyAFTransferProvider(
            hardware,
            cluster,
            workers,
            build_topology_placement(cluster, workers, hardware),
        )

    def test_service_cache_separates_algorithms_and_does_not_hide_invalid_policy(self):
        provider = self.provider()
        query = _transfer_query(
            "all_reduce", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        ring = provider.estimate(query)
        tree = provider.estimate(replace(query, collective_algorithm="tree"))
        self.assertEqual(ring.details["algorithm"], "ring")
        self.assertEqual(tree.details["algorithm"], "tree")
        self.assertIsNot(tree, ring)
        self.assertIs(
            tree, provider.estimate(replace(query, collective_algorithm="tree"))
        )
        with self.assertRaises(ConfigurationError):
            provider.estimate(replace(query, collective_algorithm="invalid"))

    def test_physical_wire_plans_cannot_silently_price_another_algorithm(self):
        provider = self.provider("physical")
        query = _transfer_query(
            "all_reduce", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        provider.estimate(query)
        for policy in ("tree", "auto", "direct"):
            with (
                self.subTest(policy=policy),
                self.assertRaisesRegex(ConfigurationError, "ring only"),
            ):
                provider.estimate(replace(query, collective_algorithm=policy))

    def test_layout_and_domain_reject_unknown_algorithm(self):
        with self.assertRaises(ConfigurationError):
            Layout(a_collective_algorithm="invalid")
        with self.assertRaises(ConfigurationError):
            DomainCollectiveConfig(
                "all_reduce", "a_compute", (0, 1), algorithm="invalid"
            )

    def test_layer_and_model_boundary_collectives_use_compiled_policy(self):
        test_lpx_native.NativeLPXTest.setUpClass()
        fixture = test_lpx_native.NativeLPXTest()
        layout = replace(fixture.base, a_collective_algorithm="direct")
        runtime = configuration(
            fixture.hardware, fixture.model, layout, namespace="policy"
        )
        self.assertTrue(
            all(
                p.algorithm == "direct"
                for p in runtime.plan.collective_plans
                if p.domain == "policy-a-domain"
            )
        )
        self.assertTrue(
            all(
                p.as_dict()["algorithm"] == "direct"
                for p in runtime.plan.collective_plans
                if p.domain == "policy-a-domain"
            )
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        env.run(until=env.process(service.forward([session])))
        events = [
            e
            for e in service.trace.events
            if e.operation in {"all_reduce", "all_gather"}
        ]
        self.assertTrue(events)
        self.assertTrue(any("boundary" in e.node_id for e in events))
        self.assertTrue(any(":l0:" in e.node_id for e in events))
        self.assertEqual({e.details["algorithm"] for e in events}, {"direct"})
        self.assertEqual({e.details["requested_algorithm"] for e in events}, {"direct"})
        doc = cluster_document(runtime.cluster)
        doc["af"]["transfer_topology_mode"] = "physical"
        doc["af"]["communication_resource_scope"] = "fabric"
        doc["af"]["endpoint_transfer_occupancy"] = "service"
        with self.assertRaisesRegex(ConfigurationError, "wire schedule"):
            build_model_runtime(doc, hardware=fixture.hardware, model=fixture.model)
