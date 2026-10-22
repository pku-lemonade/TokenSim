"""Flat collective arithmetic, compiled policies and physical-mode rejection."""

import simpy
import unittest
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.providers.operator_table import TopologyAFTransferProvider
from TokenSim.af.serving.service import AFService
from TokenSim.comm.flat_collectives import flat_collective
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.links import LinkClass
from dataclasses import replace
from scripts.lpx_runtime import Layout, configuration
from tests.af.support.config import _cluster_dict
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _transfer_query
from tests.af.support.pipeline import cluster_document
from tests.support.lpx import tiny_moe


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
        from TokenSim.hardware import HardwareContext
        from pathlib import Path

        hardware = HardwareContext.load(Path(__file__).resolve().parents[3] / "data")
        model = tiny_moe(hardware)
        layout = Layout(
            a_tp=2,
            f_ep=2,
            a_collective_algorithm="direct",
            f_collective_algorithm="direct",
        )
        runtime = configuration(hardware, model, layout, namespace="policy")
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
            build_model_runtime(doc, hardware=hardware, model=model)


class FlatCollectiveTest(unittest.TestCase):
    def setUp(self):
        self.link = LinkClass("flat", "test fabric", 100e9, 0.5)

    def estimate(self, operation="all_reduce", size=8192, algorithm="ring", ranks=8):
        return flat_collective(
            operation, ranks, size, self.link, algorithm=algorithm, dtype="bf16"
        )

    def test_ring_accounts_for_every_neighbor_round(self):
        value = self.estimate()
        self.assertEqual(value.detail["rounds"], 14)
        self.assertEqual(value.detail["wire_bytes"], 2 * 7 * 1024)
        self.assertEqual(value.latency_us, 14 * 0.5 + 14336 / 100e9 * 1e6)

    def test_auto_switches_from_startup_to_bandwidth_cost(self):
        small = self.estimate(algorithm="auto")
        large = self.estimate(size=8 << 20, algorithm="auto")
        self.assertEqual(small.detail["algorithm"], "tree")
        self.assertEqual(small.detail["rounds"], 6)
        self.assertEqual(large.detail["algorithm"], "ring")
        self.assertNotIn("direct", small.detail["algorithm_candidates"])

    def test_all_gather_uses_local_input_and_full_output(self):
        for algorithm, rounds in (("ring", 7), ("direct", 1)):
            with self.subTest(algorithm=algorithm):
                value = self.estimate("all_gather", 1024, algorithm)
                self.assertEqual(value.detail["input_bytes_per_rank"], 1024)
                self.assertEqual(value.detail["output_bytes_per_rank"], 8192)
                self.assertEqual(value.detail["wire_bytes"], 7168)
                self.assertEqual(value.detail["rounds"], rounds)
        self.assertEqual(
            self.estimate("reduce_scatter").detail["output_bytes_per_rank"], 1024
        )

    def test_empty_and_singleton_collectives_have_no_network_service(self):
        for algorithm in ("ring", "tree", "direct", "auto"):
            self.assertEqual(self.estimate(size=0, algorithm=algorithm).latency_us, 0)
            self.assertEqual(self.estimate(ranks=1, algorithm=algorithm).latency_us, 0)

    def test_link_efficiency_ramp_is_not_a_transport_startup_cost(self):
        first = self.estimate(algorithm="tree")
        self.link = replace(self.link, half_bandwidth_bytes=0, collective_efficiency=1)
        self.assertEqual(first, self.estimate(algorithm="tree"))

    def test_invalid_policy_and_fractional_dtype_shards_are_rejected(self):
        for algorithm in ("unknown", None, True):
            with self.assertRaises(ConfigurationError):
                self.estimate(algorithm=algorithm)
        with self.assertRaisesRegex(ConfigurationError, "integer number"):
            self.estimate(size=8194)
        with self.assertRaises(ConfigurationError):
            self.estimate(ranks=0)
