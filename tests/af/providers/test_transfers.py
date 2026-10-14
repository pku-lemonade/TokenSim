"""Af transfer provider."""

from __future__ import annotations

from dataclasses import replace
import unittest

from TokenSim.af.providers.operator_table import TopologyAFTransferProvider
from TokenSim.af.providers.queries import AF_MATCH_TYPES
from TokenSim.config.cluster.core import ClusterConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.config import _cluster_dict
from tests.af.support.hardware import af_test_hardware
from tests.af.support.queries import _transfer_query


class TopologyTransferProviderTest(unittest.TestCase):
    def setUp(self):
        self.cluster = ClusterConfig(
            **_cluster_dict(attention_workers=2, ffn_workers=2)
        )
        self.hardware = af_test_hardware()
        self.workers = self.cluster.workers()
        self.placement = build_topology_placement(
            self.cluster, self.workers, self.hardware
        )
        self.provider = TopologyAFTransferProvider(
            self.hardware, self.cluster, self.workers, self.placement, dtype="fp16"
        )

    def test_cross_domain_transfer_uses_node_aggregate_budget(self):
        estimate = self.provider.estimate(
            _transfer_query("af_a_to_f", payload_bytes=4096)
        )
        self.assertAlmostEqual(estimate.service_duration, 4096 / 300e9)
        self.assertEqual(estimate.payload_bytes, 4096)
        self.assertEqual(estimate.match_type, "analytical")
        self.assertEqual(estimate.details["model_version"], "node-domain-fifo-v1")
        back = self.provider.estimate(_transfer_query("af_f_to_a", payload_bytes=4096))
        self.assertEqual(back.service_duration, estimate.service_duration)
        self.assertEqual(
            self.provider.estimate(_transfer_query(payload_bytes=0)).service_duration,
            0.0,
        )

    def test_tp_collective_uses_the_collective_model_or_a_ring_fallback(self):
        query = _transfer_query(
            "tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        placed = self.provider.estimate(query)
        self.assertEqual(placed.details["model_version"], "node-domain-fifo-v1")
        self.assertGreater(placed.service_duration, 0)
        self.assertIn(placed.match_type, AF_MATCH_TYPES)
        with self.assertRaisesRegex(ConfigurationError, "requires placement"):
            TopologyAFTransferProvider(self.hardware, self.cluster, self.workers, None)
        values = _cluster_dict(attention_workers=2, ffn_workers=2)
        values["af"]["transfer_topology_mode"] = "flat_link"
        unplaced = TopologyAFTransferProvider(
            self.hardware, ClusterConfig(**values), self.workers, None
        )
        ring = unplaced.estimate(query)
        self.assertEqual(ring.details["algorithm"], "ring")
        self.assertGreater(ring.service_duration, 0)
        self.assertEqual(
            self.provider.estimate(replace(query, payload_bytes=0)).service_duration,
            0.0,
        )

    def test_unknown_endpoint_cannot_silently_use_flat_link(self):
        for payload in (0, 8192):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(
                    ConfigurationError,
                    "participants must be nonempty, distinct and mapped",
                ),
            ):
                self.provider.estimate(
                    _transfer_query(
                        payload_bytes=payload, endpoints=("a-pool:0", "missing:0")
                    )
                )

    def test_physical_endpoints_use_node_domains_not_rack_hops(self):
        from TokenSim.hardware.topology import TopologyLevel
        from TokenSim.hardware.topology import TopologyPlacement
        from TokenSim.hardware.topology import TopologySpec

        topology = TopologySpec(
            "three-level-af",
            (
                TopologyLevel("node", 8, "nvlink-test", links_per_endpoint=2),
                TopologyLevel("rack", 2, "ethernet-test"),
                TopologyLevel("cluster", None, "ethernet-test", oversubscription=2),
            ),
        )
        self.cluster.af.communication_domains = replace(
            self.cluster.af.communication_domains, node_count=4
        )
        times = []
        for index in (1, 8, 16):
            placement = TopologyPlacement(topology, {0: 0, 1: 2, 2: index, 3: 24})
            provider = TopologyAFTransferProvider(
                self.hardware, self.cluster, self.workers, placement
            )
            plan = provider.communication_plan(_transfer_query(payload_bytes=1 << 20))
            self.assertEqual(len(plan.phases[0].resource_ids), 1 if index == 1 else 2)
            times.append(
                provider.estimate(
                    _transfer_query(payload_bytes=1 << 20)
                ).service_duration
            )
        self.assertLess(times[0], times[1])
        self.assertEqual(times[1], times[2])

    def test_separated_collective_removes_software_launch_cost(self):
        from TokenSim.config.af.communication import AFCommunicationTimingConfig

        query = _transfer_query(
            "tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        legacy = self.provider.estimate(query)
        self.cluster.af.communication_timing = AFCommunicationTimingConfig(
            enabled=True, host_submit_us=5
        )
        # A provider pins its configuration; reconfiguration creates a new one.
        self.assertIs(self.provider.estimate(query), legacy)
        separated = TopologyAFTransferProvider(
            self.hardware, self.cluster, self.workers, self.placement, dtype="fp16"
        ).estimate(query)
        self.assertEqual(separated.details["model_version"], "node-domain-fifo-v1")
        self.assertEqual(separated.details["timing_scope"], "transport_service")
        self.assertEqual(legacy.service_duration, separated.service_duration)

    def test_whole_collective_measurement_is_not_split_into_phases(self):
        query = _transfer_query(
            "tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        estimate = self.provider.estimate(query)
        self.assertEqual(estimate.details["timing_scope"], "transport_service")
        self.assertEqual(estimate.match_type, "analytical")
        measured_only = TopologyAFTransferProvider(
            self.hardware,
            self.cluster,
            self.workers,
            self.placement,
            fallback="table_only",
        )
        with self.assertRaisesRegex(ConfigurationError, "phase transport_service"):
            measured_only.estimate(query)

    def test_unknown_network_or_operation_fail(self):
        values = _cluster_dict()
        values["networks"] = {"af-net": "no-such-link"}
        broken = ClusterConfig(**values)
        with self.assertRaisesRegex(ConfigurationError, "not a link class"):
            TopologyAFTransferProvider(self.hardware, broken, broken.workers(), None)
        with self.assertRaisesRegex(
            ValueError, "unsupported AF communication operation"
        ):
            self.provider.estimate(replace(_transfer_query(), operation="broadcast"))
        with self.assertRaisesRegex(ValueError, "requires explicit payload_bytes"):
            self.provider.estimate(replace(_transfer_query(), payload_bytes=None))
        with self.assertRaisesRegex(ConfigurationError, "not declared"):
            self.provider.estimate(_transfer_query(network="ghost-net"))
