"""Fixed eight-slot AF placement contracts at each physical entry point."""

from dataclasses import replace
import unittest

import simpy

from tests.test_af_executor import af_test_hardware
from tests.test_af_operator_provider import _transfer_query
from tests.test_af_parallel_planning import _cluster
from TokenSim.af.operator_provider import TopologyAFTransferProvider
from TokenSim.af.resources import AFResourceRegistry
from TokenSim.af.runtime import compile_af_runtime_plan
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.hardware.topology import TopologyPlacement, TopologySpec


def fixture(a=1, f=1, **domains):
    cluster = _cluster(a_tp=a, a_dp=1, f_tp=f, f_dp=1)
    cluster.af.communication_domains = CommunicationDomainConfig(node_count=2, **domains)
    hardware = af_test_hardware()
    workers = cluster.workers()
    placement = build_topology_placement(cluster, workers, hardware)
    return cluster, hardware, workers, placement


class PhysicalPlacementTests(unittest.TestCase):
    def entrances(self, cluster, hardware, workers, placement):
        return (
            lambda: build_topology_placement(cluster, workers, hardware),
            lambda: TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only'),
            lambda: AFResourceRegistry(simpy.Environment(), cluster, workers, placement),
            lambda: compile_af_runtime_plan(cluster_config=cluster, worker_configs=workers,
                hardware=hardware, model=hardware.model('TestModel'), block_size=4),
        )

    def test_legacy_network_grouping_requires_explicit_migration(self):
        cluster, hardware, workers, placement = fixture()
        cluster.af.communication_domains = None
        cluster.worker_groups[1].network = 'other-node'
        cluster.networks['other-node'] = 'test'
        workers = cluster.workers()
        legacy = TopologyPlacement(TopologySpec.two_level('legacy', node_size=1,
            node_link='nvlink-test', cluster_link='test'), {0: 0, 1: 1})
        for entrance in self.entrances(cluster, hardware, workers, legacy):
            with self.subTest(entrance=entrance), self.assertRaisesRegex(ConfigurationError, 'communication_domains'):
                entrance()

    def test_missing_inventory_and_non_eight_size_fail_at_all_entrances(self):
        cluster, hardware, workers, placement = fixture()
        for invalid in (CommunicationDomainConfig(), CommunicationDomainConfig(node_size=4, node_count=2)):
            cluster.af.communication_domains = invalid
            for entrance in self.entrances(cluster, hardware, workers, placement):
                with self.subTest(config=invalid, entrance=entrance), self.assertRaises(ConfigurationError):
                    entrance()

    def test_partial_nodes_keep_cross_node_service_cache_and_queue(self):
        for cache in (True, False):
            cluster, hardware, workers, placement = fixture(device_indices={'a-pool:0': 0, 'f-pool:0': 8})
            cluster.af.service_estimate_cache = cache
            provider = TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only')
            env = simpy.Environment()
            registry = AFResourceRegistry(env, cluster, workers, placement)
            query = _transfer_query(payload_bytes=300_000_000)
            self.assertEqual(placement.node_id(0), 0)
            self.assertEqual(placement.node_id(1), 1)
            self.assertEqual(dict(provider.node_by_worker), registry.node_by_worker)
            phase = provider.communication_plan(query).phases[0]
            self.assertEqual(phase.resource_ids, ('domain:node0:inter_tx', 'domain:node1:inter_rx'))
            first, second = provider.estimate(query), provider.estimate(replace(query, iteration_id=2))
            self.assertAlmostEqual(first.service_duration, .006)
            self.assertEqual(first.service_duration, second.service_duration)
            intervals = []
            def run(identity):
                result = yield from registry.domain_scheduler.run(phase.resource_ids, first.service_duration, (identity,))
                intervals.append(result)
            env.process(run(1))
            env.process(run(2))
            env.run()
            self.assertEqual(intervals, [(0, 0, .006), (0, .006, .012)])

    def test_shared_node_identity_covers_slot_boundaries(self):
        cluster, hardware, workers, _ = fixture(a=2, f=2)
        placement = TopologyPlacement(TopologySpec.two_level('boundary', node_size=8,
            node_link='nvlink-test', cluster_link='test'), {0: 0, 1: 7, 2: 8, 3: 15})
        provider = TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only')
        registry = AFResourceRegistry(simpy.Environment(), cluster, workers, placement)
        expected = {0: 0, 1: 0, 2: 1, 3: 1}
        self.assertEqual({rank: placement.node_id(rank) for rank in range(4)}, expected)
        self.assertEqual(dict(provider.node_by_worker), expected)
        self.assertEqual(registry.node_by_worker, expected)
        local = provider.communication_plan(_transfer_query(endpoints=('a-pool:0','a-pool:1')))
        self.assertEqual(local.phases[0].resource_ids, ('domain:node0:local',))

    def test_invalid_supplied_placement_is_rejected_by_both_consumers(self):
        cluster, hardware, workers, good = fixture()
        mappings = ({0: 0}, {0: 0, 1: 0}, {0: 0, 1: -1}, {0: 0, 1: 16},
                    {0: 0, 1: True}, {0: 0, 1: 1.0}, {0: 0, 1: 1, 2: 2})
        bad = [TopologyPlacement(good.topology, mapping) for mapping in mappings]
        bad.append(TopologyPlacement(TopologySpec.two_level('wrong-boundary', node_size=1,
            node_link='nvlink-test', cluster_link='test'), {0: 0, 1: 8}))
        bad.append(None)
        for placement in bad:
            for entrance in self.entrances(cluster, hardware, workers, placement)[1:3]:
                with self.subTest(placement=placement, entrance=entrance), self.assertRaises(ConfigurationError):
                    entrance()

    def test_node_interface_never_defaults_missing_mapping(self):
        _, _, _, placement = fixture()
        with self.assertRaises(ConfigurationError):
            placement.node_id(99)

    def test_mixed_dedicated_manual_keep_role_and_override_semantics(self):
        for strategy, expected in (('mixed', (0, 1, 2, 3, 4, 5, 6, 7)),
                                   ('dedicated', (0, 1, 8, 9, 10, 11, 12, 13)),
                                   ('manual', (0, 1, 2, 3, 4, 5, 6, 7))):
            cluster, hardware, workers, placement = fixture(a=2, f=6, placement_strategy=strategy)
            self.assertEqual(tuple(placement.device_index(i) for i in range(8)), expected)
            provider = TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only')
            self.assertEqual(provider.describe()['domain_config']['placement_strategy'], strategy)
        _, _, _, placement = fixture(device_indices={'a-pool:0': 7, 'f-pool:0': 15})
        self.assertEqual((placement.device_index(0), placement.device_index(1)), (7, 15))

    def test_supplied_placement_obeys_role_policies_and_explicit_overrides(self):
        for strategy in ('mixed','dedicated'):
            cluster, hardware, workers, good = fixture(placement_strategy=strategy)
            wrong = TopologyPlacement(good.topology, {0: 0, 1: 1})
            for entrance in self.entrances(cluster, hardware, workers, wrong)[1:3]:
                with self.subTest(strategy=strategy), self.assertRaises(ConfigurationError):
                    entrance()
        cluster, hardware, workers, good = fixture(device_indices={'f-pool:0': 8})
        wrong = TopologyPlacement(good.topology, {0: 0, 1: 1})
        for entrance in self.entrances(cluster, hardware, workers, wrong)[1:3]:
            with self.assertRaises(ConfigurationError):
                entrance()

    def test_legacy_topology_name_does_not_override_declared_domains(self):
        cluster, hardware, workers, _ = fixture(device_indices={'a-pool:0': 0, 'f-pool:0': 8})
        cluster.topology = 'nonexistent-legacy-name'
        placement = build_topology_placement(cluster, workers, hardware)
        self.assertEqual(placement.topology.levels[0].size, 8)
        self.assertEqual((placement.node_id(0), placement.node_id(1)), (0, 1))
        provider = TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only')
        self.assertEqual(provider.describe()['placement_source'], 'explicit:af.communication_domains')
        self.assertEqual(provider.describe()['worker_device_indices'], {0: 0, 1: 8})

    def test_non_af_and_flat_keep_legacy_topology_path(self):
        for enabled, mode in ((False, 'physical'), (True, 'flat_link')):
            cluster, hardware, _, _ = fixture()
            cluster.af.enabled, cluster.af.transfer_topology_mode = enabled, mode
            cluster.af.communication_domains = None
            cluster.worker_groups[1].network = 'other-node'
            cluster.networks['other-node'] = 'test'
            placement = build_topology_placement(cluster, cluster.workers(), hardware)
            self.assertEqual(placement.topology.levels[0].size, 1)
            self.assertEqual(tuple(placement.device_index(i) for i in range(2)), (0, 1))
            if enabled:
                # Flat pricing keeps its legacy index fallback, even without a mapping.
                unmapped = TopologyPlacement(placement.topology, {})
                provider = TopologyAFTransferProvider(hardware, cluster, cluster.workers(), unmapped,
                                                       fallback='analytical_only')
                registry = AFResourceRegistry(simpy.Environment(), cluster, cluster.workers(), unmapped)
                self.assertEqual(dict(provider.node_by_worker), {0: 0, 1: 0})
                self.assertEqual(dict(provider.node_by_worker), registry.node_by_worker)
                self.assertGreater(provider.estimate(_transfer_query()).service_duration, 0)


if __name__ == '__main__':
    unittest.main()
