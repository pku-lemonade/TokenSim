"""Communication resource, volume and readiness regression contracts."""

import unittest

import simpy

from TokenSim.af.domain_scheduler import DomainScheduler
from TokenSim.comm.traffic import collective_plan, ring_volume
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig, transmission_service


class VolumeTest(unittest.TestCase):
    def test_numeric_tensor_identity_across_hierarchical_phases(self):
        from tests.support.collective import execute_hierarchy

        for nodes in (
            {rank: 0 for rank in range(4)},
            {rank: rank // 4 for rank in range(8)},
            {rank: rank % 2 for rank in range(8)},
        ):
            ranks = tuple(nodes)
            for operation in ("all_gather", "all_reduce", "reduce_scatter"):
                inputs = {
                    rank: [rank * 100 + index for index in range(16)] for rank in ranks
                }
                plan = collective_plan(operation, ranks, 32, nodes)
                output = execute_hierarchy(plan, inputs)
                if operation == "all_gather":
                    expected = [value for rank in ranks for value in inputs[rank]]
                    self.assertTrue(all(value == expected for value in output.values()))
                else:
                    expected = [
                        sum(inputs[rank][index] for rank in ranks)
                        for index in range(16)
                    ]
                    if operation == "all_reduce":
                        self.assertTrue(
                            all(value == expected for value in output.values())
                        )
                    else:
                        chunk = 16 // len(ranks)
                        self.assertEqual(
                            output,
                            {
                                rank: expected[index * chunk : (index + 1) * chunk]
                                for index, rank in enumerate(ranks)
                            },
                        )

    def test_per_rank_and_node_totals(self):
        from TokenSim.comm.collectives import ring_traffic_factor

        for count in (1, 2, 4):
            for operation, expected in (
                ("all_gather", (count - 1) * 100),
                ("reduce_scatter", (count - 1) * 100 / count),
                ("all_reduce", 2 * (count - 1) * 100 / count),
            ):
                self.assertEqual(
                    ring_volume(operation, count, 100).send_bytes_per_rank, expected
                )
            self.assertEqual(
                ring_traffic_factor("all_gather", count) * count * 100,
                ring_volume("all_gather", count, 100).send_bytes_per_rank,
            )
        nodes = {rank: 0 for rank in range(4)}
        plan = collective_plan("all_gather", tuple(nodes), 100, nodes)
        self.assertEqual(plan.wire_bytes, 1200)
        self.assertAlmostEqual(plan.service(CommunicationDomainConfig()), 1200 / 300e9)

    def test_invalid_inputs_and_partial_nodes(self):
        for ranks, size, nodes in (
            ((), 16, {}),
            ((0, 0), 16, {0: 0}),
            ((0, 1), 16, {0: 0}),
            ((0, 1, 2), 16, {0: 0, 1: 0, 2: 1}),
        ):
            with self.assertRaises(ConfigurationError):
                collective_plan("all_reduce", ranks, size, nodes)
        for size in (-1, 1.5, True, float("nan")):
            with self.assertRaises(ConfigurationError):
                ring_volume("all_reduce", 2, size)
        with self.assertRaises(ConfigurationError):
            ring_volume("reduce_scatter", 4, 100, "bf16")
        singleton = collective_plan("all_reduce", (0,), 100, {0: 0})
        self.assertEqual(singleton.service(CommunicationDomainConfig()), 0)


class DomainFIFOTest(unittest.TestCase):
    def test_atomic_wait_does_not_hold_source_or_allow_bypass(self):
        env = simpy.Environment()
        scheduler = DomainScheduler(env, ("tx0", "rx1", "rx2", "other"))
        times = {}

        def task(name, keys, duration, delay=0):
            yield env.timeout(delay)
            times[name] = yield from scheduler.run(keys, duration, (name,))

        env.process(task("busy", ("rx1",), 10))
        env.process(task("a", ("tx0", "rx1"), 3, 1))
        env.process(task("b", ("tx0", "rx2"), 2, 2))
        env.process(task("disjoint", ("other",), 2, 2))

        def observe():
            yield env.timeout(3)
            self.assertNotIn("tx0", scheduler.owners)

        env.process(observe())
        env.run()
        self.assertEqual(times["a"], (1, 10, 13))
        self.assertEqual(times["b"], (2, 13, 15))
        self.assertEqual(times["disjoint"], (2, 2, 4))

    def test_cancellation_releases_all_resources(self):
        for waiting in (False, True):
            env = simpy.Environment()
            scheduler = DomainScheduler(env, ("tx", "rx"))
            holder = scheduler.request(("rx",), ("holder",)) if waiting else None
            results = []

            def victim():
                try:
                    yield from scheduler.run(("tx", "rx"), 10, ("victim",))
                except simpy.Interrupt:
                    pass

            process = env.process(victim())

            def cancel():
                yield env.timeout(1)
                process.interrupt()
                yield process
                if holder:
                    scheduler.release(holder)
                results.append((yield from scheduler.run(("tx", "rx"), 1, ("next",))))

            env.process(cancel())
            env.run()
            self.assertEqual(results, [(1, 1, 2)])
            self.assertFalse(scheduler.owners)
            self.assertFalse(scheduler.waiters)

    def test_source_target_minimum_and_local_independence(self):
        config = CommunicationDomainConfig(
            inter_tx_bandwidth_bytes_per_s=50e9, inter_rx_bandwidth_bytes_per_s=25e9
        )
        self.assertEqual(transmission_service(config, 0, 1, 25000), 1e-6)
        self.assertEqual(transmission_service(config, 0, 0, 300000), 1e-6)
        self.assertEqual(
            CommunicationDomainConfig(
                inter_tx_bandwidth_bytes_per_s="400 Gb/s"
            ).inter_tx_bandwidth_bytes_per_s,
            50e9,
        )
        self.assertEqual(
            CommunicationDomainConfig(
                inter_tx_bandwidth_bytes_per_s="50 GB/s"
            ).inter_tx_bandwidth_bytes_per_s,
            50e9,
        )
        for value in (True, -1, float("inf"), "300"):
            with self.assertRaises((ConfigurationError, ValueError)):
                CommunicationDomainConfig(local_bandwidth_bytes_per_s=value)
        with self.assertRaises(ValueError):
            CommunicationDomainConfig(parallel_links=6)


class PlacementTest(unittest.TestCase):
    def test_automatic_and_override_placement_is_reproducible(self):
        from tests.support.af_hardware import af_test_hardware
        from tests.support.af_parallel import _cluster
        from TokenSim.hardware.placement import build_topology_placement

        cluster = _cluster(a_tp=2, a_dp=1, f_tp=6, f_dp=1)
        for strategy, nodes in (("mixed", 1), ("dedicated", 2)):
            cluster.af.communication_domains = CommunicationDomainConfig(
                node_count=nodes,
                placement_strategy=strategy,
                device_indices={"a-pool:0": 1},
            )
            first = build_topology_placement(
                cluster, cluster.workers(), af_test_hardware()
            )
            second = build_topology_placement(
                cluster, cluster.workers(), af_test_hardware()
            )
            self.assertEqual(first, second)
            self.assertEqual(first.device_index(0), 1)
            self.assertEqual(len(set(first.device_index_by_worker.values())), 8)
            if strategy == "dedicated":
                self.assertTrue(
                    all(first.device_index(rank) // 8 == 1 for rank in range(2, 8))
                )

    def test_policy_conflicts_and_inventory_shortage_fail(self):
        from tests.support.af_hardware import af_test_hardware
        from tests.support.af_parallel import _cluster
        from TokenSim.hardware.placement import build_topology_placement

        cluster = _cluster(a_tp=2, a_dp=1, f_tp=6, f_dp=1)
        for config in (
            CommunicationDomainConfig(node_count=1, placement_strategy="dedicated"),
            CommunicationDomainConfig(
                node_count=1, placement_strategy="mixed", device_indices={"f-pool:0": 0}
            ),
        ):
            cluster.af.communication_domains = config
            with self.assertRaises(ConfigurationError):
                build_topology_placement(cluster, cluster.workers(), af_test_hardware())
