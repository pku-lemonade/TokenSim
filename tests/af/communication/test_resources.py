from __future__ import annotations

from pathlib import Path
import unittest

import simpy

from TokenSim.af.communication.resources import AFResourceRegistry
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError


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
