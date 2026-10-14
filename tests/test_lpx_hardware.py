"""LP30 catalog identity, directional budgets and installed data coverage."""

from pathlib import Path
import tomllib
import unittest

from TokenSim.hardware import HardwareContext
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware.placement import build_topology_placement

ROOT = Path(__file__).resolve().parents[1]


class LPXHardwareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")

    def test_lp30_is_distinct_from_first_generation_groq(self):
        lpu = self.hardware.device("LP30")
        self.assertEqual(lpu.device_id, "groq_lp30")
        self.assertEqual(self.hardware.device("groq").device_id, "groqchip_v1")
        self.assertEqual(lpu.memory_capacity_bytes.value, 500_000_000)
        self.assertEqual(lpu.on_chip_memory_bytes, lpu.memory_capacity_bytes)
        self.assertEqual(lpu.memory_bandwidth_bytes_per_s.value, 150e12)
        self.assertEqual(lpu.peak_compute_for("fp8").value * 256, 315e15)
        self.assertEqual(lpu.extra["power_source"], "assumption-lp30-analytical")

    def test_aggregate_link_is_not_multiplied_by_port_count(self):
        aggregate = self.hardware.link("groq_lp30_injection")
        individual = self.hardware.link("groq_lp30_c2c_112g")
        self.assertEqual(aggregate.bandwidth_per_direction_bytes_per_s, 1.25e12)
        self.assertEqual(individual.bandwidth_per_direction_bytes_per_s, 112e9 / 8)
        topology = self.hardware.topology("groq_lpx256")
        self.assertEqual(topology.levels[0].size * topology.levels[1].size, 256)
        self.assertTrue(all(level.links_per_endpoint == 1 for level in topology.levels))

    def test_scaled_model_has_explicit_proxy_identity(self):
        model = self.hardware.model("gpt-oss-2t-proxy")
        self.assertGreater(model.total_params(), 2e12)
        self.assertLess(model.total_params(), 2.02e12)
        self.assertEqual(model.sources["assumption-scaled-gpt-oss"]["grade"], "D")

    def test_automatic_lp30_topology_uses_one_injection_budget(self):
        cluster = ClusterConfig(
            num_workers=2,
            networks={"net": "ethernet1600Gb"},
            worker_groups=[
                dict(
                    hardware="groq_lp30",
                    num_workers=2,
                    network="net",
                    capabilities=["prefill", "decode_full"],
                )
            ],
        )
        placement = build_topology_placement(cluster, cluster.workers(), self.hardware)
        self.assertEqual(placement.topology.levels[0].links_per_endpoint, 1)

    def test_new_catalog_files_are_shipped(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        shipped = {
            path
            for paths in config["tool"]["setuptools"]["data-files"].values()
            for path in paths
        }
        for path in (
            "data/devices/groq_lp30.yaml",
            "data/devices/rubin.yaml",
            "data/models/gpt-oss-2t-proxy.yaml",
            "data/topologies/groq_lpx256.yaml",
            "data/topologies/rubin_nvl72.yaml",
        ):
            self.assertIn(path, shipped)
