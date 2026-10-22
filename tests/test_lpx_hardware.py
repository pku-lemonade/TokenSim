"""LP30 catalog identity, directional budgets and installed data coverage."""

from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from TokenSim.hardware import HardwareContext
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware.placement import build_topology_placement
from scripts.run_lpx import main

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
        for link in (aggregate, individual):
            self.assertEqual(link.half_bandwidth_bytes, 0)
            self.assertEqual(link.collective_efficiency, 1)
            self.assertEqual(link.transfer_us(1024), link.analytical_transfer_us(1024))
        self.assertEqual(individual.bandwidth_per_direction_bytes_per_s, 112e9 / 8)
        topology = self.hardware.topology("groq_lpx256")
        self.assertEqual(topology.levels[0].size * topology.levels[1].size, 256)
        self.assertTrue(all(level.links_per_endpoint == 1 for level in topology.levels))

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
            "data/models/gemma-4-31b-text.yaml",
            "data/topologies/groq_lpx256.yaml",
            "data/topologies/rubin_nvl72.yaml",
        ):
            self.assertIn(path, shipped)


class NativeCLITest(unittest.TestCase):
    def test_decode_outputs_and_restart_reuse_native_points(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            arguments = [
                "--model",
                "moe-toy",
                "--tp",
                "2",
                "--pp",
                "1",
                "--batches",
                "1",
                "2",
                "--context",
                "8",
                "--steps",
                "2",
                "--trace",
                "off",
                "--output",
                str(output),
            ]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(arguments), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["results"]), 2)
            self.assertEqual(len({r["checkpoint_key"] for r in summary["results"]}), 2)
            for row in summary["results"]:
                checkpoint = output / row["checkpoint_directory"]
                point = json.loads((checkpoint / "result.json").read_text())
                self.assertEqual(len(point["iterations"]), 2)
                self.assertGreater(row["tokens_per_s"], 0)
                self.assertIsNone(row["profile"])
            with (
                redirect_stdout(StringIO()),
                patch(
                    "TokenSim.af.factory.simulate_decode",
                    side_effect=AssertionError("reran"),
                ),
            ):
                self.assertEqual(main(arguments), 0)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)

    def test_invalid_partition_does_not_create_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(
                    [
                        "--model",
                        "moe-toy",
                        "--tp",
                        "2",
                        "--pp",
                        "2",
                        "--pipeline-layer-counts",
                        "1",
                        "2",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())
