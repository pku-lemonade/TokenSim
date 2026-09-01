from __future__ import annotations

import json
import unittest
from pathlib import Path

from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.parallel_config import ParallelConfig
from TransformerRoofline.roofline import Hardware, Link


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_MODELS_PATH = REPO_ROOT / "TransformerRoofline" / "hardware_models.json"
H100_CLUSTER_PATH = REPO_ROOT / "data" / "clusters" / "8_h100" / "h8.json"


class H100HardwareProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hardware_data = json.loads(HARDWARE_MODELS_PATH.read_text())

    def test_h100_preserves_calibration_and_exposes_expected_memory_and_links(self):
        profile_data = next(
            hardware
            for hardware in self.hardware_data["hardware"]
            if hardware["Name"] == "H100"
        )
        profile = Hardware(profile_data)

        self.assertEqual(profile.Capacity, 96)
        self.assertEqual(profile.Pcie, "pcie5.0x16")
        self.assertEqual(profile.Nvlink, "nvlinkx18")
        self.assertEqual(profile.TFLOPS, 1000)
        self.assertEqual(profile.BW_TBs, 3)
        self.assertEqual(profile.Static_Power, 0)
        self.assertEqual(profile.TDP, 0)
        self.assertEqual(profile.price, 240000)

    def test_h100_nvlink_resolves_expected_nvlink4_bandwidth(self):
        link_data = next(
            link
            for link in self.hardware_data["links"]
            if link["Name"] == "nvlinkx18"
        )
        link = Link(link_data)

        self.assertEqual(link.UniBW, 450)
        self.assertEqual(link.BiBW, 900)

    def test_h100_cluster_loads_with_explicit_tp8_topology(self):
        cluster = ClusterConfig.from_file(H100_CLUSTER_PATH)
        parallel_config = ParallelConfig(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

        workers = cluster.workers(parallel_config)

        self.assertEqual(cluster.num_workers, 8)
        self.assertEqual(len(workers), parallel_config.world_size)
        self.assertTrue(all(worker.role == "hybrid" for worker in workers))
        self.assertTrue(all(worker.hardware == "H100" for worker in workers))
        self.assertEqual([worker.rank_info.tp_rank for worker in workers], list(range(8)))
        self.assertTrue(all(worker.rank_info.pp_rank == 0 for worker in workers))
        self.assertTrue(all(worker.rank_info.dp_rank == 0 for worker in workers))


if __name__ == "__main__":
    unittest.main()
