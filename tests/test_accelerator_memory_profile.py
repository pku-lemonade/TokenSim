from __future__ import annotations

import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from TokenSim.config.accelerator_memory import (
    apply_accelerator_memory_profile,
    configure_run_hardware,
    resolve_accelerator_memory_profile,
)
from TokenSim.config.cluster_config import ClusterConfig, WorkerGroupConfig
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import parse_mooncake_config
from TransformerRoofline import TransformerRoofline


REPO_ROOT = Path(__file__).resolve().parents[1]


class AcceleratorMemoryResolverTest(unittest.TestCase):
    def test_all_documented_presets(self):
        expected = {
            ("H100", "HBF1"): ("GDDR7", 32.0, 1.792),
            ("H100", "HBF1-48GiB"): ("GDDR7", 48.0, 1.344),
            ("H100", "HBF2"): ("HBM3e", 48.0, 1.5),
            ("H200", "HBF1"): ("GDDR7", 32.0, 1.792),
            ("H200", "HBF2"): ("HBM3e", 48.0, 1.5),
            ("B200", "HBF1"): ("GDDR7", 64.0, 3.584),
            ("B200", "HBF1-48GiB"): ("GDDR7", 96.0, 2.688),
            ("B200", "HBF2"): ("HBM3e", 96.0, 4.0),
        }

        for (hardware, preset), values in expected.items():
            with self.subTest(hardware=hardware, preset=preset):
                profile = resolve_accelerator_memory_profile(hardware, preset)
                self.assertEqual(
                    (
                        profile.memory_type,
                        profile.capacity_gib_per_card,
                        profile.bandwidth_tb_s_per_card,
                    ),
                    values,
                )

    def test_config_parsing_validates_preset(self):
        config = parse_mooncake_config(
            {"accelerator_memory_preset": "HBF1-48GiB"}
        )
        self.assertEqual(config.accelerator_memory_preset, "HBF1-48GiB")

        with self.assertRaisesRegex(ConfigurationError, "accelerator_memory_preset"):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"accelerator_memory_preset": "HBF3"},
            )

    def test_unsupported_and_conflicting_targets_fail(self):
        with self.assertRaisesRegex(ConfigurationError, "unsupported"):
            resolve_accelerator_memory_profile("A100", "HBF1")
        with self.assertRaisesRegex(ConfigurationError, "unsupported"):
            resolve_accelerator_memory_profile("H200", "HBF1-48GiB")
        with self.assertRaisesRegex(ConfigurationError, "conflicts"):
            resolve_accelerator_memory_profile(
                "H100",
                "HBF1",
                target_hardware="B200",
            )

    def test_apply_changes_only_a_run_local_hardware_copy(self):
        base = SimpleNamespace(
            Capacity=96,
            BW_TBs=3.0,
            MM_BW_TBs=3.0,
            MM_GP_BW_TBs=3.0,
            MV_BW_TBs=3.0,
            MM_TFLOPS=1000,
            MM_GP_TFLOPS=1000,
            MV_TFLOPS=1000,
            Pcie="pcie5.0x16",
            Nvlink="nvlinkx18",
        )
        shared_hardwares = {"H100": base}
        roofline = SimpleNamespace(hardwares=shared_hardwares)
        profile = resolve_accelerator_memory_profile("H100", "HBF1-48GiB")

        effective = apply_accelerator_memory_profile(roofline, profile)

        self.assertIsNot(effective, base)
        self.assertIsNot(roofline.hardwares, shared_hardwares)
        self.assertEqual((effective.Capacity, effective.BW_TBs), (48.0, 1.344))
        self.assertEqual((base.Capacity, base.BW_TBs), (96, 3.0))
        self.assertEqual((effective.Pcie, effective.Nvlink), (base.Pcie, base.Nvlink))

    def test_run_configuration_applies_profile_and_records_shared_capacity(self):
        base = SimpleNamespace(
            Capacity=192,
            BW_TBs=8.0,
            MM_BW_TBs=8.0,
            MM_GP_BW_TBs=8.0,
            MV_BW_TBs=8.0,
            MM_TFLOPS=4500,
            MM_GP_TFLOPS=4500,
            MV_TFLOPS=4500,
        )
        roofline = SimpleNamespace(hardwares={"B200": base})
        cluster = ClusterConfig(
            num_workers=8,
            networks={"net": "ethernet100Gb"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="B200",
                    num_workers=8,
                    network="net",
                )
            ],
        )
        kv_config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "target_hardware": "B200",
                "accelerator_memory_preset": "HBF1-48GiB",
                "enable_offload": True,
                "offload_tier": "hbf",
                "hbf_capacity_gb": 48 * 1024,
            },
        )

        profile = configure_run_hardware(roofline, cluster, kv_config)

        self.assertIsNotNone(profile)
        self.assertEqual(profile.worker_count, 8)
        self.assertEqual(profile.shared_offload_capacity_gib, 48 * 1024)
        self.assertEqual(roofline.hardwares["B200"].Capacity, 96.0)
        self.assertEqual(roofline.hardwares["B200"].BW_TBs, 2.688)
        self.assertEqual((base.Capacity, base.BW_TBs), (192, 8.0))

    def test_b200_base_capacity_and_sequential_roofline_isolation(self):
        hardware_path = REPO_ROOT / "TransformerRoofline" / "hardware_models.json"
        hardware_data = json.loads(hardware_path.read_text())
        b200 = next(
            item for item in hardware_data["hardware"] if item["Name"] == "B200"
        )
        self.assertEqual(b200["Capacity"], 192)

        roofline_args = (
            hardware_path,
            REPO_ROOT / "TransformerRoofline" / "allreduce_v100.xlsx",
            REPO_ROOT / "TransformerRoofline" / "hardware_elements.json",
        )
        preset_run = TransformerRoofline(*roofline_args)
        apply_accelerator_memory_profile(
            preset_run,
            resolve_accelerator_memory_profile("H100", "HBF1-48GiB"),
        )
        base_run = TransformerRoofline(*roofline_args)

        self.assertEqual(preset_run.hardwares["H100"].Capacity, 48.0)
        self.assertEqual(preset_run.hardwares["H100"].BW_TBs, 1.344)
        self.assertEqual(base_run.hardwares["H100"].Capacity, 96)
        self.assertEqual(base_run.hardwares["H100"].BW_TBs, 3)


if __name__ == "__main__":
    unittest.main()
