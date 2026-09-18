from __future__ import annotations

import unittest

from TokenSim.comm.collectives import CollectiveModel, CollectiveQuery, ring_traffic_factor
from TokenSim.config.model_config import ModelCatalog, ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.device import DeviceCatalog, DeviceSpec, dtype_bytes, normalize_dtype
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.topology import TopologyLevel, TopologySpec
from tests.hardware_fixtures import test_device, test_links, test_model, test_topology


class DtypeTest(unittest.TestCase):
    def test_aliases_normalize_and_bytes(self):
        self.assertEqual(normalize_dtype("bfloat16"), "bf16")
        self.assertEqual(normalize_dtype("half"), "fp16")
        self.assertEqual(normalize_dtype("sq"), "int8_sq")
        self.assertEqual(dtype_bytes("fp8"), 1.0)
        self.assertEqual(dtype_bytes("nvfp4"), 0.5)
        self.assertEqual(normalize_dtype("w4a8_mxfp4_mxfp8"), "mxfp4")
        self.assertEqual(normalize_dtype("w4a16_mxfp4"), "mxfp4_wo")
        self.assertEqual(normalize_dtype("w4afp8"), "int4_a8")
        self.assertEqual(dtype_bytes("mxfp4"), 0.5)
        with self.assertRaises(ConfigurationError):
            normalize_dtype("fp12")

    def test_four_bit_pipes_fall_back_to_fp8_then_fp16(self):
        catalog = DeviceCatalog.load("data/devices")
        gb300 = catalog.get("gb300")
        self.assertEqual(gb300.peak_compute_for("mxfp4").value, gb300.peak_compute["nvfp4"].value)
        self.assertEqual(gb300.peak_compute_for("mxfp4_wo").value, gb300.peak_compute["fp16"].value)
        h100 = catalog.get("h100_sxm")
        self.assertEqual(h100.peak_compute_for("mxfp4").value, h100.peak_compute["fp8"].value)
        a100 = catalog.get("a100_sxm_80g")
        self.assertEqual(a100.peak_compute_for("int4_wo").value, a100.peak_compute["fp16"].value)
        with self.assertRaises(ConfigurationError):
            a100.peak_compute_for("fp8")


class DeviceCatalogTest(unittest.TestCase):
    def test_repository_catalog_loads_and_resolves_aliases(self):
        catalog = DeviceCatalog.load("data/devices")
        a100 = catalog.get("A100-40G")
        self.assertEqual(a100.device_id, "a100_sxm_40g")
        self.assertEqual(catalog.get("NVIDIA H100 80GB HBM3").device_id, "h100_sxm")
        self.assertEqual(catalog.get("groq").family, "groq_tsp")
        self.assertTrue(catalog.get("GroqChip").weights_resident_on_chip)
        self.assertIsNone(catalog.get("4090").scale_up_link)
        for device in catalog:
            self.assertGreater(device.peak_compute_for("fp16").value, 0)
            self.assertIn(device.memory_capacity_bytes.grade, {"A", "B", "C", "D"})
            self.assertLess(device.memory_reserved_bytes.value, device.memory_capacity_bytes.value)
            self.assertIn(device.analytical_grade, {"A", "B", "C", "D"})
            if device.analytical_grade != "D":
                self.assertIn(device.analytical_source_id, device.sources)

    def test_aiconfigurator_corrections_are_grade_c_on_nvidia_gpus(self):
        catalog = DeviceCatalog.load("data/devices")
        for name in ("a100_sxm_80g", "h100_sxm", "b200_sxm", "rtx_4090"):
            device = catalog.get(name)
            self.assertEqual(device.analytical_grade, "C", name)
            self.assertEqual(device.analytical["memory_efficiency"], 0.8)
            self.assertEqual(device.analytical["kernel_launch_us"], 3.0)
            self.assertEqual(device.analytical["collective_launch_us"], 10.0)
            self.assertEqual(device.memory_reserved_bytes.grade, "C")
            self.assertAlmostEqual(device.usable_memory_bytes, device.memory_capacity_bytes.value - 4169138176)
        groq = catalog.get("groqchip_v1")
        self.assertEqual(groq.analytical_grade, "D")
        self.assertEqual(groq.memory_reserved_bytes.value, 0.0)

    def test_analytical_source_must_be_declared(self):
        raw = {
            "device_id": "bad",
            "family": "nvidia_gpu",
            "sources": {"x": {"grade": "A", "reference": "r"}},
            "peak_compute": {"fp16": {"value": 1e12, "source_id": "x"}},
            "memory": {
                "capacity_bytes": {"value": 1e10, "source_id": "x"},
                "bandwidth_bytes_per_s": {"value": 1e12, "source_id": "x"},
            },
            "analytical": {"source_id": "undeclared", "grade": "C", "memory_efficiency": 0.8},
        }
        with self.assertRaises(ConfigurationError):
            DeviceSpec.from_mapping(raw)

    def test_weight_only_dtypes_run_on_the_fp16_pipe(self):
        device = test_device()
        self.assertEqual(device.peak_compute_for("int8_wo").value, device.peak_compute_for("fp16").value)
        # a device without fp4/fp8 pipes can only run 4-bit weights as weight-only fp16
        self.assertEqual(device.peak_compute_for("nvfp4").value, device.peak_compute_for("fp16").value)
        with self.assertRaises(ConfigurationError):
            device.peak_compute_for("fp8")

    def test_duplicate_alias_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            DeviceCatalog([test_device("a", aliases=("x",)), test_device("b", aliases=("X",))])


class ModelCatalogTest(unittest.TestCase):
    def test_repository_models_load_with_plausible_parameter_counts(self):
        catalog = ModelCatalog.load("data/models")
        llama7b = catalog.get("LLaMa-7B")
        self.assertAlmostEqual(llama7b.total_params() / 1e9, 6.74, places=1)
        llama70b = catalog.get("llama-2-70b")
        self.assertEqual(llama70b.kv_dim, 1024)
        self.assertAlmostEqual(llama70b.total_params() / 1e9, 69.0, places=0)
        mixtral = catalog.get("mixtral-8x7b")
        self.assertTrue(mixtral.is_moe)
        self.assertEqual(len(mixtral.moe_layer_indices()), 32)
        self.assertAlmostEqual(mixtral.total_params() / 1e9, 46.7, places=0)

    def test_invalid_head_configuration_fails(self):
        with self.assertRaises(ConfigurationError):
            ModelSpec(
                model_id="bad",
                hidden_size=128,
                intermediate_size=256,
                num_layers=2,
                num_attention_heads=8,
                num_key_value_heads=3,
                head_dim=16,
            )

    def test_moe_override_from_psla_metadata(self):
        base = test_model("dense")
        from TokenSim.moe.config import MoEModelConfig

        moe = MoEModelConfig(is_moe_model=True, num_experts=8, num_experts_per_tok=2, num_moe_layers=4, hidden_size=256, intermediate_size=512, num_attention_heads=8, num_key_value_heads=8)
        merged = base.with_moe_override(moe)
        self.assertTrue(merged.is_moe)
        self.assertEqual(merged.hidden_size, 256)
        self.assertEqual(merged.head_dim, 32)


class TopologyTest(unittest.TestCase):
    def test_coordinates_and_group_layout(self):
        topology = test_topology(node_size=8)
        self.assertEqual(topology.coordinates(13), (1, 0))
        self.assertEqual(topology.lowest_common_level([0, 1]), 0)
        self.assertEqual(topology.lowest_common_level([0, 9]), 1)
        self.assertEqual(topology.lowest_common_level([5]), -1)
        self.assertEqual(topology.group_layout(list(range(16))).fan, (8, 2))
        self.assertEqual(topology.group_layout([0, 8, 16, 24]).fan, (1, 4))
        self.assertEqual(topology.group_layout([3, 4]).fan, (2, 1))

    def test_three_level_topology_scales_to_thousands_of_devices(self):
        topology = TopologySpec(
            "big",
            levels=(
                TopologyLevel("node", 8, "nvlink-test"),
                TopologyLevel("rack", 9, "ethernet-test"),
                TopologyLevel("system", None, "ethernet-test"),
            ),
        )
        layout = topology.group_layout(list(range(10_440)))
        self.assertEqual(layout.fan, (8, 9, 145))
        self.assertEqual(layout.lowest_common_level, 2)

    def test_default_operator_backend_is_vllm(self):
        hardware = HardwareContext.load("data")
        self.assertEqual(hardware.operator_package("h100_sxm").backend, "vllm")
        self.assertEqual(hardware.operator_package("h100_sxm", "trtllm").backend, "trtllm")
        self.assertEqual(hardware.operator_package("intel_arc_pro_b60").backend, "vllm")
        self.assertEqual(sorted(hardware.available_backends("h100_sxm")), ["sglang", "trtllm", "vllm"])

    def test_repository_topologies_reference_known_links(self):
        hardware = HardwareContext.load("data")
        for topology in hardware.topologies:
            topology.validate_links(hardware.links)
        self.assertIn("groq_node8_rack72", {t.topology_id for t in hardware.topologies})

    def test_invalid_level_definitions_fail(self):
        with self.assertRaises(ConfigurationError):
            TopologyLevel("node", 0, "nvlink-test")
        with self.assertRaises(ConfigurationError):
            TopologySpec("bad", levels=(TopologyLevel("a", None, "x"), TopologyLevel("b", 2, "x")))


class CollectiveModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.links = LinkCatalog(test_links())
        self.topology = test_topology(node_size=4)
        self.model = CollectiveModel(self.topology, self.links, family="nvidia_gpu")

    def test_traffic_factors_follow_nccl_tests_convention(self):
        self.assertAlmostEqual(ring_traffic_factor("all_reduce", 8), 2 * 7 / 8)
        self.assertAlmostEqual(ring_traffic_factor("all_gather", 4), 3 / 4)
        self.assertEqual(ring_traffic_factor("all_reduce", 1), 0.0)

    def test_estimate_is_monotonic_in_message_size_and_group(self):
        layout4 = self.topology.group_layout([0, 1, 2, 3])
        small = self.model.estimate(CollectiveQuery("all_reduce", 1 << 12, layout4))
        big = self.model.estimate(CollectiveQuery("all_reduce", 1 << 26, layout4))
        self.assertLess(small.latency_us, big.latency_us)
        layout8 = self.topology.group_layout(list(range(8)))
        cross = self.model.estimate(CollectiveQuery("all_reduce", 1 << 20, layout8))
        intra = self.model.estimate(CollectiveQuery("all_reduce", 1 << 20, layout4))
        self.assertGreater(cross.latency_us, intra.latency_us)
        self.assertEqual(cross.bottleneck_level, "cluster")

    def test_large_messages_approach_link_bandwidth(self):
        layout = self.topology.group_layout([0, 1])
        link = self.links.get("nvlink-test")
        message = 1 << 30
        estimate = self.model.estimate(CollectiveQuery("all_reduce", message, layout))
        moved = ring_traffic_factor("all_reduce", 2) * message
        ideal_us = moved / (link.bandwidth_per_direction_bytes_per_s * link.collective_efficiency) * 1e6
        self.assertGreater(estimate.latency_us, ideal_us)
        self.assertLess(estimate.latency_us, ideal_us * 1.2)

    def test_measured_curve_extrapolates_with_model_growth(self):
        from TokenSim.operator_data.lookup import OperatorLookup
        from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord

        source = SourceRecord("nccl:test", "A", "measured")
        meta = PackageMeta("v", "TestGPU", "nccl", sources={"nccl:test": source})
        # Measured points run at 2x the analytical model's time.
        layout = self.topology.group_layout([0, 1, 2, 3])
        rows = []
        for size in (1 << 16, 1 << 18, 1 << 20):
            analytical = self.model.estimate(CollectiveQuery("all_reduce", size, layout)).latency_us
            rows.append({"dtype": "fp16", "operation": "all_reduce", "group_size": 4, "nodes": 1, "message_bytes": size, "latency_us": 2.0 * analytical, "source_id": "nccl:test"})
        package = OperatorDataPackage.from_rows(meta, {"collective": rows})
        model = CollectiveModel(self.topology, self.links, family="nvidia_gpu", measured_lookup=OperatorLookup(package))
        beyond = model.estimate(CollectiveQuery("all_reduce", 1 << 23, layout, dtype="fp16"))
        analytical_beyond = self.model.estimate(CollectiveQuery("all_reduce", 1 << 23, layout)).latency_us
        self.assertEqual(beyond.match_type, "extrapolated")
        self.assertIn("analytical_scaled", beyond.detail["flags"])
        # boundary efficiency (2x) is preserved past the measured range
        self.assertAlmostEqual(beyond.latency_us / analytical_beyond, 2.0, places=6)

    def test_measured_table_takes_precedence(self):
        from TokenSim.operator_data.lookup import OperatorLookup
        from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord

        source = SourceRecord("nccl:test", "A", "measured")
        meta = PackageMeta("v", "TestGPU", "nccl", sources={"nccl:test": source})
        rows = [
            {"dtype": "fp16", "operation": "all_reduce", "group_size": 4, "nodes": 1, "message_bytes": size, "latency_us": 40.0 + size / 1e5, "source_id": "nccl:test"}
            for size in (1 << 16, 1 << 20, 1 << 24)
        ]
        package = OperatorDataPackage.from_rows(meta, {"collective": rows})
        model = CollectiveModel(self.topology, self.links, measured_lookup=OperatorLookup(package))
        layout = self.topology.group_layout([0, 1, 2, 3])
        exact = model.estimate(CollectiveQuery("all_reduce", 1 << 20, layout, dtype="fp16"))
        between = model.estimate(CollectiveQuery("all_reduce", 3 << 19, layout, dtype="fp16"))
        other_group = model.estimate(CollectiveQuery("all_reduce", 1 << 20, self.topology.group_layout([0, 1]), dtype="fp16"))
        self.assertEqual(exact.match_type, "measured")
        self.assertAlmostEqual(exact.latency_us, 40.0 + (1 << 20) / 1e5)
        self.assertEqual(between.match_type, "interpolated")
        self.assertEqual(other_group.match_type, "analytical")

    def test_ep_all2all_prefers_measured_deepep_rows(self):
        from TokenSim.comm.collectives import EP_ALL2ALL_MODE_SCALE, EPAllToAllQuery
        from TokenSim.operator_data.lookup import OperatorLookup
        from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord

        layout = self.topology.group_layout([0, 1, 2, 3])
        query = EPAllToAllQuery(layout, num_tokens=32, hidden_size=4096, top_k=8, num_experts=256, dtype="bf16", mode="deepep_high_throughput")
        analytical = self.model.ep_all2all(query)
        plain = self.model.estimate(CollectiveQuery("all_to_all", query.payload_bytes, layout, "bf16"))
        self.assertEqual(analytical.match_type, "analytical")
        self.assertAlmostEqual(analytical.latency_us, 2 * plain.latency_us * EP_ALL2ALL_MODE_SCALE["deepep_high_throughput"])
        naive = self.model.ep_all2all(EPAllToAllQuery(layout, 32, 4096, 8, 256, "bf16", "naive"))
        self.assertGreater(naive.latency_us, analytical.latency_us)

        source = SourceRecord("deepep:test", "A", "measured")
        meta = PackageMeta("v", "TestGPU", "vllm", sources={"deepep:test": source})
        rows = []
        for phase, lat in (("dispatch", 40.0), ("combine", 50.0)):
            for tokens in (16, 64):
                rows.append({"dtype": "bf16", "phase": phase, "mode": "deepep_high_throughput", "ep_size": 4, "nodes": 1, "hidden_size": 4096, "top_k": 8, "num_experts": 256, "num_tokens": tokens, "latency_us": lat * tokens / 16, "source_id": "deepep:test"})
        package = OperatorDataPackage.from_rows(meta, {"ep_all2all": rows})
        model = CollectiveModel(self.topology, self.links, family="nvidia_gpu", measured_lookup=OperatorLookup(package))
        exact = model.ep_all2all(EPAllToAllQuery(layout, 16, 4096, 8, 256, "bf16", "deepep_high_throughput"))
        self.assertEqual(exact.match_type, "measured")
        self.assertAlmostEqual(exact.latency_us, 90.0)
        between = model.ep_all2all(EPAllToAllQuery(layout, 32, 4096, 8, 256, "bf16", "deepep_high_throughput"))
        self.assertEqual(between.match_type, "interpolated")
        # requested LL mode has no rows: falls back to the other DeepEP mode
        other = model.ep_all2all(EPAllToAllQuery(layout, 16, 4096, 8, 256, "bf16", "deepep_low_latency"))
        self.assertEqual(other.detail["mode"], "deepep_high_throughput")
        # unmeasured expert count and top-k are interpolation axes, not hard misses
        experts = model.ep_all2all(EPAllToAllQuery(layout, 16, 4096, 6, 128, "bf16", "deepep_high_throughput"))
        self.assertIn(experts.match_type, {"interpolated", "extrapolated"})
        # non-DeepEP backends never read DeepEP rows
        legacy = model.ep_all2all(EPAllToAllQuery(layout, 16, 4096, 8, 256, "bf16", "allgather_reducescatter"))
        self.assertEqual(legacy.match_type, "analytical")

    def test_point_to_point_uses_lowest_common_level_link(self):
        same_node = self.model.point_to_point(1 << 20, self.topology.group_layout([0, 1]))
        cross_node = self.model.point_to_point(1 << 20, self.topology.group_layout([0, 5]))
        self.assertEqual(same_node.levels[0].link, "nvlink-test")
        self.assertEqual(cross_node.levels[0].link, "ethernet-test")
        self.assertLess(same_node.latency_us, cross_node.latency_us)


if __name__ == "__main__":
    unittest.main()
