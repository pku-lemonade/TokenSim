"""Independent work, physical-resource, live-memory and unit-conversion checks."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.runtime.lpx_memory import (
    LP30Memory,
    validate_queries,
    workspace_components,
)
from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.operator_data.analytical import analytical_model_for
from TokenSim.operator_data.analytical.lp30 import LP30Pipeline
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.fabric import FabricRefinement, LP30Fabric
from scripts.lpx.fidelity import validate, with_fidelity
from scripts.lpx.native_search import build_runtime, memory_report, screen
from scripts.lpx.partition import capacity_partition
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import capture_profile
from scripts.lpx.token_units import TokenUnits, measure

ROOT = Path(__file__).resolve().parents[1]


class UnitAccountingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")

    def estimate(self, table, key, **policy):
        hardware = with_fidelity(self.hardware, {"operator": policy})
        device = hardware.device("groq_lp30")
        return analytical_model_for(device.family).estimate(table, key, device)

    def test_mixed_precision_counts_external_bytes_and_vector_casts(self):
        key = dict(dtype="fp8", m=3, n=64, k=128)
        estimate = self.estimate("gemm", key)
        self.assertEqual(estimate.work.weight_bytes, 64 * 128)
        self.assertEqual(estimate.work.activation_bytes, 3 * (64 + 128) * 2)
        self.assertEqual(estimate.details["conversion_elements"], 3 * (64 + 128))
        self.assertGreater(estimate.details["unit_service_us"]["vxm"], 0)
        slow = self.estimate("gemm", key, vector_ops_per_s=1e8)
        self.assertGreater(slow.latency_us, estimate.latency_us)
        self.assertEqual(
            slow.details["unit_service_us"]["mxm"],
            estimate.details["unit_service_us"]["mxm"],
        )

    def test_softmax_has_distinct_exponential_cost_and_context_scaling(self):
        key = dict(
            attn_dtype="bf16",
            kv_cache_dtype="fp8",
            batch_size=1,
            context_len=1000,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            window_size=0,
        )
        short = self.estimate("generation_attention", key)
        long = self.estimate("generation_attention", {**key, "context_len": 2000})
        self.assertEqual(short.details["exponential_operations"], 2000)
        self.assertEqual(long.details["exponential_operations"], 4000)
        self.assertGreater(
            long.details["vector_operations"], short.details["vector_operations"]
        )
        slower = self.estimate("generation_attention", key, exp_ops_per_s=1e7)
        self.assertGreater(slower.latency_us, short.latency_us)
        capped = self.estimate(
            "generation_attention", {**key, "context_len": 2000, "window_size": 1000}
        )
        self.assertEqual(capped.latency_us, short.latency_us)

    def test_unit_schedule_preserves_serial_upper_bound(self):
        key = dict(dtype="fp8", m=1, n=128, k=256)
        pipelined = self.estimate("gemm", key)
        serialized = self.estimate("gemm", key, schedule="serialized")
        services = pipelined.details["unit_service_us"]
        self.assertGreaterEqual(pipelined.latency_us, max(services.values()))
        self.assertLessEqual(pipelined.latency_us, serialized.latency_us)
        self.assertAlmostEqual(
            serialized.latency_us, sum(services.values()) + serialized.overhead_us
        )

    def test_logit_softcap_counts_vector_work_and_nonlinearity(self):
        estimate = self.estimate(
            "elementwise",
            dict(
                op_name="other_residual",
                dtype="bf16",
                num_tokens=2,
                hidden_size=128,
                lpx_vector_operation="logit_softcap",
            ),
        )
        self.assertEqual(estimate.details["exponential_operations"], 256)
        self.assertEqual(estimate.details["vector_operations"], 1024)

    def test_lse_merge_counts_kv_parallel_fragments(self):
        # Two tokens, two retained heads and four fragments of 512 outputs + LSE.
        estimate = self.estimate(
            "elementwise",
            dict(
                op_name="weighted_combine",
                dtype="bf16",
                num_tokens=16,
                hidden_size=513,
                lpx_vector_operation="lse_merge",
                kv_parallel_size=4,
            ),
        )
        self.assertEqual(estimate.details["exponential_operations"], 16)
        self.assertEqual(estimate.details["vector_operations"], 16 * (2 * 512 + 3))
        self.assertEqual(estimate.work.activation_bytes, 16 * 513 * 4 + 4 * 512 * 2)

    def test_explicit_configuration_rejects_invalid_assumptions(self):
        for raw in (
            {"typo": {}},
            {"operator": {"exp_ops_per_s": 0}},
            {"operator": {"unknown": 1}},
            {
                "operator": {"attention_tile_keys": 32},
                "memory": {"attention_tile_keys": 64},
            },
            {"fabric": {"p2p_lanes": 97}},
            {"compute": {"gemm_mfu": 2}},
        ):
            with self.subTest(raw=raw), self.assertRaises(ConfigurationError):
                validate(raw)
        original = self.hardware.device("groq_lp30").analytical
        changed = with_fidelity(self.hardware, {"compute": {"gemm_mfu": 0.3}})
        self.assertEqual(changed.device("groq_lp30").analytical["gemm_mfu"], 0.3)
        self.assertNotEqual(original["gemm_mfu"], 0.3)
        self.assertEqual(changed.device("h100_sxm"), self.hardware.device("h100_sxm"))


class FabricOracleTest(unittest.TestCase):
    def test_routes_cannot_borrow_unallocated_chips(self):
        builder = SimpleNamespace(worker_indices={"rank0": 0, "rank1": 1})
        fabric = FabricRefinement(
            builder, LP30Fabric(routes=({"source": 0, "target": 1, "via": [16]},))
        )
        with self.assertRaises(ConfigurationError):
            fabric.routed_phase([(0, 1, 320)], 6)
        builder.worker_indices["router"] = 16
        resources, _, _ = fabric.routed_phase([(0, 1, 320)], 6)
        self.assertIn("lp30:16:tx", resources)

    def test_scope_is_decided_by_physical_chips_and_routes(self):
        fabric = LP30Fabric(routes=({"source": 0, "target": 2, "via": [16]},))
        self.assertEqual(fabric.propagation_s((0, 15)), 0.35e-6)
        self.assertEqual(fabric.propagation_s((15, 16)), 0.75e-6)
        self.assertAlmostEqual(fabric.propagation_s((127, 128)), 1.15e-6)
        keys, wire, flight = fabric.phase([(0, 2, 840)], lanes=6)
        self.assertAlmostEqual(wire, 840 / (6 * 14e9))
        self.assertAlmostEqual(flight, 1.5e-6)
        self.assertIn("lp30:16:tx", keys)
        self.assertIn("lp30:16:rx", keys)

    def test_shared_port_serialization_and_flight_release(self):
        fabric = LP30Fabric(ports_per_chip=1, lanes_per_peer=1, p2p_lanes=1)
        first = fabric.phase([(0, 1, 14000)], 1)
        second = fabric.phase([(0, 2, 14000)], 1)
        nodes = tuple(
            ReplayNode((), keys, keys[0], 0, wire + flight, wire, rank)
            for rank, (keys, wire, flight) in enumerate((first, second))
        )
        kernel = ReplayKernel()
        try:
            kernel.submit(kernel.add(nodes), 0)
            self.assertAlmostEqual(kernel.advance().end, 2e-6 + 0.35e-6)
        finally:
            kernel.close()
        _, combined, _ = fabric.phase([(0, 1, 14000), (0, 2, 14000)], 1)
        self.assertAlmostEqual(combined, 2e-6)


class MemoryAndNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")
        cls.model = replace(
            cls.hardware.models.resolve("gemma-4-31b-text"),
            dtype="fp8",
            kv_cache_dtype="fp8",
        )

    def layout(self):
        return Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=16,
            f_tp=16,
            a_pp=16,
            f_pp=16,
            f_ep=1,
        )

    def test_workspace_reuses_sequential_arenas_and_scales_prefill(self):
        c = SimpleNamespace(
            category="dense_ffn",
            shape=(("hidden_size", 128), ("local_intermediate_size", 256)),
        )
        one = dict(
            workspace_components(
                (c,), self.model, dict(prefill_chunk=1, reserve_bytes=0)
            )
        )
        many = dict(
            workspace_components(
                (c, c), self.model, dict(prefill_chunk=4, reserve_bytes=0)
            )
        )
        self.assertEqual(one["lp30_live_scratch"], (3 * 256 + 2 * 128) * 2)
        self.assertEqual(many["lp30_live_scratch"], one["lp30_live_scratch"] * 4)
        with self.assertRaises(ConfigurationError):
            validate_queries((LP30Memory(prefill_chunk=4),), 5, 1000)

    def test_native_capture_contains_ports_reductions_and_memory_reservations(self):
        hardware = with_execution(
            self.hardware,
            ExecutionAssumptions(
                mode="compiled", instruction_floor_us=0.05, c2c_hop_us=0.35
            ),
        )
        hardware = with_fidelity(hardware, dict(operator={}, memory={}, fabric={}))
        runtime = build_runtime(hardware, self.model, self.layout())
        memory = memory_report(runtime, 102000)
        self.assertGreater(
            memory["peak_rank"]["buffer_breakdown"]["lp30_live_scratch"], 0
        )
        profile = capture_profile(
            runtime,
            "search",
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=1001,
            dp_rank=0,
            seed=42,
            sample=0,
            streaming={"collective_launch_us": 0},
        )
        counts = profile.metadata["streaming"]["fabric"]["counters"]
        self.assertGreater(counts["reduction_operations_per_endpoint"], 0)
        self.assertGreater(counts["pipeline_handoffs"], 0)
        self.assertTrue(any(":port:" in r for n in profile.nodes for r in n.resources))
        profile.verify()
        service = AFService(simpy.Environment(), runtime, trace=False)
        session = service.admit(0, 0, maximum_context=102000, current_context=0)
        with self.assertRaises(ConfigurationError):
            service.env.run(
                until=service.env.process(
                    service.forward([session], phase="prefill", query_tokens=17)
                )
            )

    def test_capacity_partition_matches_native_live_memory(self):
        hardware = with_fidelity(self.hardware, dict(memory={}))
        layout = replace(self.layout(), a_pp=8, f_pp=8)
        partition = capacity_partition(hardware, self.model, layout, 102000, {})
        runtime = build_runtime(
            hardware, self.model, layout, layer_counts=partition["layer_counts"]
        )
        report = memory_report(runtime, 102000)
        self.assertGreater(report["minimum_headroom_bytes"], 0)
        forecast = max(max(stage) for stage in partition["stage_rank_bytes"])
        self.assertGreaterEqual(forecast, report["peak_rank"]["total_bytes"])
        tiny = with_fidelity(hardware, dict(memory=dict(reserve_bytes=500_000_000)))
        with self.assertRaises(ConfigurationError):
            capacity_partition(tiny, self.model, layout, 102000, {})

    def test_model_and_operator_activation_precision_must_match(self):
        hardware = with_fidelity(
            self.hardware, dict(operator=dict(activation_dtype="fp8"))
        )
        with self.assertRaises(ConfigurationError):
            build_runtime(hardware, self.model, self.layout())

    def test_nonidentity_token_ratio_changes_native_lengths_and_reported_rate(self):
        workloads = ((1000, 2, 3400), (10000, 2, 3382), (100000, 2, 3431))
        with patch("scripts.lpx.native_search.WORKLOADS", workloads):
            result = screen(
                self.hardware,
                self.model,
                self.layout(),
                verify=True,
                requests=1,
                warmup=0,
                token_units=dict(native_per_aa_input=0.8, native_per_aa_output=2),
            )
        self.assertEqual(result["status"], "ok")
        for row in result["workloads"]:
            self.assertEqual(row["native_input_tokens"], int(row["input_tokens"] * 0.8))
            self.assertEqual(row["native_output_tokens"], 4)
            self.assertEqual(row["generation_tps"], row["native_generation_tps"] / 2)
            self.assertEqual(row["serial_replay"]["executed_decode_steps"], 3)


class TokenUnitsTest(unittest.TestCase):
    def test_same_text_ratio_and_fractional_first_native_token(self):
        result = measure(
            [dict(input="abcd", output="abcdef")],
            list,
            lambda text: [text[i : i + 2] for i in range(0, len(text), 2)],
            tokenizer="fixture",
            revision="pinned",
        )
        units = TokenUnits(**result["token_units"])
        self.assertEqual(units.native_lengths(1000, 1500), (2000, 3000))
        self.assertEqual(units.aa_intervals(3000), 1499.5)
        self.assertTrue(units.measured)
        self.assertEqual(len(units.corpus_sha256), 64)
        self.assertEqual(TokenUnits().native_lengths(1000, 1000), (1000, 1000))
        for raw in ({"native_per_aa_output": 0}, {"measured": True}):
            with self.assertRaises(ConfigurationError):
                TokenUnits(**raw)


if __name__ == "__main__":
    unittest.main()
