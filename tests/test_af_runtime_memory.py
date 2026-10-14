"""Af runtime memory."""

from __future__ import annotations

from dataclasses import replace
import unittest

from TokenSim.af.runtime import kv_head_partitions
from TokenSim.errors import ConfigurationError
from tests.support.af_parallel import _cluster
from tests.support.af_runtime import _moe_config, _runtime_hardware, _runtime_plan


class AFRuntimePlanTest(unittest.TestCase):
    def test_kv_heads_shard_replicate_and_reject_ambiguous_topology(self):
        sharded = kv_head_partitions(
            num_kv_heads=8,
            tensor_parallel_size=2,
            head_dim=2,
            num_layers=2,
            block_size=4,
            dtype_size_bytes=2,
        )
        replicated = kv_head_partitions(
            num_kv_heads=1,
            tensor_parallel_size=2,
            head_dim=2,
            num_layers=2,
            block_size=4,
            dtype_size_bytes=2,
        )

        self.assertEqual(
            [item.global_head_ids for item in sharded], [(0, 1, 2, 3), (4, 5, 6, 7)]
        )
        self.assertEqual([item.replica_factor for item in sharded], [1, 1])
        self.assertEqual([item.global_head_ids for item in replicated], [(0,), (0,)])
        self.assertEqual([item.replica_factor for item in replicated], [2, 2])
        with self.assertRaisesRegex(ConfigurationError, "cannot be evenly sharded"):
            kv_head_partitions(
                num_kv_heads=3,
                tensor_parallel_size=2,
                head_dim=2,
                num_layers=2,
                block_size=4,
                dtype_size_bytes=2,
            )

    def test_component_ownership_matches_hand_calculation(self):
        plan = _runtime_plan()
        a_leader = plan.physical_state("a-pool:0")
        a_follower = plan.physical_state("a-pool:1")
        f_ep0 = plan.physical_state("f-pool:0")
        f_ep1 = plan.physical_state("f-pool:2")

        self.assertEqual(a_leader.component_bytes()["attention"], 2048)
        self.assertEqual(a_leader.component_bytes()["embedding_lm_head"], 1600000)
        self.assertEqual(a_leader.component_bytes()["router"], 128)
        self.assertNotIn("router", a_follower.component_bytes())
        self.assertNotIn("routed_expert", a_leader.component_bytes())
        self.assertNotIn("router", f_ep0.component_bytes())
        self.assertEqual(f_ep0.component_bytes()["dense_ffn"], 3072)
        self.assertEqual(f_ep0.component_bytes()["routed_expert"], 3072)
        self.assertEqual(f_ep0.component_bytes()["shared_expert"], 1536)
        self.assertNotIn("router", f_ep1.component_bytes())
        self.assertEqual(f_ep1.component_bytes()["dense_ffn"], 3072)
        self.assertEqual(f_ep1.component_bytes()["routed_expert"], 3072)
        self.assertEqual(f_ep1.component_bytes()["shared_expert"], 1536)
        self.assertEqual(f_ep0.component_bytes(), f_ep1.component_bytes())

        a_bindings = [
            binding
            for binding in plan.rank_bindings
            if binding.capability == "af_attention"
        ]
        for binding in a_bindings:
            state = plan.physical_state(binding.physical_identity)
            if binding.is_scheduler_leader:
                self.assertIn("router", state.component_bytes())
            else:
                self.assertNotIn("router", state.component_bytes())
        for binding in plan.rank_bindings:
            if binding.capability == "af_ffn":
                self.assertNotIn(
                    "router",
                    plan.physical_state(binding.physical_identity).component_bytes(),
                )

        self.assertEqual([layer.layer_type for layer in plan.layers], ["moe", "dense"])
        for state in plan.physical_states:
            for component in state.components:
                if component.layer_id == 0:
                    self.assertNotEqual(component.category, "dense_ffn")
                if component.layer_id == 1:
                    self.assertNotIn(
                        component.category,
                        {"router", "routed_expert", "shared_expert"},
                    )
            self.assertEqual(
                len(state.components),
                len({component.component_id for component in state.components}),
            )

    def test_attention_memory_uses_actual_mha_gqa_and_mqa_widths(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        expected_bytes = {8: 1024, 2: 640, 1: 640}
        for kv_heads, expected in expected_bytes.items():
            with self.subTest(kv_heads=kv_heads):
                plan = _runtime_plan(
                    cluster,
                    hardware=_runtime_hardware(layers=1),
                    moe_config=replace(
                        _moe_config(),
                        num_moe_layers=1,
                        num_key_value_heads=kv_heads,
                    ),
                )
                state = plan.physical_state("a-pool:0")
                attention = next(
                    component
                    for component in state.components
                    if component.category == "attention"
                )
                self.assertEqual(attention.bytes, expected)
                self.assertEqual(
                    dict(attention.shape)["kv_heads_local"], 4 if kv_heads == 8 else 1
                )
                self.assertIn("Q/K/V/O", attention.formula)

    def test_weight_and_kv_dtypes_change_independent_ledgers(self):
        baseline_cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        typed_cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        typed_cluster.af = replace(
            typed_cluster.af,
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp32",
        )
        typed_cluster.af.links = [
            replace(
                typed_cluster.af.links[0],
                a_to_f_dtype="fp8",
                f_to_a_dtype="fp16",
            )
        ]

        baseline = _runtime_plan(baseline_cluster, dtype="bf16")
        typed = _runtime_plan(typed_cluster, dtype="bf16")
        baseline_a = baseline.physical_state("a-pool:0")
        typed_a = typed.physical_state("a-pool:0")
        baseline_binding = baseline.binding("a-domain", 0)
        typed_binding = typed.binding("a-domain", 0)

        self.assertEqual(typed.effective_dtypes.weight_dtype, "fp8")
        self.assertEqual(typed.effective_dtypes.activation_dtype, "bf16")
        self.assertEqual(typed.effective_dtypes.kv_cache_dtype, "fp32")
        self.assertEqual(typed.effective_dtypes.a_to_f_dtype, "fp8")
        self.assertEqual(typed.effective_dtypes.f_to_a_dtype, "fp16")
        self.assertEqual(typed_a.model_bytes * 2, baseline_a.model_bytes)
        self.assertEqual(
            typed_binding.kv_partition.bytes_per_block,
            baseline_binding.kv_partition.bytes_per_block * 2,
        )

    def test_oom_and_missing_metadata_fail_before_execution(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            (
                "domain/rank=.*physical_worker=.*component_bytes=.*model_bytes=.*"
                "capacity_bytes=.*required_bytes=.*available_bytes"
            ),
        ):
            _runtime_plan(hardware=_runtime_hardware(capacity=0.000001))

        unknown_hardware = _cluster()
        for group in unknown_hardware.worker_groups:
            group.hardware = "missing-device"
        with self.assertRaisesRegex(
            ConfigurationError, "unknown hardware 'missing-device'"
        ):
            _runtime_plan(unknown_hardware)
        with self.assertRaisesRegex(ConfigurationError, "no explicit byte size"):
            _runtime_plan(dtype="mystery")
