"""Operator queries follow AF stage ownership and rank-local geometry."""

from dataclasses import replace
import unittest

from TokenSim.af.operator_provider import OperatorTableAFComputeProvider
from TokenSim.hardware.device import DeviceCatalog
from tests.support.af_hardware import af_test_hardware
from tests.support.af_queries import _compute_query, _moe


class AFProviderCorrectnessTest(unittest.TestCase):
    def setUp(self):
        hardware = af_test_hardware(moe=_moe())
        hardware = replace(
            hardware,
            devices=DeviceCatalog(
                [
                    replace(
                        device,
                        peak_compute={
                            **device.peak_compute,
                            "fp8": device.peak_compute["fp16"],
                        },
                    )
                    for device in hardware.devices
                ]
            ),
        )
        self.provider = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="analytical_only"
        )

    def test_runtime_router_prices_topk_without_repeating_attention_norm(self):
        query = replace(_compute_query("router"), layer_type="moe")
        components = self.provider.estimate(query).details["components"]
        self.assertEqual([c["name"] for c in components], ["router", "router_topk"])
        self.assertEqual(components[1]["key"]["op_name"], "topk")

    def test_routed_experts_do_not_own_residual_addition(self):
        query = replace(
            _compute_query("expert"), stage="routed_expert", layer_type="moe"
        )
        components = self.provider.estimate(query).details["components"]
        self.assertEqual([c["name"] for c in components], ["moe_expert"])

    def test_combine_and_residual_use_the_owned_hidden_interval(self):
        for stage in ("weighted_combine", "residual"):
            query = replace(
                _compute_query("a", tp_size=2),
                stage=stage,
                operation=stage,
                layer_type="moe",
            )
            components = self.provider.estimate(query).details["components"]
            self.assertEqual(len(components), 1)
            self.assertEqual(components[0]["key"]["hidden_size"], 8)

    def test_weight_and_kv_dtypes_are_independent_of_activations(self):
        query = replace(
            _compute_query("a", dtype="bf16", context=(8192,)),
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp32",
        )
        estimate = self.provider.estimate(query)
        components = estimate.details["components"]
        for component in components:
            if component["table"] == "gemm":
                self.assertEqual(component["key"]["dtype"], "fp8")
            elif component["table"] == "elementwise":
                self.assertEqual(component["key"]["dtype"], "bf16")
            elif component["table"] == "generation_attention":
                self.assertEqual(component["key"]["kv_cache_dtype"], "fp32")
        smaller_kv = self.provider.estimate(replace(query, kv_cache_dtype="fp16"))
        self.assertGreater(estimate.service_duration, smaller_kv.service_duration)

    def test_router_and_expert_queries_use_weight_precision(self):
        for stage in ("router", "expert"):
            query = replace(_compute_query(stage, dtype="bf16"), weight_dtype="fp8")
            components = self.provider.estimate(query).details["components"]
            weights = [c for c in components if c["table"] in {"gemm", "moe"}]
            self.assertTrue(weights)
            self.assertTrue(all(c["key"]["dtype"] == "fp8" for c in weights))

    def test_prefill_reads_cached_keys_in_addition_to_new_queries(self):
        query = replace(
            _compute_query("a", phase="prefill", tokens=8, context=(8,)),
            query_lengths=(8,),
        )
        plain = self.provider.estimate(query)
        cached = self.provider.estimate(replace(query, context_lengths=(8192,)))
        self.assertGreater(cached.service_duration, plain.service_duration)
        attention = next(
            c for c in cached.details["components"] if c["name"] == "attention_core"
        )
        self.assertEqual(attention["key"]["input_seq_len"], 8)
        self.assertEqual(attention["key"]["context_len"], 8192)

    def test_uneven_prefill_queries_are_not_replaced_by_their_mean(self):
        query = replace(
            _compute_query("a", phase="prefill", tokens=7, context=(12, 32)),
            query_lengths=(2, 5),
        )
        components = self.provider.estimate(query).details["components"]
        attention = [c for c in components if c["name"] == "attention_core"]
        self.assertEqual(
            [(c["key"]["input_seq_len"], c["key"]["context_len"]) for c in attention],
            [(2, 12), (5, 32)],
        )

    def test_prefill_rejects_queries_past_available_keys(self):
        query = replace(
            _compute_query("a", phase="prefill", tokens=8, context=(4,)),
            query_lengths=(8,),
        )
        with self.assertRaisesRegex(ValueError, "KV contexts"):
            self.provider.estimate(query)
