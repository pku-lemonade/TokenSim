"""Operator queries follow AF stage ownership and rank-local geometry."""

from dataclasses import replace
import unittest

from TokenSim.af.operator_provider import OperatorTableAFComputeProvider
from tests.test_af_executor import af_test_hardware
from tests.test_af_operator_provider import _moe, _compute_query


class AFProviderCorrectnessTest(unittest.TestCase):
    def setUp(self):
        hardware = af_test_hardware(moe=_moe())
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
