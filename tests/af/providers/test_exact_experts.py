"""Exact analytical routing and fixed-batch memory admission regressions."""

from dataclasses import replace
import json
import unittest

from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.analytical.base import analytical_model_for
from TokenSim.operator_data.workload import work_for_table
from tests.af.support.config import _cluster_dict
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _compute_query, _moe


class ExactExpertsTest(unittest.TestCase):
    def test_one_assignment_is_independent_of_ep_world_size(self):
        hardware = af_test_hardware(moe=_moe())
        provider = OperatorTableAFComputeProvider(hardware, hardware.model("TestModel"))
        times = []
        for ep in (1, 4, 256):
            query = replace(
                _compute_query("expert", tokens=1, ep_size=ep),
                stage="routed_expert",
                layer_type="moe",
            )
            estimate = provider.estimate(query)
            component = estimate.details["components"][0]
            self.assertEqual(component["table"], "routed_experts")
            self.assertEqual(json.loads(component["key"]["expert_tokens"]), [1])
            times.append(estimate.service_duration)
        self.assertEqual(times, [times[0]] * 3)

    def test_active_experts_control_weight_reads(self):
        key = dict(dtype="bf16", hidden_size=16, inter_size=32, expert_tokens="[4]")
        one = work_for_table("routed_experts", key)
        two = work_for_table("routed_experts", {**key, "expert_tokens": "[2,2]"})
        self.assertEqual(one.flops, two.flops)
        self.assertEqual(two.weight_bytes, 2 * one.weight_bytes)
        ungated = work_for_table("routed_experts", key, gated=False)
        self.assertEqual(ungated.weight_bytes, 2 * 16 * 32 * 2)

    def test_groq_supports_exact_routed_experts(self):
        hardware = af_test_hardware()
        device = replace(hardware.device("f-device"), family="groq_tsp")
        result = analytical_model_for("groq_tsp").estimate(
            "routed_experts",
            dict(dtype="fp16", hidden_size=16, inter_size=32, expert_tokens="[1,3]"),
            device,
        )
        self.assertEqual(result.work.weight_bytes, 2 * 3 * 16 * 32 * 2)

    def test_ungated_runtime_reserves_two_ffn_projections(self):
        hardware = af_test_hardware()
        model = replace(hardware.model("TestModel"), activation="relu")
        runtime = build_model_runtime(_cluster_dict(), hardware=hardware, model=model)
        f_state = runtime.plan.physical_state("f-pool:0")
        self.assertEqual(f_state.model_bytes, 2 * 16 * 64 * 2)

    def test_fixed_batch_rejects_unresident_kv_and_invalid_counts(self):
        hardware = af_test_hardware()
        doc = _cluster_dict()
        doc["kv_cache_capacity_tokens_per_dp_rank"] = 32
        runtime = build_model_runtime(
            doc, hardware=hardware, model=hardware.model("TestModel")
        )
        simulate_decode(runtime, 1, 15, fallback="analytical_only", trace=False)
        with self.assertRaisesRegex(ConfigurationError, "KV does not fit"):
            simulate_decode(runtime, 2, 32)
        with self.assertRaisesRegex(ConfigurationError, "context limit"):
            simulate_decode(runtime, 1, 4096)
        with self.assertRaises(ValueError):
            simulate_decode(runtime, True, 16)
