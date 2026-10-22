"""Communication ownership in the conventional AF exploration."""

import json
from pathlib import Path
import unittest

from TokenSim.hardware import HardwareContext
from scripts.lpx.aa_model import Capacity, GemmaText
from scripts.lpx.aa_search import request_cost
from scripts.lpx.context_sensitivity import clear_caches, generation, make_costs


ROOT = Path(__file__).resolve().parents[1]


class ConventionalAFTest(unittest.TestCase):
    def tearDown(self):
        clear_caches()

    def test_attention_owns_kv_and_each_output_pays_the_same_af_transfer(self):
        hardware = HardwareContext.load(ROOT / "data")
        raw = json.loads((ROOT / "experiments/lpx/aa-gemma4.json").read_text())
        model = GemmaText(**raw["model"])
        capacity = Capacity(hardware, model, weight_dtype="bf16", kv_dtype="bf16")
        case = dict(
            name="conventional-af",
            mode="af",
            decode=(
                capacity.place("rubin", 8, "attention", 102000),
                capacity.place("groq_lp30", 12, "ffn", 102000),
            ),
            prefill=capacity.place("rubin", 8, "full", 102000),
        )
        for settings in ({}, {"cross_latency_us": 0}, {"link_latency_scale": 0}):
            costs = make_costs(hardware, model, settings)
            requests = [request_cost(costs, case, w) for w in raw["workloads"]]
            expected = costs.af_boundary(8, 12)
            for request in requests:
                self.assertEqual(request["handoff_bytes"], 0)
                self.assertEqual(request["handoff_us"], 0)
                self.assertEqual(request["components"]["cross_us"], expected)
            self.assertGreater(expected, 0)
            rates = [generation(costs, case, w)["tps"] for w in raw["workloads"]]
            self.assertTrue(all(a > b for a, b in zip(rates, rates[1:])))
            clear_caches()


if __name__ == "__main__":
    unittest.main()
