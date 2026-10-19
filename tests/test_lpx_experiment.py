"""Small deterministic serving, capacity and speculative-verification cases."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Case, Layout, Workload, load_cases
from scripts.lpx.operators import Operators
from scripts.lpx.simulator import expected_committed_tokens, pareto, point
from scripts.reproduce_lpx import main

ROOT = Path(__file__).resolve().parents[1]


class LPXExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")
        base = cls.hardware.model("gpt-oss-2t-proxy")
        cls.model = replace(
            base,
            hidden_size=32,
            intermediate_size=64,
            num_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                base.moe, num_experts=8, num_moe_layers=4, moe_intermediate_size=64
            ),
        )
        cls.workload = Workload(
            cached_tokens=16,
            new_tokens=4,
            output_tokens=6,
            prefill_chunk=2,
            context_samples=2,
        )
        cls.base = Layout(a_tp=2, f_ep=2)
        cls.lpu = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=2,
            a_cp=2,
            a_pp=4,
            f_ep=4,
            f_pp=4,
            colocated=False,
        )

    def test_four_serving_modes_are_deterministic(self):
        cases = [
            Case("baseline", "rubin", self.base),
            Case(
                "af",
                "af",
                Layout(a_tp=2, f_device="groq_lp30", f_ep=4, colocated=False),
                self.base,
            ),
            Case("pd", "pd", self.lpu, self.base),
            Case(
                "draft",
                "draft",
                self.base,
                draft=replace(self.lpu, a_cp=1, f_ep=2, colocated=True),
            ),
        ]
        for case in cases:
            with self.subTest(case=case.mode):
                result = point(
                    self.hardware, self.model, self.workload, case, 2, trace=True
                )
                repeat = point(
                    self.hardware, self.model, self.workload, case, 2, trace=True
                )
                self.assertEqual(result, repeat)
                self.assertGreater(result["timings"]["ttft_s"], 0)
                self.assertGreater(result["tokens_per_s_per_user"], 0)
                self.assertTrue(result["trace"])
                self.assertAlmostEqual(
                    result["timings"]["turn_s"], sum(result["pool_service_s"].values())
                )
        self.assertEqual(result["timings"]["verification_width"], 5)

    def test_expected_prefix_and_full_verification_cost(self):
        self.assertEqual(expected_committed_tokens(4, 0), 1)
        self.assertEqual(expected_committed_tokens(4, 1), 5)
        self.assertEqual(expected_committed_tokens(4, 0.5), 1.9375)
        case = Case(
            "draft",
            "draft",
            self.base,
            draft=replace(self.lpu, a_cp=1, f_ep=2, colocated=True),
        )
        reject = point(
            self.hardware,
            self.model,
            replace(self.workload, acceptance_probability=0),
            case,
            1,
        )
        accept = point(
            self.hardware,
            self.model,
            replace(self.workload, acceptance_probability=1),
            case,
            1,
        )
        self.assertEqual(
            reject["timings"]["verification_width"],
            accept["timings"]["verification_width"],
        )
        self.assertAlmostEqual(
            reject["timings"]["decode_s"], 5 * accept["timings"]["decode_s"]
        )
        self.assertEqual(
            reject["resident_memory_bytes_per_rank"],
            accept["resident_memory_bytes_per_rank"],
        )

    def test_power_changes_efficiency_without_changing_latency(self):
        case = Case("baseline", "rubin", self.base)
        low = point(
            self.hardware,
            self.model,
            self.workload,
            case,
            1,
            power_overrides={"rubin": 1000},
        )
        high = point(
            self.hardware,
            self.model,
            self.workload,
            case,
            1,
            power_overrides={"rubin": 2000},
        )
        self.assertEqual(low["timings"], high["timings"])
        self.assertEqual(low["tokens_per_s_per_mw"], 2 * high["tokens_per_s_per_mw"])

    def test_cache_handoff_moves_the_warm_prefix_once(self):
        case = Case("pd", "pd", self.lpu, self.base)
        result = point(self.hardware, self.model, self.workload, case, 2)
        m = self.model
        expected = 2 * 20 * m.kv_elements_per_token() * m.num_layers * 2
        self.assertEqual(result["timings"]["kv_handoff_bytes"], expected)

    def test_context_partition_reduces_kv_without_removing_weight_storage(self):
        operators = Operators(self.hardware, self.model)
        whole = operators.resident_bytes(replace(self.lpu, a_cp=1), 4, 100)
        split = operators.resident_bytes(replace(self.lpu, a_cp=4), 4, 100)
        self.assertEqual(whole["kv"], 4 * split["kv"])
        self.assertEqual(whole["a_weights"], split["a_weights"])

    def test_unresident_weights_and_invalid_layouts_fail(self):
        real = self.hardware.model("gpt-oss-2t-proxy")
        with self.assertRaisesRegex(ConfigurationError, "memory does not fit"):
            Operators(self.hardware, real).resident_bytes(
                replace(self.base, a_device="groq_lp30", f_device="groq_lp30"),
                1,
                404000,
            )
        with self.assertRaises(ConfigurationError):
            Layout(a_cp=2)
        with self.assertRaises(ConfigurationError):
            Workload(acceptance_probability=float("nan"))
        with self.assertRaises(ConfigurationError):
            expected_committed_tokens(True, 0.5)

    def test_dominated_and_infeasible_points_are_excluded(self):
        rows = [
            dict(status="ok", tokens_per_s_per_user=x, tokens_per_s_per_mw=y)
            for x, y in ((1, 2), (2, 2), (3, 1), (2, 2))
        ]
        rows += [dict(status="infeasible", reason="capacity")]
        self.assertEqual(
            [
                (r["tokens_per_s_per_user"], r["tokens_per_s_per_mw"])
                for r in pareto(rows)
            ],
            [(2, 2), (3, 1)],
        )

    def test_default_config_and_cli_reject_bad_batches_before_output(self):
        raw, workload, cases = load_cases(ROOT / "experiments/lpx/sweep.json")
        self.assertEqual(workload.cached_tokens, 400000)
        self.assertEqual({case.mode for case in cases}, {"rubin", "af", "pd", "draft"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad-run"
            self.assertEqual(
                main(["--batches", "0", "--skip-audit", "--output", str(output)]), 1
            )
            self.assertFalse(output.exists())
