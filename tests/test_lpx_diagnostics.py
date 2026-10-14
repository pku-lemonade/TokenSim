"""Diagnostic interventions preserve identities and distinguish latency terms."""

from pathlib import Path
import unittest

from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Case, Layout
from scripts.lpx.diagnostics import altered_hardware, latency_probes
from tests.support.hardware import make_model
from tests.af.support.model import _moe


class LPXDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = make_model(
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=8,
            vocab_size=64,
            moe=_moe(),
        )

    def test_probes_preserve_sources_and_conserve_breakdown(self):
        original = dict(self.hardware.device("rubin").analytical)
        case = Case("small", "rubin", Layout(a_tp=1, f_ep=1))
        result = latency_probes(self.hardware, self.model, case, context=256)
        self.assertEqual(self.hardware.device("rubin").analytical, original)
        self.assertAlmostEqual(
            result["baseline"]["step_us"], sum(result["breakdown_us"].values())
        )
        self.assertGreater(result["rubin_launch_contribution_us"], 0)
        self.assertGreater(
            result["baseline"]["step_us"],
            result["zero_rubin_launch_lower_bound"]["step_us"],
        )
        self.assertGreater(
            result["baseline"]["step_us"],
            result["alternating_local_full_latency_probe"]["step_us"],
        )

    def test_launch_intervention_does_not_change_lp30(self):
        changed = altered_hardware(self.hardware, zero_rubin_launch=True)
        self.assertEqual(changed.device("groq_lp30"), self.hardware.device("groq_lp30"))
        self.assertEqual(changed.device("rubin").analytical["kernel_launch_us"], 0)
