"""AA serial requests, native token conversion and physical fit boundaries."""

from pathlib import Path
import unittest

from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.aa import serial, scaled_efficiency
from scripts.lpx.config import Layout


class AASerialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = ModelSpec(
            "tiny",
            32,
            64,
            2,
            4,
            2,
            8,
            vocab_size=64,
            dtype="bf16",
            max_position_embeddings=1024,
        )
        cls.layout = Layout(a_device="groq_lp30", f_device="groq_lp30", a_tp=2, f_ep=2)

    def test_single_client_excludes_warmup_and_preserves_distinct_requests(self):
        result = serial(
            self.hardware,
            self.model,
            self.layout,
            input_tokens=10,
            output_tokens=8,
            requests=3,
            warmup=2,
            prefill_chunk=4,
            rho_in=2,
            rho_out=0.5,
        )
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["clients"], 1)
        rows = result["request_timings"]
        self.assertEqual([row["request_id"] for row in rows], list(range(5)))
        self.assertEqual(
            [row["warmup"] for row in rows], [True, True, False, False, False]
        )
        for row in rows:
            self.assertEqual(row["native_input_tokens"], 20)
            self.assertEqual(row["native_output_tokens"], 4)
            self.assertEqual(
                row["standardized_generation_tps"], 2 * row["native_generation_tps"]
            )
            self.assertEqual(row["submitted_at"], row["admitted_at"])
        for previous, following in zip(rows, rows[1:]):
            self.assertEqual(previous["last_output_at"], following["submitted_at"])
        self.assertEqual(
            result["operator_statistics"]["operator_step_count"], 5 * (5 + 3)
        )

    def test_efficiency_fit_cannot_exceed_peak(self):
        scaled = scaled_efficiency(self.hardware, 0.5)
        self.assertEqual(scaled.device("groq_lp30").analytical["gemm_mfu"], 0.4)
        self.assertEqual(self.hardware.device("groq_lp30").analytical["gemm_mfu"], 0.8)
        with self.assertRaises(ConfigurationError):
            scaled_efficiency(self.hardware, 2)


if __name__ == "__main__":
    unittest.main()
