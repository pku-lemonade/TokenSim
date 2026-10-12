"""Synchronized data-parallel workers consume messages and finish requests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


class LockstepRuntimeTest(unittest.TestCase):
    def test_parallel_workers_complete_synchronized_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                [
                    "--model",
                    str(ROOT / "data/models/moe-toy.yaml"),
                    "--cluster",
                    str(ROOT / "data/clusters/8_a100/moe_tp2_dp4.json"),
                    "--latency_backend",
                    "analytical",
                    "--batching",
                    "paged-attn",
                    "--dp_lockstep",
                    "--distribution",
                    "burst",
                    "--request_count",
                    "8",
                    "--prefill_mean_len",
                    "16",
                    "--decode_mean_len",
                    "3",
                    "--max_num_seqs",
                    "2",
                    "--verbose",
                    "none",
                    "--results_path",
                    directory,
                ]
            )
            with redirect_stdout(io.StringIO()):
                engine = main(args)
            result = json.loads((Path(directory) / "result_inf.json").read_text())
            self.assertEqual(result["termination_reason"], "completed")
            self.assertEqual(result["notdone"], [])
            self.assertEqual(len(engine.timing_recorder.timing.time), 8)
            self.assertAlmostEqual(result["output_token_ps"] * result["duration"], 24)
            self.assertIsNotNone(result["dp_step_stats"])
