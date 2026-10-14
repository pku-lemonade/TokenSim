"""Independent process execution preserves deterministic native point results."""

from dataclasses import replace
from pathlib import Path
import unittest

from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.runner import evaluate_jobs


class LPXJobsTest(unittest.TestCase):
    def test_parallel_and_sequential_points_match_and_preserve_rejections(self):
        h = HardwareContext.load(Path(__file__).resolve().parents[1] / "data")
        base = h.model("gpt-oss-2t-proxy")
        model = replace(
            base,
            num_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                base.moe, num_experts=8, num_moe_layers=2, moe_intermediate_size=64
            ),
        )
        workload = Workload(cached_tokens=16, new_tokens=2, output_tokens=4)
        case = Case("small", "rubin", Layout(a_tp=1, f_ep=1))
        settings = dict(warmup_steps=1, measurement_seconds=0.0001)
        jobs = [(model, workload, case, n, settings) for n in (1, 2)]
        jobs.append(
            (replace(model, max_position_embeddings=4), workload, case, 1, settings)
        )
        sequential = dict(evaluate_jobs(jobs))
        parallel = dict(evaluate_jobs(jobs, processes=2))
        self.assertEqual(sequential, parallel)
        self.assertEqual(parallel[2]["status"], "infeasible")
