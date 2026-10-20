"""Independent process execution preserves deterministic native point results."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.runner import evaluate, evaluate_jobs


class LPXJobsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware = HardwareContext.load(Path(__file__).resolve().parents[1] / "data")
        base = hardware.model("gpt-oss-2t-expert-proxy")
        cls.model = replace(
            base,
            num_layers=2,
            attention_windows=(),
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

    def test_parallel_and_sequential_points_match_and_preserve_rejections(self):
        model = self.model
        workload = Workload(cached_tokens=16, new_tokens=2, output_tokens=4)
        case = Case("small", "rubin", Layout(a_tp=1, f_ep=1))
        settings = dict(warmup_steps=1, measurement_seconds=0.0001)
        jobs = [(model, workload, case, n, settings) for n in (1, 2)]
        jobs.append(
            (replace(model, max_position_embeddings=4), workload, case, 1, settings)
        )
        separate = Case(
            "separate",
            "af",
            Layout(a_tp=1, f_device="groq_lp30", f_ep=2, colocated=False),
            prefill=Layout(a_tp=1, f_ep=1),
        )
        jobs.append((model, replace(workload, seed=43), separate, 2, settings))
        sequential = dict(evaluate_jobs(jobs))
        parallel = dict(evaluate_jobs(jobs, processes=2))
        self.assertEqual(sequential, parallel)
        self.assertEqual(parallel[2]["status"], "infeasible")
        self.assertEqual(parallel[3]["status"], "ok")

    def test_parallel_duplicates_commit_once_and_reuse_without_execution(self):
        model = self.model
        workload = Workload(cached_tokens=16, new_tokens=2, output_tokens=4)
        case = Case("cached", "rubin", Layout(a_tp=1, f_ep=1))
        settings = dict(warmup_steps=1, measurement_seconds=1e-9)
        jobs = [(model, workload, case, count, settings) for count in (1, 1, 2)]
        expected = dict(evaluate_jobs(jobs))
        with TemporaryDirectory() as directory:
            cached = [
                (*job[:4], dict(job[4], checkpoint_output=directory)) for job in jobs
            ]
            actual = dict(evaluate_jobs(cached, processes=2))
            for index, row in expected.items():
                self.assertEqual(json.loads(json.dumps(row)), actual[index])
            self.assertEqual(len(list(Path(directory).glob("*/manifest.json"))), 2)
            with patch(
                "scripts.lpx.runner.point", side_effect=AssertionError("executed")
            ):
                self.assertEqual(dict(evaluate_jobs(cached)), actual)
                changed = list(cached[0])
                changed[1] = replace(workload, seed=workload.seed + 1)
                with self.assertRaisesRegex(AssertionError, "executed"):
                    list(evaluate_jobs([tuple(changed)]))
                forced = (*cached[0][:4], dict(cached[0][4], resume=False))
                with self.assertRaisesRegex(AssertionError, "executed"):
                    list(evaluate_jobs([forced]))

    def test_reference_results_reuse_the_same_complete_checkpoint(self):
        workload = Workload(cached_tokens=16, new_tokens=2, output_tokens=4)
        case = Case("reference", "rubin", Layout(a_tp=1, f_ep=1))
        with TemporaryDirectory() as directory:
            settings = dict(
                warmup_steps=1, measurement_seconds=1e-9, checkpoint_output=directory
            )
            job = (self.model, workload, case, 1, settings)
            full = evaluate(job)
            with patch(
                "scripts.lpx.runner.point", side_effect=AssertionError("executed")
            ):
                reference = evaluate(
                    (*job[:4], dict(settings, result_format="reference"))
                )
            self.assertTrue(reference["reused"])
            self.assertEqual(reference["point_audit"], "passed")
            self.assertEqual(
                reference["output_tokens_per_s"], full["output_tokens_per_s"]
            )
            self.assertNotIn("token_events", reference)
            self.assertEqual(
                json.loads(Path(reference["result_file"]).read_text()), full
            )
            self.assertEqual(
                json.loads(Path(reference["manifest_file"]).read_text())["key"],
                reference["checkpoint_key"],
            )
