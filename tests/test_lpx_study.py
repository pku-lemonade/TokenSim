"""Study layouts retain model-specific partitions and explicit rack inventories."""

import unittest

from scripts.run_lpx_study import study_cases, extension_candidates, formal_candidates
from scripts.lpx.config import with_collective_algorithms


class LPXStudyTest(unittest.TestCase):
    def test_policy_override_reaches_every_pool_and_preserves_inventory(self):
        _, cases = study_cases("gpt-oss-2t-expert-proxy")
        changed = with_collective_algorithms(cases, gpu="auto", lpu="direct")
        for before, after in zip(cases, changed):
            self.assertEqual(before.inventory(), after.inventory())
            for role in ("decode", "prefill", "draft"):
                layout = getattr(after, role)
                if layout is None:
                    continue
                for domain in ("a", "f"):
                    device = getattr(layout, f"{domain}_device")
                    self.assertEqual(
                        getattr(layout, f"{domain}_collective_algorithm"),
                        "auto" if device == "rubin" else "direct",
                    )
        self.assertTrue(all(c.decode.a_collective_algorithm == "ring" for c in cases))

    def test_wide_study_reduces_ffn_inventory_without_changing_target(self):
        workload, cases = study_cases("gpt-oss-2t-wide-proxy")
        cases = {case.name: case for case in cases}
        self.assertEqual(
            (workload.cached_tokens, workload.new_tokens, workload.output_tokens),
            (400000, 4000, 400),
        )
        self.assertEqual(cases["af-tp8"].inventory()["groq_lp30"], 4096)
        self.assertEqual(cases["af-f4-ep128-pp5"].inventory()["groq_lp30"], 2560)
        self.assertEqual(
            cases["draft-lpx256"].inventory(), {"rubin": 8, "groq_lp30": 256}
        )
        self.assertEqual(
            cases["draft-pipeline-lpx256"].inventory(), {"rubin": 64, "groq_lp30": 256}
        )
        self.assertEqual(cases["pd-compact"].decode.a_pp, 40)

    def test_expert_study_uses_its_own_layer_and_expert_counts(self):
        _, cases = study_cases("gpt-oss-2t-expert-proxy")
        names = [case.name for case in cases]
        self.assertEqual(len(names), len(set(names)))
        cases = {case.name: case for case in cases}
        self.assertEqual(cases["rubin-pipeline"].inventory()["rubin"], 72)
        self.assertEqual(cases["af-f4-ep160-pp4"].decode.f_ep, 160)
        self.assertEqual(cases["af-f4-ep160-pp4"].inventory()["groq_lp30"], 2560)
        self.assertEqual(cases["pd-compact"].decode.a_pp, 18)

    def test_user_scan_extends_only_a_feasible_growing_boundary(self):
        rows = [
            dict(case="grow", status="ok", batch=8, output_tokens_per_s=100),
            dict(case="grow", status="ok", batch=16, output_tokens_per_s=110),
            dict(case="flat", status="ok", batch=8, output_tokens_per_s=100),
            dict(case="flat", status="ok", batch=16, output_tokens_per_s=101),
            dict(case="full", status="infeasible", batch=16),
        ]
        self.assertEqual(extension_candidates(rows, 16), {"grow"})

    def test_formal_selection_retains_frontiers_for_both_power_scopes(self):
        rows = [
            dict(
                case="small",
                mode="rubin",
                batch=1,
                status="ok",
                inventory={"rubin": 8},
                power_watts=14400,
                tokens_per_s_per_user=1000,
                output_tokens_per_s=1000,
            ),
            dict(
                case="pipeline",
                mode="rubin",
                batch=32,
                status="ok",
                inventory={"rubin": 72},
                power_watts=129600,
                tokens_per_s_per_user=800,
                output_tokens_per_s=3000,
            ),
            dict(case="full", mode="rubin", batch=64, status="infeasible"),
        ]
        self.assertEqual(formal_candidates(rows), {("small", 1), ("pipeline", 32)})
        self.assertNotIn("power_scope", rows[0])
        # Selection also supports a study containing only AF candidates.
        for row in rows:
            row["mode"] = "af"
        self.assertEqual(formal_candidates(rows), {("small", 1), ("pipeline", 32)})


if __name__ == "__main__":
    unittest.main()
