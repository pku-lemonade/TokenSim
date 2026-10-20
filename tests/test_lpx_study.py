"""Study layouts retain model-specific partitions and explicit rack inventories."""

import unittest

from scripts.run_lpx_study import study_cases, extension_candidates


class LPXStudyTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
