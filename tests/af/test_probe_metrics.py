"""Timing-only mode preserves scalar simulation and provider observations."""

import unittest

from scripts import af_probe_metrics, run_deepseek_mla
from scripts.deepseek_v3_runtime import configuration


class MetricsOnlyTest(unittest.TestCase):
    def test_summary_values_and_queries_match_with_empty_trace(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        results = []
        for disabled in (False, True):
            if disabled:
                context = af_probe_metrics.installed()
            else:
                from contextlib import nullcontext

                context = nullcontext()
            with context:
                executor, rows, compute, transfer = run_deepseek_mla.simulate(
                    config, 2, 8, steps=8, trace=True, fallback="analytical_only"
                )
                try:
                    self.assertEqual(len(executor.trace.events) == 0, disabled)
                    results.append((rows, compute.stats_dict(), transfer.describe()))
                finally:
                    executor.trace.close()
        self.assertEqual(results[0], results[1])
