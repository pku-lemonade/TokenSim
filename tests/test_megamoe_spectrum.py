"""Collector-family integer snapshots, routing constraints and RNG isolation."""

import json
from pathlib import Path
from unittest import TestCase

import numpy as np

from TokenSim.errors import ConfigurationError
from TokenSim.latency.deepseek_v4_moe_spectrum import integer_spectra, spectrum_geometry


class IntegerSpectrumTest(TestCase):
    def test_port_matches_original_cpu_helper_snapshots(self):
        data = json.loads(Path("data/reference/agentx-moe-spectrum/collector-count-snapshots.json").read_text())
        for case in data["cases"]:
            with self.subTest(global_rows=case["global_rows"], ep=case["ep"]):
                actual = integer_spectra(case["global_rows"], 384, 6, case["ep"])
                np.testing.assert_array_equal(actual, np.asarray(case["counts"]))

    def test_small_routes_have_exact_integer_selections_and_unique_topk(self):
        for rows in (1, 2, 8, 17, 64, 128):
            with self.subTest(rows=rows):
                samples = integer_spectra(rows, 384, 6, 8)
                self.assertEqual(samples.dtype, np.dtype("int64"))
                for sample in samples:
                    counts = sample.reshape(-1)
                    self.assertEqual(int(counts.sum()), rows * 6)
                    self.assertTrue(np.all((counts >= 0) & (counts <= rows)))
                    order = np.argsort(-counts)
                    tokens = np.repeat(order, counts[order]).reshape(6, rows).T
                    self.assertTrue(np.all(np.diff(np.sort(tokens, axis=1), axis=1) > 0))

    def test_rank_variance_is_retained_and_collector_max_load_moves_to_zero(self):
        samples = integer_spectra(128, 384, 6, 8)
        loads = samples.sum(axis=2)
        self.assertTrue(np.all(loads[:, 0] == loads.max(axis=1)))
        self.assertTrue(np.any(loads != loads[:, :1]))
        self.assertFalse(np.array_equal(samples[0], samples[1]))

    def test_geometry_keeps_integer_task_tails_and_readonly_cache(self):
        samples = integer_spectra(128, 384, 6, 8)
        geometry = spectrum_geometry(128, 384, 6, 8)
        np.testing.assert_array_equal(geometry.mma_rows, (((samples + 15) // 16) * 16).sum(axis=2))
        self.assertTrue(np.all(geometry.mma_rows >= geometry.selections))
        for array in (samples, geometry.mma_rows, geometry.selections, geometry.m_tiles):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = -1

    def test_local_draws_do_not_advance_global_numpy_rng(self):
        before = np.random.get_state()
        integer_spectra(131, 384, 6, 8)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_empty_and_invalid_geometry_are_explicit(self):
        self.assertTrue(np.all(integer_spectra(0, 384, 6, 8) == 0))
        for args in ((-1, 384, 6, 8), (8, 384, 0, 8), (8, 384, 385, 8), (8, 384, 6, 5)):
            with self.subTest(args=args), self.assertRaises(ConfigurationError):
                integer_spectra(*args)
