"""Operator dataset lookup behavior and failure paths."""

from __future__ import annotations

import unittest
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import OperatorDataPackage
from tests.support.operator_data import SOURCE, ROOT, _meta, _gemm_rows


class LookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.package = OperatorDataPackage.from_rows(_meta(), {"gemm": _gemm_rows()})
        self.lookup = OperatorLookup(self.package)

    def test_exact_interpolated_and_extrapolated(self):
        exact = self.lookup.lookup(
            "gemm", {"dtype": "bfloat16", "m": 16, "n": 1024, "k": 1024}
        )
        self.assertEqual(exact.match_type, "exact")
        mid = self.lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 64, "n": 1024, "k": 1024}
        )
        self.assertEqual(mid.match_type, "interpolated")
        lo = self.lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 16, "n": 1024, "k": 1024}
        ).latency_us
        hi = self.lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 256, "n": 1024, "k": 1024}
        ).latency_us
        self.assertTrue(lo < mid.latency_us < hi)
        two_axis = self.lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 64, "n": 2048, "k": 2048}
        )
        self.assertEqual(two_axis.match_type, "interpolated")
        beyond = self.lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 1024, "n": 1024, "k": 1024}
        )
        self.assertEqual(beyond.match_type, "extrapolated")
        self.assertGreater(beyond.latency_us, hi)

    def test_analytical_extrapolation_keeps_boundary_efficiency(self):
        # Measured rows are exactly half of a synthetic "model" that grows
        # linearly with m; past the range the ratio must stay 0.5.
        def reference(table, key):
            return 100.0 * key["m"] + 1000.0

        rows = [
            {
                "dtype": "bf16",
                "m": m,
                "n": 1024,
                "k": 1024,
                "latency_us": 0.5 * reference("gemm", {"m": m}),
                "source_id": "measured:test",
            }
            for m in (8, 64, 256)
        ]
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": rows})
        lookup = OperatorLookup(
            package, LookupPolicy(extrapolate="analytical"), analytical_scaler=reference
        )
        beyond = lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024}
        )
        self.assertEqual(beyond.match_type, "extrapolated")
        self.assertIn("analytical_scaled", beyond.detail["flags"])
        self.assertAlmostEqual(beyond.latency_us, 0.5 * reference("gemm", {"m": 2048}))
        # below the range the same rule applies (linear "scale" would not shrink)
        below = lookup.lookup("gemm", {"dtype": "bf16", "m": 1, "n": 1024, "k": 1024})
        self.assertAlmostEqual(below.latency_us, 0.5 * reference("gemm", {"m": 1}))
        # a scaler that cannot price the key falls back to linear scaling
        fallback = OperatorLookup(
            package,
            LookupPolicy(extrapolate="analytical"),
            analytical_scaler=lambda table, key: None,
        )
        linear = fallback.lookup(
            "gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024}
        )
        self.assertIn("linear_scaled", linear.detail["flags"])
        self.assertAlmostEqual(
            linear.latency_us, 0.5 * reference("gemm", {"m": 256}) * 8
        )
        held = OperatorLookup(package, LookupPolicy(extrapolate="hold")).lookup(
            "gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024}
        )
        self.assertAlmostEqual(held.latency_us, 0.5 * reference("gemm", {"m": 256}))

    def test_empirical_extrapolation_fits_sublinear_scaling(self):
        # Measured rows grow as m^0.5 (sub-linear); empirical mode should
        # recover this exponent and produce lower estimates than analytical.
        import math as _math

        rows = [
            {
                "dtype": "bf16",
                "m": m,
                "n": 1024,
                "k": 1024,
                "latency_us": 10.0 * _math.sqrt(m),
                "source_id": "measured:test",
            }
            for m in (4, 16, 64, 256)
        ]
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": rows})
        # Explicit empirical mode.
        empirical = OperatorLookup(package, LookupPolicy(extrapolate="empirical"))
        result = empirical.lookup(
            "gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024}
        )
        self.assertEqual(result.match_type, "extrapolated")
        self.assertIn("empirical_scaled", result.detail["flags"])
        # The empirical fit should detect alpha~0.5, so the estimate at
        # m=2048 should be close to 10 * sqrt(2048) ≈ 452.5.
        expected = 10.0 * _math.sqrt(2048)
        self.assertAlmostEqual(result.latency_us, expected, delta=expected * 0.1)
        # In contrast, analytical (linear) would give: boundary_lat * (2048/256)
        # = 10*sqrt(256) * 8 = 160 * 8 = 1280, much higher.
        analytical = OperatorLookup(
            package,
            LookupPolicy(extrapolate="analytical"),
            analytical_scaler=lambda table, key: float(key["m"]),
        )
        ana_result = analytical.lookup(
            "gemm", {"dtype": "bf16", "m": 2048, "n": 1024, "k": 1024}
        )
        self.assertGreater(ana_result.latency_us, result.latency_us * 1.5)

    def test_empirical_falls_back_to_analytical_with_few_points(self):
        # Only 2 measured points — too few for fitting.
        def reference(table, key):
            return 100.0 * key["m"]

        rows = [
            {
                "dtype": "bf16",
                "m": m,
                "n": 1024,
                "k": 1024,
                "latency_us": 50.0 * m,
                "source_id": "measured:test",
            }
            for m in (8, 64)
        ]
        package = OperatorDataPackage.from_rows(_meta(), {"gemm": rows})
        lookup = OperatorLookup(package, analytical_scaler=reference)
        result = lookup.lookup(
            "gemm", {"dtype": "bf16", "m": 512, "n": 1024, "k": 1024}
        )
        self.assertEqual(result.match_type, "extrapolated")
        # Should fall back to analytical since <3 points.
        self.assertIn("analytical_scaled", result.detail["flags"])

    def test_missing_discrete_key_and_disabled_extrapolation_fail(self):
        with self.assertRaises(MissingOperatorDataError):
            self.lookup.lookup("gemm", {"dtype": "fp8", "m": 16, "n": 1024, "k": 1024})
        strict = OperatorLookup(self.package, LookupPolicy(extrapolate="none"))
        with self.assertRaises(MissingOperatorDataError):
            strict.lookup("gemm", {"dtype": "bf16", "m": 1024, "n": 1024, "k": 1024})
        with self.assertRaises(MissingOperatorDataError):
            self.lookup.lookup(
                "gemm", {"dtype": "bf16", "m": 1024 * 1024, "n": 1024, "k": 1024}
            )
        with self.assertRaises(MissingOperatorDataError):
            self.lookup.lookup("gemm", {"dtype": "bf16", "m": 1})
