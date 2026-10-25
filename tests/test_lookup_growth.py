"""Analytical transfers retain the complete coordinates of each measured anchor."""

import math
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.latency.operator_table import (
    OperatorResolver,
    OperatorTableLatencyBackend,
)
from TokenSim.operator_data.analytical import analytical_model_for
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.support.hardware import make_device, make_model


class LookupGrowthTest(unittest.TestCase):
    def setUp(self):
        self.device = make_device()
        self.model = analytical_model_for(self.device.family)
        self.key = dict(
            attn_dtype="bf16",
            kv_cache_dtype="bf16",
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            window_size=0,
            batch_size=1,
            context_len=256,
        )

    def reference(self, table, key):
        return self.model.estimate(table, key, self.device).latency_us

    def package(self, table, keys, reference=None):
        reference = reference or self.reference
        source = SourceRecord("synthetic", "D", "analytical")
        return OperatorDataPackage.from_rows(
            PackageMeta(
                "growth-v1",
                self.device.device_id,
                "test",
                sources={source.source_id: source},
            ),
            {
                table: [
                    {
                        **k,
                        "latency_us": reference(table, k),
                        "source_id": source.source_id,
                    }
                    for k in keys
                ]
            },
        )

    def test_roofline_transfer_matches_target_on_both_axes(self):
        package = self.package("generation_attention", [self.key])
        lookup = OperatorLookup(package, analytical_scaler=self.reference)
        for batch, context in ((16, 1024), (4, 64), (1, 512)):
            key = dict(self.key, batch_size=batch, context_len=context)
            original = dict(key)
            with self.subTest(batch=batch, context=context):
                result = lookup.lookup("generation_attention", key)
                self.assertAlmostEqual(
                    result.latency_us, self.reference(result.table, key)
                )
                self.assertEqual(key, original)
                self.assertEqual(result.source_id, "synthetic")
                self.assertEqual(result.match_type, "extrapolated")

    def test_nonseparable_three_axis_transfer(self):
        anchor = dict(dtype="bf16", m=4, n=8, k=16)

        def reference(table, key):
            return 5 + key["m"] * key["n"] + key["n"] * key["k"]

        package = self.package("gemm", [anchor], reference)
        lookup = OperatorLookup(package, analytical_scaler=reference)
        for target in (dict(anchor, m=16, n=32, k=64), dict(anchor, m=1, n=32, k=8)):
            with self.subTest(target=target):
                self.assertAlmostEqual(
                    lookup.lookup("gemm", target).latency_us, reference("gemm", target)
                )

    def test_interpolation_resolves_inner_growth_at_each_outer_node(self):
        def reference(table, key):
            return 4 + key["batch_size"] * key["context_len"]

        keys = [
            dict(self.key, batch_size=b, context_len=c) for b, c in ((1, 4), (4, 8))
        ]
        package = self.package("generation_attention", keys, reference)
        key = dict(self.key, batch_size=2, context_len=16)
        result = OperatorLookup(package, analytical_scaler=reference).lookup(
            "generation_attention", key
        )
        expected = (
            reference("", dict(key, batch_size=1))
            + reference("", dict(key, batch_size=4))
        ) / 2
        self.assertAlmostEqual(result.latency_us, expected)

    def test_empirical_samples_bind_the_sampled_outer_coordinate(self):
        def reference(table, key):
            return 4 + key["batch_size"] * key["context_len"]

        keys = [dict(self.key, batch_size=b, context_len=4) for b in (1, 2, 4)]
        package = self.package("generation_attention", keys, reference)
        key = dict(self.key, batch_size=8, context_len=16)
        result = OperatorLookup(
            package, LookupPolicy(extrapolate="empirical"), reference
        ).lookup("generation_attention", key)
        alpha = math.log((4 + 4 * 16) / (4 + 16)) / math.log(4)
        self.assertAlmostEqual(result.latency_us, (4 + 4 * 16) * 2**alpha)

    def test_ordinary_and_af_consumers_share_corrected_semantics(self):
        package = self.package("generation_attention", [self.key])
        key = dict(self.key, batch_size=16, context_len=1024)
        for consumer in (
            OperatorResolver(device=self.device, package=package),
            OperatorTableLatencyBackend(
                device=self.device, model=make_model(), package=package
            ),
        ):
            with self.subTest(consumer=type(consumer).__name__):
                self.assertAlmostEqual(
                    consumer.query("generation_attention", key).latency_us,
                    self.reference("generation_attention", key),
                )

    def test_exact_values_and_policy_boundaries(self):
        package = self.package("generation_attention", [self.key])
        exact = OperatorLookup(package).lookup("generation_attention", self.key)
        self.assertEqual(exact.latency_us, self.reference(exact.table, self.key))
        self.assertEqual(exact.match_type, "exact")
        key = dict(self.key, batch_size=32)
        with self.assertRaises(MissingOperatorDataError):
            OperatorLookup(package, analytical_scaler=self.reference).lookup(
                exact.table, key
            )
        with self.assertRaises(MissingOperatorDataError):
            OperatorLookup(package, LookupPolicy(extrapolate="none")).lookup(
                exact.table, dict(self.key, batch_size=2)
            )
        held = OperatorLookup(package, LookupPolicy(extrapolate="hold"), self.reference)
        self.assertEqual(
            held.lookup(exact.table, dict(self.key, batch_size=2)).latency_us,
            exact.latency_us,
        )

    def test_invalid_analytical_reference_retains_scale_fallback(self):
        package = self.package("generation_attention", [self.key])
        key = dict(self.key, batch_size=2, context_len=512)
        for value in (None, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                result = OperatorLookup(
                    package, analytical_scaler=lambda *args: value
                ).lookup("generation_attention", key)
                self.assertAlmostEqual(
                    result.latency_us, 4 * self.reference(result.table, self.key)
                )

    def test_axis_binding_requires_boolean(self):
        with self.assertRaises(ConfigurationError):
            OperatorLookup(
                self.package("generation_attention", [self.key]),
                bind_analytical_axes="false",
            )
