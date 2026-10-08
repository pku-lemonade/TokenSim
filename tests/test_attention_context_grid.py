"""Query/prefix diagonal-grid interpolation and real source-floor regressions."""

from unittest import TestCase

from TokenSim.operator_data.schema import TABLE_SPECS
import tests.test_native_attention as native_attention


class AttentionContextGridTest(TestCase):
    @classmethod
    def setUpClass(cls):
        native_attention.NativeAttentionTest.setUpClass()

    def estimator(self, sku="b200", tp=1):
        return native_attention.NativeAttentionTest().backend(sku, tp=tp, native_moe=False).native_attention

    def key(self, estimator, kind, queries, prefix):
        row = next(row for row in estimator.backend.package.tables["dsv4_attention"]
                   if row["phase"] == "context" and row["layer_kind"] == kind
                   and row["tp_size"] == estimator.tp and row["batch_size"] == 1
                   and row["input_seq_len"] == 512)
        return {**{field: row[field] for field in TABLE_SPECS["dsv4_attention"].key_fields},
                "input_seq_len": queries, "context_len": prefix}

    def test_diagonal_prefix_does_not_replace_a_supported_smaller_query(self):
        for sku in ("b200", "b300"):
            for tp in (1, 2, 4, 8):
                estimator = self.estimator(sku, tp)
                for kind in ("csa", "hca"):
                    with self.subTest(sku=sku, tp=tp, kind=kind):
                        key = self.key(estimator, kind, 512, 49152)
                        expected = estimator.lookup.lookup("dsv4_attention", key)
                        actual = estimator._query(kind, "context", 1, 512, 49152)
                        self.assertEqual(actual.source_query_tokens, 512)
                        self.assertEqual(actual.source_batch_size, 1)
                        self.assertEqual(actual.latency_us, expected.latency_us)
                        self.assertEqual(actual.match, expected.match_type)

    def test_short_context_tails_keep_their_actual_query_width(self):
        estimator = self.estimator()
        for kind in ("csa", "hca"):
            for queries, prefix in ((256, 61440), (128, 65024), (64, 65280)):
                with self.subTest(kind=kind, queries=queries, prefix=prefix):
                    expected = estimator.lookup.lookup("dsv4_attention", self.key(estimator, kind, queries, prefix))
                    actual = estimator._query(kind, "context", 1, queries, prefix)
                    self.assertEqual(actual.source_query_tokens, queries)
                    self.assertEqual(actual.latency_us, expected.latency_us)

    def test_one_token_prefix_change_does_not_cause_false_query_jump(self):
        estimator = self.estimator()
        for kind in ("csa", "hca"):
            values = [estimator._query(kind, "context", 1, 512, prefix)
                      for prefix in (49151, 49152, 49153)]
            self.assertTrue(all(value.source_query_tokens == 512 for value in values))
            self.assertLess(max(v.latency_us for v in values) / min(v.latency_us for v in values), 1.001)

    def test_coupled_measured_points_remain_exact(self):
        for sku in ("b200", "b300"):
            estimator = self.estimator(sku)
            for kind in ("csa", "hca"):
                for queries, prefix in ((16384, 49152), (4096, 61440), (512, 65024)):
                    with self.subTest(sku=sku, kind=kind, queries=queries):
                        expected = estimator.lookup.lookup("dsv4_attention", self.key(estimator, kind, queries, prefix))
                        self.assertEqual(expected.match_type, "exact")
                        actual = estimator._query(kind, "context", 1, queries, prefix)
                        self.assertEqual(actual.source_query_tokens, queries)
                        self.assertEqual(actual.latency_us, expected.latency_us)
                        self.assertEqual(actual.match, "exact")

    def test_genuine_zero_prefix_query_and_generation_batch_floors_are_retained(self):
        estimator = self.estimator("b300")
        context = estimator._query("csa", "context", 1, 1, 0)
        self.assertGreater(context.source_query_tokens, 1)
        self.assertEqual(context.match, "clamped")
        generation = estimator._query("csa", "generation", 4, 1, 8192)
        self.assertEqual(generation.source_batch_size, 64)
        self.assertEqual(generation.source_query_tokens, 1)
        self.assertEqual(generation.match, "clamped")

    def test_profile_reset_does_not_change_queries_or_table_rows(self):
        estimator = self.estimator()
        rows = estimator.backend.package.tables["dsv4_attention"]
        before = estimator._query("csa", "context", 1, 512, 49152)
        estimator.reset_stats()
        after = estimator._query("csa", "context", 1, 512, 49152)
        self.assertEqual(before, after)
        self.assertIs(estimator.backend.package.tables["dsv4_attention"], rows)
        self.assertEqual(estimator.lookup.policy.max_extrapolation_ratio, 64)
