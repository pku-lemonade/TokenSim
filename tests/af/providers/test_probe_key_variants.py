"""Diagnostic keys preserve mutable descendants, scalar types and cache hits."""

from dataclasses import dataclass, replace
import unittest

from TokenSim.af.caching import estimates
from scripts import (
    af_probe_inlinescalars,
    af_probe_keyhash,
    af_probe_namespacehash,
    af_probe_namespaceplain,
    af_probe_tupletypes,
)
from tests.af.providers import test_cache
from tests.af.support.model import _compute_query


@dataclass(frozen=True)
class Record:
    value: object


PROBES = (
    af_probe_inlinescalars,
    af_probe_keyhash,
    af_probe_tupletypes,
    af_probe_namespacehash,
)


class ProbeKeyVariantsTest(unittest.TestCase):
    def test_structural_values_and_mutable_children_preserve_reference_keys(self):
        for probe in PROBES:
            with self.subTest(probe=probe.__name__), probe.installed():
                memo = estimates.FingerprintMemo(capacity=4)
                child = {"values": [1, 2]}
                mutable = Record((child,))
                values = [
                    (1, True, 1.0, None),
                    (),
                    Record(estimates.FrozenDict({"v": (1, 2.0)})),
                    mutable,
                    [Record("x")],
                ]
                for _ in range(2):
                    for value in values:
                        self.assertEqual(memo(value), estimates.identity(value))
                        self.assertLessEqual(len(memo.values), 4)
                before = memo(mutable)
                child["values"][0] = True
                self.assertNotEqual(memo(mutable), before)
                self.assertNotIn(id(mutable), memo.values)
                self.assertEqual(len({memo((v,)) for v in (1, True, 1.0)}), 3)

    def test_hot_service_cache_revisits_mutable_queries_and_retains_hits(self):
        for probe in PROBES:
            with self.subTest(probe=probe.__name__), probe.installed():
                provider = test_cache.CacheFastPathTest().compute()
                query = _compute_query(context=[8, 8])
                first = provider.estimate(query)
                self.assertIs(provider.estimate(replace(query, iteration_id=9)), first)
                hits = provider.estimate_cache_hits
                query.context_lengths[0] = 16
                self.assertIsNot(provider.estimate(query), first)
                self.assertEqual(provider.estimate_cache_hits, hits)

    def test_hash_reuse_keeps_tuple_dictionary_interoperation(self):
        structural = (("x", (1, 2)), ("y", (float, 1.0)))
        cached = af_probe_keyhash.HashedTuple(structural)
        self.assertEqual(hash(cached), hash(structural))
        self.assertEqual({structural: 1}[cached], 1)
        self.assertEqual({cached: 2}[structural], 2)

    def test_probe_scopes_restore_original_normalizers(self):
        original_class = estimates.FingerprintMemo
        original_method = original_class._visit
        for probe in PROBES:
            with probe.installed():
                memo = estimates.FingerprintMemo()
                self.assertEqual(
                    memo(Record((True, 1))), estimates.identity(Record((True, 1)))
                )
            self.assertIs(estimates.FingerprintMemo, original_class)
            self.assertIs(original_class._visit, original_method)

    def test_namespace_extensions_invalidate_hits_with_complete_structural_equality(
        self,
    ):
        with af_probe_namespaceplain.installed(), af_probe_namespacehash.installed():
            provider = test_cache.CacheFastPathTest().compute()
            namespace = provider._estimate_namespace
            self.assertEqual(namespace, type(namespace)(namespace.values))
            first = provider.estimate(_compute_query())
            provider._estimate_namespace += (("profile", "updated"),)
            self.assertNotEqual(namespace, provider._estimate_namespace)
            self.assertIsNot(provider.estimate(_compute_query()), first)
            self.assertEqual(provider.estimate_cache_hits, 0)
