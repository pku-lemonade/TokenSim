"""Configuration hash reuse preserves full equality and namespace invalidation."""

from dataclasses import FrozenInstanceError
import unittest

from TokenSim.af.caching.namespace import EstimateNamespace
from tests.af.providers import test_cache
from tests.af.support.model import _compute_query


class CacheNamespaceTest(unittest.TestCase):
    def test_equal_hashes_do_not_alias_distinct_configuration_values(self):
        # Python deliberately gives these unequal integers the same hash.
        left, right = EstimateNamespace((-1,)), EstimateNamespace((-2,))
        self.assertEqual(hash(left), hash(right))
        self.assertNotEqual(left, right)
        cache = {left: "left", right: "right"}
        self.assertEqual(cache[EstimateNamespace((-1,))], "left")
        self.assertEqual(cache[EstimateNamespace((-2,))], "right")

    def test_extension_keeps_previous_namespace_immutable_and_recomputes_hash(self):
        original = EstimateNamespace(("fallback", ("scope", "a")))
        extended = original + (("profile", "new"),)
        self.assertEqual(original.values, ("fallback", ("scope", "a")))
        self.assertEqual(extended.values, original.values + (("profile", "new"),))
        self.assertEqual(hash(extended), hash(extended.values))
        self.assertEqual(extended, EstimateNamespace(extended.values))
        with self.assertRaises(FrozenInstanceError):
            original.digest = 0
        with self.assertRaises(FrozenInstanceError):
            original.values = ()

    def test_service_cache_checks_full_namespace_after_a_hash_collision(self):
        provider = test_cache.CacheFastPathTest().compute()
        provider._estimate_namespace = EstimateNamespace((-1,))
        query = _compute_query()
        first = provider.estimate(query)
        self.assertIs(provider.estimate(query), first)
        hits = provider.estimate_cache_hits
        provider._estimate_namespace = EstimateNamespace((-2,))
        self.assertIsNot(provider.estimate(query), first)
        self.assertEqual(provider.estimate_cache_hits, hits)
