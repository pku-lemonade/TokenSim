"""Identity reuse retains structural cache, validation and mutable input contracts."""

from scripts.af_probe_frontcache import HashedKey, installed
from tests.af.providers import test_cache
from tests.af.support.model import _compute_query


class FrontCacheTest(test_cache.CacheFastPathTest):
    def run(self, result=None):
        with installed():
            return super().run(result)

    def test_evicted_service_entries_are_recomputed(self):
        provider = self.compute()
        query = _compute_query()
        expected = provider.estimate(query)
        self.assertEqual(provider.estimate(query), expected)
        hits = provider.estimate_cache_hits
        provider._estimate_cache.clear()
        self.assertEqual(provider.estimate(query), expected)
        self.assertEqual(provider.estimate_cache_hits, hits)

    def test_prehashed_keys_keep_tuple_equality_and_dictionary_interoperation(self):
        structural = ("namespace", (1, 2, ("nested", 3)))
        key = HashedKey(structural)
        self.assertEqual(hash(key), hash(structural))
        self.assertEqual(key, structural)
        self.assertEqual(structural, key)
        self.assertEqual({structural: 1}[key], 1)
        self.assertEqual({key: 2}[structural], 2)
        self.assertEqual({key: 3}[HashedKey(structural)], 3)

    def test_mutable_query_children_change_the_service_key(self):
        provider = self.compute()
        query = _compute_query(context=[8, 8])
        provider.estimate(query)
        query.context_lengths[0] = 16
        hits = provider.estimate_cache_hits
        provider.estimate(query)
        self.assertEqual(provider.estimate_cache_hits, hits)
