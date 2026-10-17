"""Sparse normalization runs the provider cache's complete behavior contracts."""

from scripts.af_probe_sparse import installed
from tests.af.providers import test_cache


class SparseCacheFastPathTest(test_cache.CacheFastPathTest):
    def run(self, result=None):
        with installed():
            return super().run(result)
