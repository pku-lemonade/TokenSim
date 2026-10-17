"""Larger archive blocks retain history, snapshot and transaction contracts."""

from scripts.af_probe_archive import installed
from tests.af.trace import test_archive
from TokenSim.af.trace.archive_store import TraceArchiveStore


class ArchiveBlocksTest(test_archive.TraceArchiveTest):
    def run(self, result=None):
        with installed(chunk_size=4096):
            return super().run(result)

    def test_default_changes_are_scoped_and_explicit_bounds_are_preserved(self):
        self.assertEqual(TraceArchiveStore().chunk_size, 4096)
        self.assertEqual(TraceArchiveStore(chunk_size=17).chunk_size, 17)
        with installed(chunk_size=1024):
            self.assertEqual(TraceArchiveStore().chunk_size, 1024)
        self.assertEqual(TraceArchiveStore().chunk_size, 4096)
