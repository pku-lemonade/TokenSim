"""Opt-in larger archive blocks without changing explicit caller settings."""

from contextlib import contextmanager

from TokenSim.af.trace.archive_store import TraceArchiveStore


@contextmanager
def installed(output=None, chunk_size=1024):
    original = TraceArchiveStore.__init__

    def initialize(self, **options):
        options.setdefault("chunk_size", chunk_size)
        original(self, **options)

    TraceArchiveStore.__init__ = initialize
    try:
        yield
    finally:
        TraceArchiveStore.__init__ = original
