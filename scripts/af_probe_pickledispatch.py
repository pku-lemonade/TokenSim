"""Diagnostic C pickler dispatch table for immutable provider dictionaries."""

from contextlib import contextmanager
import copyreg
import pickle

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.trace import archive_store


def _reduce_frozen(value):
    return FrozenDict, (dict(value),)


class DispatchPickler(pickle.Pickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatch_table = dict(copyreg.dispatch_table)
        self.dispatch_table[FrozenDict] = _reduce_frozen


@contextmanager
def installed(output=None):
    original = archive_store._TracePickler
    archive_store._TracePickler = DispatchPickler
    try:
        yield
    finally:
        archive_store._TracePickler = original
