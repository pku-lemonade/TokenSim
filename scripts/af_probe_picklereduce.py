"""Diagnostic immutable dictionary reduction without a global object callback."""

from contextlib import contextmanager
import pickle

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.trace import archive_store


def _reduce_frozen(self, protocol):
    return FrozenDict, (dict(self),)


@contextmanager
def installed(output=None):
    original = archive_store._TracePickler
    FrozenDict.__reduce_ex__ = _reduce_frozen
    archive_store._TracePickler = pickle.Pickler
    try:
        yield
    finally:
        del FrozenDict.__reduce_ex__
        archive_store._TracePickler = original
