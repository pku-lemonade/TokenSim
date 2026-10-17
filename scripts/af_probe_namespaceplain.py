"""Diagnostic reference using the original tuple configuration namespace."""

from contextlib import contextmanager

from TokenSim.af.caching import estimates


@contextmanager
def installed(output=None):
    original = estimates.EstimateNamespace
    estimates.EstimateNamespace = tuple
    try:
        yield
    finally:
        estimates.EstimateNamespace = original
