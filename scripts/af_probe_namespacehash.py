"""Diagnostic hash reuse for complete provider configuration namespaces."""

from contextlib import contextmanager

from TokenSim.af.caching import estimates
from TokenSim.af.caching.namespace import EstimateNamespace
from TokenSim.af.providers import operator_table, traffic


@contextmanager
def installed(output=None):
    owners = (estimates, operator_table, traffic)
    originals = tuple(owner.initialize_cache for owner in owners)

    def initialize(provider, *args, **kwargs):
        originals[0](provider, *args, **kwargs)
        if isinstance(provider._estimate_namespace, tuple):
            provider._estimate_namespace = EstimateNamespace(
                provider._estimate_namespace
            )

    for owner in owners:
        owner.initialize_cache = initialize
    try:
        yield
    finally:
        for owner, original in zip(owners, originals):
            owner.initialize_cache = original
