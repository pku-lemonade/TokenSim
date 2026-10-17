"""Opt-in sparse provider-key normalization with unchanged cache semantics."""

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps

from TokenSim.af.caching import estimates
from TokenSim.af.providers.compute import ComputeEstimateMixin
from TokenSim.af.providers.transfer import TransferEstimateMixin


class SparseKeyPlan:
    def __init__(self, skeleton):
        self.positions = {name: i for i, name in enumerate(skeleton.names)}
        self.seeds = []
        self.mutable_defaults = []
        memo = estimates.FingerprintMemo()
        for index, name in enumerate(skeleton.names):
            value = skeleton._key_defaults[name]
            normalized, immutable = memo._visit(value)
            self.seeds.append(normalized)
            if not immutable:
                self.mutable_defaults.append((index, value))

    def key(self, values, normalize):
        result = self.seeds.copy()
        for index, value in self.mutable_defaults:
            result[index] = normalize(value)
        positions = self.positions
        for name, value in values.items():
            index = positions.get(name)
            if index is not None:
                result[index] = normalize(value)
        return tuple(result)


def decorate(method):
    @wraps(method)
    def estimate(self, query):
        self.estimate_requests += 1
        validator = getattr(self, "validate_query", None)
        if validator is not None:
            validator(query)
        if not self.cache_enabled:
            result = method(self, query)
            return replace(result, details=estimates.freeze(result.details))
        prepared = isinstance(query, estimates.PreparedQuery)
        skeleton = query.skeleton if prepared else estimates.query_skeleton(type(query))
        values = query.values if prepared else vars(query)
        plan = vars(skeleton).get("_probe_sparse_key_plan")
        if plan is None:
            plan = SparseKeyPlan(skeleton)
            skeleton._probe_sparse_key_plan = plan
        key = (
            self._estimate_namespace,
            skeleton.query_type,
            (
                self.estimate_signature(query)
                if hasattr(self, "estimate_signature")
                else ()
            ),
            plan.key(values, self._fingerprints),
        )
        result = self._estimate_cache.get(key)
        if result is not None:
            self.estimate_cache_hits += 1
            self._estimate_cache.move_to_end(key)
            return result
        if prepared:
            query = query.materialize()
            self.query_constructions += 1
        result = method(self, query)
        result = replace(result, details=estimates.freeze(result.details))
        self._estimate_cache[key] = result
        if len(self._estimate_cache) > self.estimate_cache_capacity:
            self._estimate_cache.popitem(last=False)
        return result

    estimate._prepared_query = estimate
    return estimate


@contextmanager
def installed(output=None):
    owners = (ComputeEstimateMixin, TransferEstimateMixin)
    originals = tuple(owner.estimate for owner in owners)
    for owner, method in zip(owners, originals):
        owner.estimate = decorate(method.__wrapped__)
    try:
        yield
    finally:
        for owner, method in zip(owners, originals):
            owner.estimate = method
