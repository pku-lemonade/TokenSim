"""Opt-in identity front cache for already-proven immutable query fields."""

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps

from TokenSim.af.caching import estimates
from TokenSim.af.providers.compute import ComputeEstimateMixin
from TokenSim.af.providers.transfer import TransferEstimateMixin


_SCALARS = (str, int, type(None))
_TAGGED = (bool, float)
_IDENTITY = object()
_TUPLE = object()


class HashedKey:
    """Retain full structural equality while reusing the hash on front hits."""

    __slots__ = ("values", "digest")

    def __init__(self, values):
        self.values = values
        self.digest = hash(values)

    def __hash__(self):
        return self.digest

    def __eq__(self, other):
        return self.values == (other.values if isinstance(other, HashedKey) else other)


def front_key(values, memo):
    components, references = [], []
    for value in values:
        kind = type(value)
        if kind in _SCALARS:
            components.append(value)
        elif kind in _TAGGED:
            components.append((kind, value))
        elif kind is tuple and all(type(child) in _SCALARS for child in value):
            components.append((_TUPLE, value))
        else:
            previous = memo.values.get(id(value))
            if previous is None or previous[0] is not value:
                return None
            components.append((_IDENTITY, id(value)))
            references.append(value)
    return tuple(components), tuple(references)


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
        raw = query.values if prepared else vars(query)
        values = skeleton.key_values(raw)
        signature = (
            self.estimate_signature(query)
            if hasattr(self, "estimate_signature")
            else ()
        )
        prefix = self._estimate_namespace, skeleton.query_type, signature
        front = vars(self).setdefault("_probe_query_front", {})
        quick = front_key(values, self._fingerprints)
        previous = None if quick is None else front.get((prefix, quick[0]))
        if previous is None:
            key = HashedKey((*prefix, tuple(map(self._fingerprints, values))))
        else:
            key = previous[0]
        result = self._estimate_cache.get(key)
        if result is not None:
            self.estimate_cache_hits += 1
            self._estimate_cache.move_to_end(key)
        else:
            if prepared:
                query = query.materialize()
                self.query_constructions += 1
            result = method(self, query)
            result = replace(result, details=estimates.freeze(result.details))
            self._estimate_cache[key] = result
            if len(self._estimate_cache) > self.estimate_cache_capacity:
                self._estimate_cache.popitem(last=False)
        if previous is None:
            quick = front_key(values, self._fingerprints)
            if quick is not None:
                if len(front) >= 1024:
                    front.pop(next(iter(front)))
                front[(prefix, quick[0])] = key, quick[1]
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
