"""Immutable provider service estimates, independent of SimPy scheduling."""

from collections import OrderedDict
from dataclasses import MISSING, fields, is_dataclass, replace
from functools import lru_cache, wraps
from collections.abc import Mapping
import hashlib


class FrozenDict(dict):
    """JSON-compatible immutable mapping (including immutable nested values)."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("cached AF estimate is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = (
        __ior__
    ) = _immutable

    def __deepcopy__(self, memo):
        return self


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def identity(value):
    # Exact scalar types avoid expensive dataclass/ABC probes and prevent
    # invalid True/1 or 1.0/1 queries from hitting a previously validated key.
    if type(value) in (str, int, float, bool, type(None)):
        return (type(value), value) if type(value) in (bool, float) else value
    if is_dataclass(value):
        return tuple(
            (field.name, identity(getattr(value, field.name)))
            for field in fields(value)
        )
    if isinstance(value, Mapping):
        return tuple(sorted((key, identity(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(identity(item) for item in value)
    return value


def traffic_feature_fingerprint(values):
    """Compact complete routing/layout features without retaining whole graphs.

    FingerprintMemo deliberately retains immutable inputs. Applying it to a
    large per-token graph would keep thousands of old graphs alive. Hash each
    record separately so temporary normalized data is bounded by one record.
    """
    digest = hashlib.sha256()
    for value in values:
        encoded = repr(identity(value)).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class FingerprintMemo:
    """Bounded identity memo for deeply immutable objects, never mutable queries."""

    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.values = {}

    def __call__(self, value):
        if type(value) in (str, int, type(None)):
            return value
        if type(value) in (bool, float):
            return (type(value), value)
        return self._visit(value)[0]

    def _visit(self, value):
        if type(value) in (str, int, float, bool, type(None)):
            return (
                (type(value), value) if type(value) in (bool, float) else value
            ), True
        previous = self.values.get(id(value))
        if previous is not None and previous[0] is value:
            return previous[1], True
        if is_dataclass(value):
            children = tuple(
                (field.name, self._visit(getattr(value, field.name)))
                for field in fields(value)
            )
            key = tuple((name, item[0]) for name, item in children)
            safe = (
                value.__dataclass_params__.frozen
                or getattr(type(value), "_af_readonly", False)
            ) and all(item[1] for _, item in children)
        elif isinstance(value, Mapping):
            children = tuple(
                sorted((name, self._visit(item)) for name, item in value.items())
            )
            key = tuple((name, item[0]) for name, item in children)
            safe = isinstance(value, FrozenDict) and all(
                item[1] for _, item in children
            )
        elif isinstance(value, (tuple, list)):
            children = tuple(self._visit(item) for item in value)
            key = tuple(item[0] for item in children)
            safe = isinstance(value, tuple) and all(item[1] for item in children)
        else:
            return value, False
        if safe:
            if len(self.values) >= self.capacity:
                self.values.pop(next(iter(self.values)))
            self.values[id(value)] = (value, key)
        return key, safe


class QuerySkeleton:
    """Compile field layout/defaults once; execution values stay per-call."""

    def __init__(self, query_type):
        self.query_type = query_type
        self.names = tuple(
            field.name
            for field in fields(query_type)
            if field.name not in _EXECUTION_FIELDS
        )
        self.defaults = {
            field.name: (
                field.default
                if field.default is not MISSING
                else freeze(field.default_factory())
            )
            for field in fields(query_type)
            if field.default is not MISSING or field.default_factory is not MISSING
        }


@lru_cache(maxsize=2)
def query_skeleton(query_type):
    return QuerySkeleton(query_type)


class PreparedQuery:
    __slots__ = ("skeleton", "values")

    def __init__(self, skeleton, values):
        self.skeleton = skeleton
        self.values = values

    def __getattr__(self, name):
        if name in self.values:
            return self.values[name]
        try:
            return self.skeleton.defaults[name]
        except KeyError:
            raise AttributeError(name) from None

    def materialize(self):
        return self.skeleton.query_type(**self.values)


def prepare_query(provider, query_type, **values):
    # Opt in only the two built-ins, preserving subclass/custom provider calls
    # and instrumentation replacing estimate with an ordinary query consumer.
    method = getattr(provider.estimate, "__func__", None)
    if (provider.cache_enabled if hasattr(provider, "cache_enabled") else False) and (
        type(provider).__module__ == "TokenSim.af.operator_provider"
        and method is getattr(method, "_prepared_query", None)
        and getattr(provider, "query_fast_path", True)
    ):
        return PreparedQuery(query_skeleton(query_type), values)
    if hasattr(provider, "query_constructions"):
        provider.query_constructions += 1
    return query_type(**values)


# These identify executions or routing ownership, rather than the provider's
# service calculation. Payload and normalized local/expert work remain in keys.
_EXECUTION_FIELDS = frozenset(
    {
        "iteration_id",
        "block_id",
        "layer_id",
        "node_id",
        "logical_stage_id",
        "leg_id",
        "routing_edges",
        "tensor_slices",
    }
)


def cached_estimate(method):
    @wraps(method)
    def estimate(self, query):
        self.estimate_requests += 1
        validator = getattr(self, "validate_query", None)
        if validator is not None:
            validator(query)
        if not self.cache_enabled:
            result = method(self, query)
            return replace(result, details=freeze(result.details))
        skeleton = (
            query.skeleton
            if isinstance(query, PreparedQuery)
            else query_skeleton(type(query))
        )
        values = query.values if isinstance(query, PreparedQuery) else vars(query)
        normalize = self._fingerprints
        key = (
            self._estimate_namespace,
            skeleton.query_type,
            tuple(
                normalize(values.get(name, skeleton.defaults.get(name)))
                for name in skeleton.names
            ),
        )
        result = self._estimate_cache.get(key)
        if result is not None:
            self.estimate_cache_hits += 1
            self._estimate_cache.move_to_end(key)
            return result
        if isinstance(query, PreparedQuery):
            query = query.materialize()
            self.query_constructions += 1
        result = method(self, query)
        result = replace(result, details=freeze(result.details))
        self._estimate_cache[key] = result
        if len(self._estimate_cache) > self.estimate_cache_capacity:
            self._estimate_cache.popitem(last=False)
        return result

    estimate._prepared_query = estimate
    return estimate


def initialize_cache(provider, enabled=True, capacity=4096):
    provider.cache_enabled = enabled
    provider.estimate_cache_capacity = capacity
    provider._estimate_cache = OrderedDict()
    provider._plan_cache = OrderedDict()
    provider.plan_cache_hits = 0
    provider.estimate_requests = 0
    provider.estimate_cache_hits = 0
    provider.query_constructions = 0
    provider.query_fast_path = True
    provider._fingerprints = FingerprintMemo()
    cluster = getattr(provider, "cluster_config", None)
    provider._estimate_namespace = (
        provider.fallback,
        identity(getattr(provider, "lookup_policy", None)),
        (
            (cluster.af.transfer_topology_mode, cluster.af.communication_timing.enabled)
            if cluster
            else None
        ),
        (
            cluster.af.execution_protocol
            if cluster
            else getattr(provider, "execution_protocol", None)
        ),
        getattr(provider, "require_device_scope", False),
        identity(cluster.af.operation_service_profile) if cluster else None,
        (
            identity(getattr(cluster.af, "communication_domains", None))
            if cluster
            else None
        ),
    )


def cached_plan(provider, key, factory):
    """Full static-plan key supplied by its owner, never _EXECUTION_FIELDS."""
    if not provider.cache_enabled:
        return factory()
    key = (provider._estimate_namespace, provider._fingerprints(key))
    result = provider._plan_cache.get(key)
    if result is not None:
        provider.plan_cache_hits += 1
        provider._plan_cache.move_to_end(key)
        return result
    result = factory()
    provider._plan_cache[key] = result
    if len(provider._plan_cache) > provider.estimate_cache_capacity:
        provider._plan_cache.popitem(last=False)
    return result


def cached_phase_estimate(method):
    """Share the service LRU without counting phases as external query builds."""

    @wraps(method)
    def estimate(self, phase):
        if not self.cache_enabled:
            result = method(self, phase)
            return replace(result, details=freeze(result.details))
        key = ("phase-service", self._estimate_namespace, self._fingerprints(phase))
        result = self._estimate_cache.get(key)
        if result is None:
            result = method(self, phase)
            result = replace(result, details=freeze(result.details))
            self._estimate_cache[key] = result
            if len(self._estimate_cache) > self.estimate_cache_capacity:
                self._estimate_cache.popitem(last=False)
        else:
            self._estimate_cache.move_to_end(key)
        return result

    return estimate
