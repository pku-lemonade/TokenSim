"""Immutable runtime snapshots, prepared queries and bounded service caches."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass, replace
from functools import lru_cache, wraps
import hashlib
from operator import itemgetter

from TokenSim.af.caching.namespace import EstimateNamespace

_SCALARS = (str, int, type(None))
_TAGGED_SCALARS = (bool, float)


@lru_cache(maxsize=128)
def _field_names(record_type):
    return tuple(field.name for field in fields(record_type))


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
        if type(value) in _SCALARS:
            return value
        if type(value) in _TAGGED_SCALARS:
            return (type(value), value)
        return self._visit(value)[0]

    def _visit(self, value):
        value_type = type(value)
        if value_type in _SCALARS:
            return value, True
        if value_type in _TAGGED_SCALARS:
            return (value_type, value), True
        previous = self.values.get(id(value))
        if previous is not None and previous[0] is value:
            return previous[1], True
        if is_dataclass(value):
            safe = value.__dataclass_params__.frozen or getattr(
                value_type, "_af_readonly", False
            )
            children = []
            for name in _field_names(value_type):
                child, immutable = self._visit(getattr(value, name))
                children.append((name, child))
                safe &= immutable
            key = tuple(children)
        elif isinstance(value, Mapping):
            safe = isinstance(value, FrozenDict)
            children = []
            for name, item in value.items():
                child, immutable = self._visit(item)
                children.append((name, child))
                safe &= immutable
            key = tuple(sorted(children))
        elif isinstance(value, (tuple, list)):
            if value_type is tuple and all(type(item) in _SCALARS for item in value):
                return value, True
            safe = isinstance(value, tuple)
            children = []
            for item in value:
                child, immutable = self._visit(item)
                children.append(child)
                safe &= immutable
            key = tuple(children)
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
        self._extract = itemgetter(*self.names) if len(self.names) > 1 else None
        self._key_defaults = dict.fromkeys(self.names) | self.defaults

    def key_values(self, values):
        # Dictionary merge and itemgetter run their field loops in C. Keep
        # PreparedQuery.values unchanged so materialization retains defaults.
        complete = self._key_defaults | values
        if self._extract is not None:
            return self._extract(complete)
        return tuple(complete[name] for name in self.names)


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
    from TokenSim.af.providers.operator_table import (
        OperatorTableAFComputeProvider,
        TopologyAFTransferProvider,
    )

    method = getattr(provider.estimate, "__func__", None)
    if (provider.cache_enabled if hasattr(provider, "cache_enabled") else False) and (
        type(provider) in (OperatorTableAFComputeProvider, TopologyAFTransferProvider)
        and method is getattr(method, "_prepared_query", None)
        and getattr(provider, "query_fast_path", True)
    ):
        return PreparedQuery(query_skeleton(query_type), values)
    if hasattr(provider, "query_constructions"):
        provider.query_constructions += 1
    return query_type(**values)


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
            (
                self.estimate_signature(query)
                if hasattr(self, "estimate_signature")
                else ()
            ),
            tuple(map(normalize, skeleton.key_values(values))),
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
    provider._estimate_namespace = EstimateNamespace(
        (
            provider.fallback,
            identity(getattr(provider, "lookup_policy", None)),
            (
                (
                    cluster.af.transfer_topology_mode,
                    cluster.af.communication_timing.enabled,
                )
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


def _reject_write(self, *args, **kwargs):
    raise TypeError("AF runtime configuration is read-only; create a new runtime")


@lru_cache(maxsize=None)
def _readonly_type(cls):
    return type(
        f"ReadOnly{cls.__name__}",
        (cls,),
        {
            "__setattr__": _reject_write,
            "__delattr__": _reject_write,
            "__deepcopy__": lambda self, memo: self,
            "_af_readonly": True,
        },
    )


def snapshot(value, _memo=None):
    """Detach nested containers too, preserving dataclass methods/properties."""
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if getattr(type(value), "_af_readonly", False):
        return value
    memo = {} if _memo is None else _memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = FrozenDict({key: snapshot(item, memo) for key, item in value.items()})
    elif isinstance(value, (tuple, list)):
        result = tuple(snapshot(item, memo) for item in value)
    elif isinstance(value, (set, frozenset)):
        result = frozenset(snapshot(item, memo) for item in value)
    elif is_dataclass(value):
        cls = type(value)
        # Frozen records keep their type so equality/hash remain compatible.
        result = object.__new__(
            cls if cls.__dataclass_params__.frozen else _readonly_type(cls)
        )
        memo[id(value)] = result
        items = (
            vars(value).items()
            if hasattr(value, "__dict__")
            else ((field.name, getattr(value, field.name)) for field in fields(value))
        )
        for name, item in items:
            object.__setattr__(result, name, snapshot(item, memo))
    else:
        return value
    memo[id(value)] = result
    return result


class ConfigurationOwner:
    """Prevent replacing pinned configuration while allowing runtime counters."""

    _configuration_fields = ()

    def __setattr__(self, name, value):
        if name in self._configuration_fields and name in self.__dict__:
            _reject_write(self)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._configuration_fields:
            _reject_write(self)
        object.__delattr__(self, name)


class HardwareSnapshot(ConfigurationOwner):
    """Pin only used devices/packages; never scan the operator-data matrix."""

    _configuration_fields = ("links",)

    def __init__(self, source):
        from TokenSim.hardware.links import LinkCatalog

        self._source = source
        self._devices = {}
        self._packages = {}
        self._topologies = {}
        self.links = LinkCatalog(snapshot(link) for link in source.links)
        self.links._links = FrozenDict(self.links._links)

    def device(self, name):
        if name not in self._devices:
            self._devices[name] = snapshot(self._source.device(name))
        return self._devices[name]

    def link(self, name):
        return self.links.get(name)

    def topology(self, name):
        if name not in self._topologies:
            self._topologies[name] = snapshot(self._source.topology(name))
        return self._topologies[name]

    def operator_package(self, device_id, backend=None):
        key = (device_id, backend)
        if key not in self._packages:
            self._packages[key] = snapshot(
                self._source.operator_package(device_id, backend)
            )
        return self._packages[key]
