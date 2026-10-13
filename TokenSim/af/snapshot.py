"""Instance-scoped read-only configuration; create a new runtime to reconfigure."""

from dataclasses import fields, is_dataclass
from functools import lru_cache

from TokenSim.af.estimate_cache import FrozenDict


def _reject_write(self, *args, **kwargs):
    raise TypeError("AF runtime configuration is read-only; create a new runtime")


@lru_cache(maxsize=None)
def _readonly_type(cls):
    return type(f"ReadOnly{cls.__name__}", (cls,), {
        "__setattr__": _reject_write, "__delattr__": _reject_write,
        "__deepcopy__": lambda self, memo: self,
        "_af_readonly": True,
    })


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
        result = object.__new__(cls if cls.__dataclass_params__.frozen else _readonly_type(cls))
        memo[id(value)] = result
        items = vars(value).items() if hasattr(value, "__dict__") else (
            (field.name, getattr(value, field.name)) for field in fields(value)
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
            self._packages[key] = snapshot(self._source.operator_package(device_id, backend))
        return self._packages[key]
