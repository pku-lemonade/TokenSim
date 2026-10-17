"""Diagnostic structural normalization with scalar children handled in-place."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import is_dataclass

from TokenSim.af.caching import estimates


class InlineFingerprintMemo(estimates.FingerprintMemo):
    def _visit(self, value):
        kind = type(value)
        if kind in estimates._SCALARS:
            return value, True
        if kind in estimates._TAGGED_SCALARS:
            return (kind, value), True
        previous = self.values.get(id(value))
        if previous is not None and previous[0] is value:
            return previous[1], True
        if is_dataclass(value):
            safe = value.__dataclass_params__.frozen or getattr(
                kind, "_af_readonly", False
            )
            children = []
            for name in estimates._field_names(kind):
                item = getattr(value, name)
                item_type = type(item)
                if item_type in estimates._SCALARS:
                    child = item
                elif item_type in estimates._TAGGED_SCALARS:
                    child = (item_type, item)
                else:
                    child, immutable = self._visit(item)
                    safe &= immutable
                children.append((name, child))
            key = tuple(children)
        elif isinstance(value, Mapping):
            safe = isinstance(value, estimates.FrozenDict)
            children = []
            for name, item in value.items():
                item_type = type(item)
                if item_type in estimates._SCALARS:
                    child = item
                elif item_type in estimates._TAGGED_SCALARS:
                    child = (item_type, item)
                else:
                    child, immutable = self._visit(item)
                    safe &= immutable
                children.append((name, child))
            key = tuple(sorted(children))
        elif isinstance(value, (tuple, list)):
            safe = isinstance(value, tuple)
            children = []
            all_scalars = kind is tuple
            for item in value:
                item_type = type(item)
                if item_type in estimates._SCALARS:
                    child = item
                elif item_type in estimates._TAGGED_SCALARS:
                    child = (item_type, item)
                    all_scalars = False
                else:
                    all_scalars = False
                    child, immutable = self._visit(item)
                    safe &= immutable
                children.append(child)
            if all_scalars:
                return value, True
            key = tuple(children)
        else:
            return value, False
        if safe:
            if len(self.values) >= self.capacity:
                self.values.pop(next(iter(self.values)))
            self.values[id(value)] = (value, key)
        return key, safe


@contextmanager
def installed(output=None):
    original = estimates.FingerprintMemo
    estimates.FingerprintMemo = InlineFingerprintMemo
    try:
        yield
    finally:
        estimates.FingerprintMemo = original
