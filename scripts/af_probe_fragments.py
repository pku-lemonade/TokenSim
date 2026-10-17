"""Opt-in immutable JSON-fragment reuse behind the reference trace exporter."""

from contextlib import contextmanager
from dataclasses import is_dataclass
import json
import re
import sys
from types import SimpleNamespace

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.trace import export


_MARKER = re.compile(r'"~tokensim-json:[0-9]+~"')
_SCALARS = (str, int, float, bool, type(None))


class FragmentEncoder:
    def __init__(self, **options):
        self.reference = json.JSONEncoder(**options)
        self.fast = json.JSONEncoder(**(options | {"default": self.default}))
        self.memo = {}
        self.used = {}
        self.capacity = 4096

    def immutable(self, value):
        kind = type(value)
        if kind in _SCALARS:
            return True
        if kind is tuple:
            return all(self.immutable(child) for child in value)
        if kind is FrozenDict:
            return all(self.immutable(child) for child in value.values())
        if is_dataclass(value) and value.__dataclass_params__.frozen:
            layout = export._field_layout(kind)
            if layout is not None and layout[1]:
                return all(
                    self.immutable(child)
                    for child in export._record_dict(value).values()
                )
        return False

    def default(self, record):
        key = id(record)
        cached = self.memo.get(key)
        if cached is None or cached[0] is not record:
            if not self.immutable(record):
                return export._record_dict(record)
            payload = self.reference.encode(record)
            marker = f'"~tokensim-json:{key}~"'
            cached = record, marker, payload
            if len(self.memo) >= self.capacity:
                self.memo.pop(next(iter(self.memo)))
            self.memo[key] = cached
        _, marker, payload = cached
        entry = self.used.get(marker)
        if entry is None:
            self.used[marker] = [payload, 1, 0]
        else:
            entry[1] += 1
        return marker[1:-1]

    def encode(self, item):
        try:
            encoded = self.fast.encode(item)
            if not self.used:
                return encoded

            def substitute(match):
                entry = self.used.get(match[0])
                if entry is None:
                    return match[0]
                entry[2] += 1
                return entry[0]

            # A single substitution pass never scans fragments it inserts.
            # A literal marker collision adds a match and selects reference JSON.
            result = _MARKER.sub(substitute, encoded)
            if any(expected != seen for _, expected, seen in self.used.values()):
                return self.reference.encode(item)
            return result
        finally:
            self.used.clear()


@contextmanager
def installed(output=None):
    original_json = export.json
    export.json = SimpleNamespace(JSONEncoder=FragmentEncoder, dumps=json.dumps)
    pipeline = sys.modules.get("scripts.af_probe_pipeline")
    original_init = None if pipeline is None else pipeline.PrefixWriter.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.encoder = FragmentEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
            default=export._record_dict,
            check_circular=False,
        )

    if pipeline is not None:
        pipeline.PrefixWriter.__init__ = initialize
    try:
        yield
    finally:
        export.json = original_json
        if pipeline is not None:
            pipeline.PrefixWriter.__init__ = original_init
