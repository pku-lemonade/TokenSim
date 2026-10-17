"""Opt-in native JSON encoding with bounded immutable eligibility caching."""

from contextlib import contextmanager
from dataclasses import is_dataclass
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.runtime.host import host_collection
from TokenSim.af.trace.export import _record_dict


class CompatibleEncoder:
    def __init__(self):
        import orjson

        self.native = orjson
        self.memo = {}
        self.reference = json.JSONEncoder(
            ensure_ascii=False, separators=(",", ":"), default=_record_dict
        )

    def eligible(self, value, active):
        kind = type(value)
        if kind in (str, int, bool, type(None)):
            return True, True
        if kind is float:
            return math.isfinite(value), True
        key = id(value)
        previous = self.memo.get(key)
        if previous is not None and previous[0] is value:
            return previous[1], True
        if key in active:
            return False, False
        active.add(key)
        try:
            if is_dataclass(value):
                immutable = value.__dataclass_params__.frozen
                children = _record_dict(value).values()
            elif kind in (dict, FrozenDict):
                if any(type(name) is not str for name in value):
                    return False, False
                immutable = kind is FrozenDict
                children = value.values()
            elif kind in (list, tuple):
                immutable = kind is tuple
                children = value
            else:
                return False, False
            for child in children:
                allowed, safe = self.eligible(child, active)
                if not allowed:
                    return False, False
                immutable &= safe
            if immutable:
                if len(self.memo) >= 4096:
                    self.memo.pop(next(iter(self.memo)))
                self.memo[key] = value, True
            return True, immutable
        finally:
            active.remove(key)

    def encode(self, item):
        if self.eligible(item, set())[0]:
            try:
                return self.native.dumps(
                    item,
                    default=_record_dict,
                    option=self.native.OPT_PASSTHROUGH_DATACLASS,
                )
            except TypeError:
                pass
        return self.reference.encode(item).encode("utf-8")


def write_chrome_trace(path, events, metadata=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = CompatibleEncoder()
    temporary = None
    try:
        with (
            host_collection(),
            NamedTemporaryFile(
                mode="wb",
                buffering=1024 * 1024,
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream,
        ):
            temporary = Path(stream.name)
            stream.write(b'{"displayTimeUnit":"us","traceEvents":[')
            for index, event in enumerate(events):
                row = _record_dict(event)
                item = dict(
                    name=row["stage"] or row["operation"] or row["event_type"],
                    cat=row["event_type"],
                    ph="X",
                    pid=1,
                    tid=row["resource_id"],
                    ts=row["start_at"] * 1e6,
                    dur=row["service_duration"] * 1e6,
                    args=row,
                )
                if index:
                    stream.write(b",")
                stream.write(encoder.encode(item))
            stream.write(b'],"metadata":')
            stream.write(
                json.dumps(
                    dict(metadata) if metadata is not None else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            stream.write(b"}")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


@contextmanager
def installed(output=None):
    from scripts import deepseek_v3_runtime

    original = deepseek_v3_runtime.write_chrome_trace
    deepseek_v3_runtime.write_chrome_trace = write_chrome_trace
    try:
        yield
    finally:
        deepseek_v3_runtime.write_chrome_trace = original
