"""Incremental Chrome trace export without copying the full event history."""

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
from inspect import getattr_static
from types import GetSetDescriptorType
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from TokenSim.af.runtime.host import host_collection
from TokenSim.af.trace.metrics import AFTraceEvent


@lru_cache(maxsize=128)
def _field_layout(record_type):
    if not is_dataclass(record_type):
        return None
    names = tuple(field.name for field in fields(record_type))
    direct = (
        record_type.__getattribute__ is object.__getattribute__
        and isinstance(
            getattr_static(record_type, "__dict__", None), GetSetDescriptorType
        )
        and all(
            not hasattr(getattr_static(record_type, name, None), "__get__")
            for name in names
        )
    )
    return names, direct


def _record_dict(record):
    # Nested immutable records stay shared until the JSON encoder visits them.
    layout = _field_layout(type(record))
    if layout is None:
        raise TypeError(
            f"Object of type {type(record).__name__} is not JSON serializable"
        )
    names, direct = layout
    values = getattr(record, "__dict__", None) if direct else None
    # Reuse the existing field mapping only when its contents and order match
    # the declared schema. Cached properties and arbitrary extra attributes
    # must never become exported fields; slots and class defaults still work.
    if type(values) is dict and tuple(values) == names:
        return values
    return {name: getattr(record, name) for name in names}


def write_chrome_trace(
    path: str | Path,
    events: Iterable[AFTraceEvent],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write complete events in input order, with temporary memory per event.

    Read final event records after execution has patched their dependencies.
    JSON values match AFTraceEvent.as_dict(); whitespace is compact.
    A failed export leaves any existing destination intact.
    """
    encoder = json.JSONEncoder(
        ensure_ascii=False, separators=(",", ":"), default=_record_dict
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with (
            host_collection(),
            NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                delete=False,
                prefix=f".{output.name}.",
                suffix=".tmp",
            ) as stream,
        ):
            temporary = Path(stream.name)
            stream.write('{"displayTimeUnit":"us","traceEvents":[')
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
                    stream.write(",")
                stream.write(encoder.encode(item))
            stream.write('],"metadata":')
            stream.write(
                json.dumps(
                    dict(metadata) if metadata is not None else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            stream.write("}")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
