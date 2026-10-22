"""Incremental Chrome trace export without copying the full event history."""

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from functools import lru_cache
from inspect import getattr_static
from types import GetSetDescriptorType
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from TokenSim.trace.host import host_collection
from TokenSim.trace.events import TraceEvent
from TokenSim.trace.config import TraceConfig, TIMING_FIELDS
from TokenSim.errors import ConfigurationError


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


def _event_encoder():
    return json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        default=_record_dict,
        check_circular=False,
        allow_nan=False,
    )


def _encode_event(event, options, encoder, *, track=None):
    selected = options.fields
    if selected is None and options.detail == "timing":
        selected = TIMING_FIELDS
    row = (
        {name: getattr(event, name, None) for name in selected}
        if selected is not None
        else _record_dict(event)
    )
    item = dict(
        name=event.stage or event.operation or event.event_type,
        cat=event.event_type,
        ph="X",
        pid=1,
        tid=event.resource_id if track is None else track,
        ts=event.start_at * 1e6,
        dur=event.service_duration * 1e6,
        args=row,
    )
    try:
        return encoder.encode(item)
    except RecursionError:
        # Retry with cycle checks to preserve the standard failure type.
        return json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_record_dict,
            allow_nan=False,
        )


def _export_options(trace_config, metadata):
    if trace_config is not None and not isinstance(trace_config, TraceConfig):
        raise ConfigurationError("trace_config must be an TraceConfig")
    options = trace_config or TraceConfig(mode="full")
    if options.mode != "full":
        raise ConfigurationError("trace export requires full mode")
    selection = options.describe()
    selection.pop("format")
    selection.pop("output")
    defaults = TraceConfig(mode="full").describe()
    defaults.pop("format")
    defaults.pop("output")
    if selection != defaults:
        metadata = dict(metadata or {}, trace_config=selection)
    return options, metadata


def write_chrome_trace(
    path: str | Path,
    events: Iterable[TraceEvent],
    metadata: Mapping[str, Any] | None = None,
    *,
    trace_config: TraceConfig | None = None,
) -> Path:
    """Write complete events in input order, with temporary memory per event.

    Read final event records after execution has patched their dependencies.
    JSON values match TraceEvent.as_dict(); whitespace is compact.
    A failed export leaves any existing destination intact.
    """
    options, metadata = _export_options(trace_config, metadata)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = getattr(events, "_write_prepared_trace", None)
    if prepared is not None and getattr(events, "trace_format", "json") == "json":
        result = prepared(output, metadata, options)
        if result is not None:
            return result
    if options.filtered:
        events = (event for event in events if options.includes(event))
    encoder = _event_encoder()
    modern = metadata is not None and metadata.get("schema_version") == 2
    if modern:
        from TokenSim.trace.flows import FlowIndex
    temporary = None
    try:
        with (
            host_collection(),
            FlowIndex() if modern else nullcontext(None) as flows,
            NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                buffering=1024 * 1024,
                dir=output.parent,
                delete=False,
                prefix=f".{output.name}.",
                suffix=".tmp",
            ) as stream,
        ):
            temporary = Path(stream.name)
            stream.write('{"displayTimeUnit":"us","traceEvents":[')
            count = 0

            def append(encoded):
                nonlocal count
                if count:
                    stream.write(",")
                stream.write(encoded)
                count += 1

            for event in events:
                track = None
                if flows is not None:
                    track, item = flows.add(event)
                    if item is not None:
                        append(encoder.encode(item))
                append(_encode_event(event, options, encoder, track=track))
            if flows is not None:
                for item in flows.flows():
                    append(encoder.encode(item))
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
