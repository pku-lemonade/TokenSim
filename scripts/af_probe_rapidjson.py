"""Opt-in RapidJSON exporter with standard encoder fallback semantics."""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from TokenSim.af.runtime.host import host_collection
from TokenSim.af.trace.export import _record_dict


class CompatibleEncoder:
    def __init__(self):
        import rapidjson

        self.native = rapidjson
        self.reference = json.JSONEncoder(
            ensure_ascii=False, separators=(",", ":"), default=_record_dict
        )

    @staticmethod
    def default(value):
        if isinstance(value, tuple):
            return list(value)
        return _record_dict(value)

    def encode(self, item):
        try:
            return self.native.dumps(
                item,
                ensure_ascii=False,
                allow_nan=True,
                default=self.default,
                bytes_mode=self.native.BM_NONE,
                iterable_mode=self.native.IM_ONLY_LISTS,
                mapping_mode=self.native.MM_ONLY_DICTS,
            )
        except (TypeError, ValueError, OverflowError, RecursionError):
            return self.reference.encode(item)


def write_chrome_trace(path, events, metadata=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = CompatibleEncoder()
    temporary = None
    try:
        with (
            host_collection(),
            NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                buffering=1024 * 1024,
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
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


@contextmanager
def installed(output=None):
    from scripts import deepseek_v3_runtime

    original = deepseek_v3_runtime.write_chrome_trace
    deepseek_v3_runtime.write_chrome_trace = write_chrome_trace
    try:
        yield
    finally:
        deepseek_v3_runtime.write_chrome_trace = original
