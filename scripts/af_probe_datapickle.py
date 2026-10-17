"""Chunked data-only pickle with native memoization and fixed reader globals.

Dataclasses become ordinary dictionaries while the native pickler traverses
each chunk; encoded files do not depend on TokenSim class definitions.
This probe retains the production archive and replaces only prepared output.
"""

from contextlib import contextmanager
from dataclasses import is_dataclass
from io import BytesIO
import json
import pickle
import struct

from scripts import af_probe_msgpack as base
from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.trace import pipeline
from TokenSim.af.trace.export import _encode_event, _record_dict

MAGIC = b"TokenSimDataTrace\x00\x01\n"
FRAME = struct.Struct(">BQ")
MAX_FRAME = 16 * 1024 * 1024


class _DataPickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, FrozenDict):
            return dict, (dict(value),)
        if is_dataclass(value) and not isinstance(value, type):
            return dict, (_record_dict(value),)
        if value is dict:
            return NotImplemented
        raise TypeError(f"unsupported trace value: {type(value).__name__}")


class _DataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("builtins", "dict"):
            return dict
        raise pickle.UnpicklingError(f"unsupported trace constructor: {module}.{name}")


class _MappingEncoder:
    def encode(self, item):
        return item


class DataWriter(pipeline._PrefixWriter):
    def __init__(self, directory, options):
        super().__init__(directory, options)
        self.encoder = _MappingEncoder()
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(MAGIC)

    def _chunk(self, rows):
        stream = BytesIO()
        _DataPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(tuple(rows))
        payload = stream.getvalue()
        if len(payload) > MAX_FRAME:
            raise ValueError("diagnostic trace chunk exceeds its frame limit")
        self.stream.write(FRAME.pack(1, len(payload)))
        self.stream.write(payload)

    def append(self, events):
        rows = []
        options = self.options
        includes = options.includes if options.filtered else None
        for event in events:
            if includes is not None and not includes(event):
                continue
            rows.append(_encode_event(event, options, self.encoder))
            self.count += 1
            if len(rows) == 1024:
                self._chunk(rows)
                rows.clear()
        if rows:
            self._chunk(rows)

    def finish(self, output, metadata):
        output = output.with_suffix(".pickle")
        try:
            footer = json.dumps(
                dict(metadata=dict(metadata or {}), event_count=self.count),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.stream.write(FRAME.pack(2, len(footer)))
            self.stream.write(footer)
            self.stream.close()
            self.path.replace(output)
        finally:
            self.close()
        return output


def _records(stream):
    if stream.read(len(MAGIC)) != MAGIC:
        raise ValueError("unsupported data-only trace header")
    while True:
        frame = stream.read(FRAME.size)
        if not frame:
            return
        if len(frame) != FRAME.size:
            raise ValueError("incomplete trace frame header")
        kind, size = FRAME.unpack(frame)
        if size > MAX_FRAME:
            raise ValueError("trace frame exceeds its buffer limit")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("incomplete trace frame payload")
        if kind == 1:
            payload = BytesIO(data)
            rows = _DataReader(payload).load()
            if not isinstance(rows, tuple) or payload.tell() != len(data):
                raise ValueError("invalid data-only trace chunk")
            yield from rows
        elif kind == 2:
            yield json.loads(data)
        else:
            raise ValueError("unsupported trace frame type")


def convert(source, destination):
    """Use the existing atomic JSON publisher with a data-only chunk reader."""

    # The base converter owns framing, complete record validation and publication.
    class Unpacker:
        def __init__(self, stream, **options):
            self.stream = stream
            self.records = iter((base.HEADER,))
            self.body = _records(stream)

        def __next__(self):
            return next(self.records, None) or next(self.body)

        def __iter__(self):
            return self

        def tell(self):
            return self.stream.tell()

    return base.convert(source, destination, unpacker_type=Unpacker)


@contextmanager
def installed(output=None):
    original = pipeline._PrefixWriter
    pipeline._PrefixWriter = DataWriter
    try:
        yield
    finally:
        pipeline._PrefixWriter = original
