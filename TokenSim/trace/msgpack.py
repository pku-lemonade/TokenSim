"""Versioned MessagePack streams preserving complete Chrome event values.

The stream contains a versioned header, ordinary Chrome event mappings and a
footer with metadata and event count; the converter emits the existing JSON.
Readback and prefix export retain the private indexed archive.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from TokenSim.trace import pipeline
from TokenSim.trace.export import _encode_event, _record_dict
from TokenSim.errors import ConfigurationError

HEADER = {
    "format": "tokensim-chrome-trace-msgpack",
    "version": 1,
    "displayTimeUnit": "us",
}


def require_msgpack():
    try:
        import msgpack
    except ImportError as exc:
        raise ConfigurationError(
            "MessagePack traces require msgpack; install tokensim[trace]"
        ) from exc
    return msgpack


class _BinaryEncoder:
    def __init__(self):
        msgpack = require_msgpack()
        self.packer = msgpack.Packer(
            default=_record_dict, use_bin_type=True, use_single_float=False
        )

    def encode(self, value):
        return self.packer.pack(value)


class MessagePackWriter(pipeline._PrefixWriter):
    def __init__(self, directory, options):
        super().__init__(directory, options)
        self.encoder = _BinaryEncoder()
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(self.encoder.encode(HEADER))

    def append(self, events):
        buffer = bytearray()
        options, encoder = self.options, self.encoder
        includes = options.includes if options.filtered else None
        for event in events:
            if includes is not None and not includes(event):
                continue
            buffer.extend(_encode_event(event, options, encoder))
            self.count += 1
            if len(buffer) >= 1024 * 1024:
                self.stream.write(buffer)
                buffer.clear()
        if buffer:
            self.stream.write(buffer)

    def finish(self, output, metadata):
        output = output.with_suffix(".msgpack")
        try:
            self.stream.write(
                self.encoder.encode(
                    dict(
                        metadata=dict(metadata) if metadata is not None else {},
                        event_count=self.count,
                    )
                )
            )
            self.stream.close()
            self.path.replace(output)
        finally:
            self.close()
        return output


def convert(
    source, destination, *, header=HEADER, unpack_options=None, unpacker_type=None
):
    """Convert in bounded memory and publish only a complete, valid stream."""
    if unpacker_type is None:
        unpacker_type = require_msgpack().Unpacker

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with (
            source.open("rb") as stream,
            NamedTemporaryFile(
                mode="wb",
                buffering=1024 * 1024,
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output,
        ):
            temporary = Path(output.name)
            unpacker = unpacker_type(
                stream,
                raw=False,
                strict_map_key=False,
                max_buffer_size=16 * 1024 * 1024,
                **(unpack_options or {}),
            )
            if next(unpacker, None) != header:
                raise ValueError("unsupported trace header")
            complete_bytes = unpacker.tell()
            output.write(b'{"displayTimeUnit":"us","traceEvents":[')
            count, footer = 0, None
            encoder = pipeline._event_encoder()
            for record in unpacker:
                complete_bytes = unpacker.tell()
                if footer is not None or not isinstance(record, dict):
                    raise ValueError("unexpected trace record")
                if record.get("ph") == "X":
                    if count:
                        output.write(b",")
                    output.write(encoder.encode(record).encode("utf-8"))
                    count += 1
                else:
                    footer = record
            if (
                footer is None
                or footer.get("event_count") != count
                or not isinstance(footer.get("metadata"), dict)
                or complete_bytes != source.stat().st_size
            ):
                raise ValueError("missing or incomplete trace footer")
            output.write(b'],"metadata":')
            output.write(
                json.dumps(
                    footer["metadata"], ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )
            output.write(b"}")
        temporary.replace(destination)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MessagePackTraceRecorder(pipeline.PipelinedTraceRecorder):
    trace_format = "msgpack"

    def __init__(self, directory, **options):
        super().__init__(directory, writer_factory=MessagePackWriter, **options)
