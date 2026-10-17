"""Diagnostic MessagePack trace output with complete Chrome event values.

The stream contains a versioned header, ordinary Chrome event mappings and a
footer with metadata and event count; the converter emits the existing JSON.
Production defaults and their temporary SQLite archive are unchanged.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from TokenSim.af.trace import pipeline
from TokenSim.af.trace.export import _encode_event, _record_dict

HEADER = {
    "format": "tokensim-chrome-trace-msgpack",
    "version": 1,
    "displayTimeUnit": "us",
}


class _BinaryEncoder:
    def __init__(self):
        import msgpack

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


def convert(source, destination):
    """Convert in bounded memory and publish only a complete, valid stream."""
    import msgpack

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
            unpacker = msgpack.Unpacker(
                stream,
                raw=False,
                strict_map_key=False,
                max_buffer_size=16 * 1024 * 1024,
            )
            if next(unpacker, None) != HEADER:
                raise ValueError("unsupported diagnostic trace header")
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
                raise ValueError("missing or incomplete diagnostic trace footer")
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


@contextmanager
def installed(output):
    """Install only the output writer while retaining the production recorder."""
    original = pipeline._PrefixWriter
    pipeline._PrefixWriter = MessagePackWriter
    try:
        yield
    finally:
        pipeline._PrefixWriter = original


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(f"Converted {convert(arguments.source, arguments.destination)} trace events")
