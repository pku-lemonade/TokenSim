"""Reuse dataclass identities during each synchronous completed-prefix export.

The binary decoder expands references into the original JSON-compatible maps.
The memo resets between completed prefixes and has a fixed record capacity.
This diagnostic assumes that values do not mutate during synchronous export.
"""

from contextlib import contextmanager
import json

from scripts import af_probe_msgpack as base
from TokenSim.af.trace import pipeline
from TokenSim.af.trace.export import _record_dict

HEADER = dict(base.HEADER, extensions="shared-records-v1")


class SharedEncoder:
    def __init__(self):
        import msgpack

        self.native = msgpack
        self.packer = msgpack.Packer(
            default=self.default, use_bin_type=True, use_single_float=False
        )
        self.hits = self.definitions = 0
        self.reset()

    def reset(self):
        self.memo = {}
        self.pending_reset = True

    def default(self, record):
        key = id(record)
        previous = self.memo.get(key)
        if previous is not None:
            self.hits += 1
            return self.native.ExtType(2, previous[1].to_bytes(4, "big"))
        row = _record_dict(record)
        if len(self.memo) >= 8192:
            return row
        index = len(self.memo)
        self.memo[key] = record, index
        self.definitions += 1
        tag = 3 if self.pending_reset else 1
        self.pending_reset = False
        data = self.native.packb((index, row), default=self.default, use_bin_type=True)
        return self.native.ExtType(tag, data)

    def encode(self, value):
        return self.packer.pack(value)


class SharedDecoder:
    def __init__(self):
        self.memo = {}

    def extension(self, tag, data):
        import msgpack

        if tag == 2:
            if len(data) != 4 or int.from_bytes(data, "big") not in self.memo:
                raise ValueError("unknown shared trace reference")
            return self.memo[int.from_bytes(data, "big")]
        if tag not in (1, 3):
            raise ValueError("unsupported shared trace extension")
        if tag == 3:
            self.memo.clear()
        index, row = msgpack.unpackb(
            data, raw=False, strict_map_key=False, ext_hook=self.extension
        )
        if (
            type(index) is not int
            or not 0 <= index < 8192
            or index in self.memo
            or not isinstance(row, dict)
        ):
            raise ValueError("invalid shared trace definition")
        self.memo[index] = row
        return row


class SharedWriter(base.MessagePackWriter):
    def __init__(self, directory, options):
        super().__init__(directory, options)
        self.encoder = SharedEncoder()
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(self.encoder.encode(HEADER))

    def append(self, events):
        self.encoder.reset()
        super().append(events)


def convert(source, destination):
    return base.convert(
        source,
        destination,
        header=HEADER,
        unpack_options={"ext_hook": SharedDecoder().extension},
    )


@contextmanager
def installed(output=None):
    original, encoders = pipeline._PrefixWriter, []

    def writer(*args, **kwargs):
        result = SharedWriter(*args, **kwargs)
        encoders.append(result.encoder)
        return result

    pipeline._PrefixWriter = writer
    try:
        yield
    finally:
        pipeline._PrefixWriter = original
        if output is not None:
            (output / "shared-records.json").write_text(
                json.dumps(
                    [dict(hits=e.hits, definitions=e.definitions) for e in encoders]
                )
                + "\n"
            )
