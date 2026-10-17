"""Diagnostic bounded batches for the production prefix JSON writer."""

from contextlib import contextmanager
from itertools import islice
import json

from TokenSim.af.trace import pipeline
from TokenSim.af.trace.export import _encode_event, _record_dict


class _Capture:
    @staticmethod
    def encode(item):
        return item


def append(self, events):
    options, encoder = self.options, self.encoder
    capture = _Capture()
    includes = options.includes if options.filtered else None
    selected = (event for event in events if includes is None or includes(event))
    while batch := list(islice(selected, 64)):
        rows = [_encode_event(event, options, capture) for event in batch]
        try:
            data = encoder.encode(rows)[1:-1]
        except RecursionError:
            data = ",".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_record_dict,
                )
                for row in rows
            )
        if self.count:
            self.stream.write(b",")
        self.stream.write(data.encode("utf-8"))
        self.count += len(batch)


@contextmanager
def installed(output=None):
    original = pipeline._PrefixWriter.append
    pipeline._PrefixWriter.append = append
    try:
        yield
    finally:
        pipeline._PrefixWriter.append = original
