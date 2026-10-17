"""Hash parsed Chrome trace values in bounded memory, preserving array order."""

import hashlib
import json


def trace_digest(path):
    decoder = json.JSONDecoder()
    digest = hashlib.sha256()
    count = 0
    prefix = '{"displayTimeUnit":"us","traceEvents":['
    with path.open(encoding="utf-8") as stream:
        if stream.read(len(prefix)) != prefix:
            raise ValueError("unexpected Chrome trace framing")
        buffer, at = "", 0
        while True:
            if at >= len(buffer):
                buffer, at = stream.read(1024 * 1024), 0
                if not buffer:
                    raise ValueError("incomplete trace")
            if buffer[at] == ",":
                at += 1
                continue
            if buffer[at] == "]":
                tail = buffer[at + 1 :] + stream.read()
                value = json.loads('{"displayTimeUnit":"us"' + tail)
                digest.update(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                )
                return dict(events=count, sha256=digest.hexdigest())
            try:
                value, end = decoder.raw_decode(buffer, at)
            except json.JSONDecodeError:
                more = stream.read(1024 * 1024)
                if not more:
                    raise
                buffer, at = buffer[at:] + more, 0
                continue
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            )
            digest.update(b"\n")
            count += 1
            at = end
