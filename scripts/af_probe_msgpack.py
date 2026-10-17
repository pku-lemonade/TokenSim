"""Diagnostic hook for the production MessagePack implementation."""

from contextlib import contextmanager
from pathlib import Path

from TokenSim.af.trace import pipeline
from TokenSim.af.trace.msgpack import HEADER, MessagePackWriter, _BinaryEncoder, convert


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
