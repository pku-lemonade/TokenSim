"""Legacy diagnostic hook for the production indexed archive implementation."""

from contextlib import contextmanager
from pathlib import Path

from TokenSim.af import factory
from TokenSim.af.trace.binary_archive import ArchiveTraceRecorder, FORMAT, convert


class ArchiveOnlyRecorder(ArchiveTraceRecorder):
    # The legacy probe intentionally intercepts the ordinary JSON export hook.
    trace_format = "json"

    def _write_prepared_trace(self, output, metadata, options):
        # Preserve the exploratory probe's partial-JSON fallback for its tests.
        if self.events.retained_count:
            return None
        return super()._write_prepared_trace(output, metadata, options)


@contextmanager
def installed(output=None):
    original = factory.PipelinedTraceRecorder
    factory.PipelinedTraceRecorder = ArchiveOnlyRecorder
    try:
        yield
    finally:
        factory.PipelinedTraceRecorder = original


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(
        f"Converted {convert(arguments.source, arguments.destination)} archived events"
    )
