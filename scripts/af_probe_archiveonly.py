"""Publish an indexed binary archive snapshot without a second event encoder.

This diagnostic requires the same TokenSim record definitions for readback.
Only a fixed set of project record constructors is accepted by the converter.
Production defaults and the archive's existing mutable readback are unchanged.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import pickle
import sqlite3
from tempfile import NamedTemporaryFile

from TokenSim.af import factory
from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.providers import queries
from TokenSim.af.trace.archive import ArchivedTraceRecorder
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.af.trace.pipeline import PipelinedTraceRecorder

FORMAT = "tokensim-indexed-trace-probe-v1"
_RECORDS = (AFTraceEvent, FrozenDict) + tuple(
    getattr(queries, name)
    for name in (
        "AFEndpoint",
        "AFTensorSlice",
        "AFRoutingEdge",
        "AFComputeQuery",
        "AFTransferQuery",
        "AFServiceEstimate",
    )
)
_CONSTRUCTORS = {(kind.__module__, kind.__name__): kind for kind in _RECORDS}


class _RecordReader(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _CONSTRUCTORS[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"unsupported trace record: {module}.{name}")


class ArchiveOnlyRecorder(PipelinedTraceRecorder):
    def complete_iteration(self, iteration_id):
        ArchivedTraceRecorder.complete_iteration(self, iteration_id)

    def _write_prepared_trace(self, output, metadata, options):
        if self.events.retained_count:
            return None
        output = output.with_suffix(".sqlite3")
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=output.parent, suffix=".tmp", delete=False
            ) as f:
                temporary = Path(f.name)
            connection = sqlite3.connect(temporary)
            try:
                self.archive._open().backup(connection)
                with connection:
                    connection.execute(
                        "CREATE TABLE trace_export "
                        "(format TEXT, metadata TEXT, options TEXT, event_count INTEGER)"
                    )
                    connection.execute(
                        "INSERT INTO trace_export VALUES (?, ?, ?, ?)",
                        (
                            FORMAT,
                            json.dumps(dict(metadata or {}), ensure_ascii=False),
                            json.dumps(asdict(options)),
                            len(self.events),
                        ),
                    )
            finally:
                connection.close()
            temporary.replace(output)
            return output
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _events(connection, expected):
    cache = OrderedDict()
    count = 0
    cursor = connection.execute(
        "SELECT position, chunk_id, ordinal FROM positions ORDER BY position"
    )
    try:
        for position, chunk, ordinal in cursor:
            if position != count:
                raise ValueError("trace positions must be contiguous")
            if chunk not in cache:
                payload = connection.execute(
                    "SELECT payload FROM chunks WHERE id=?", (chunk,)
                ).fetchone()
                if payload is None:
                    raise ValueError("missing trace chunk")
                stream = BytesIO(payload[0])
                records = _RecordReader(stream).load()
                if stream.tell() != len(payload[0]) or not isinstance(records, tuple):
                    raise ValueError("invalid trace chunk")
                cache[chunk] = records
                if len(cache) > 2:
                    cache.popitem(last=False)
            cache.move_to_end(chunk)
            records = cache[chunk]
            if not 0 <= ordinal < len(records) or not isinstance(
                records[ordinal], AFTraceEvent
            ):
                raise ValueError("invalid trace chunk event")
            count += 1
            yield records[ordinal]
        if count != expected:
            raise ValueError("trace event count does not match its export record")
    finally:
        cursor.close()


def convert(source, destination):
    """Read this prototype's indexed snapshot with fixed record constructors."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT format, metadata, options, event_count FROM trace_export"
        ).fetchall()
        if len(row) != 1 or row[0][0] != FORMAT:
            raise ValueError("unsupported indexed trace snapshot")
        _, metadata, options, count = row[0]
        if type(count) is not int or count < 0:
            raise ValueError("invalid trace event count")
        write_chrome_trace(
            destination,
            _events(connection, count),
            json.loads(metadata),
            trace_config=AFTraceConfig.from_mapping(json.loads(options)),
        )
        return count
    finally:
        connection.close()


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
