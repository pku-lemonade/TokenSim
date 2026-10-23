"""Indexed archive snapshots with bounded readback and atomic publication.

The snapshot requires compatible TokenSim record definitions for readback.
Only a fixed set of project record constructors is accepted by the converter.
The published database is independent of subsequent live archive changes.
"""

from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict
from io import BytesIO
from itertools import islice
import json
from pathlib import Path
import pickle
import sqlite3
from tempfile import NamedTemporaryFile

from TokenSim.trace.archive import ArchivedTraceRecorder
from TokenSim.trace.archive_store import TraceArchiveStore, _encode
from TokenSim.trace.config import TraceConfig
from TokenSim.trace.export import write_chrome_trace
from TokenSim.trace.events import TraceEvent, RECORD_TYPES
from TokenSim.trace.pipeline import PipelinedTraceRecorder

FORMAT = "tokensim-indexed-trace-v1"
LEGACY_FORMAT = "tokensim-indexed-trace-probe-v1"


class _RecordReader(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return RECORD_TYPES[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"unsupported trace record: {module}.{name}")


class ArchiveTraceRecorder(PipelinedTraceRecorder):
    trace_format = "archive"

    def complete_iteration(self, iteration_id):
        ArchivedTraceRecorder.complete_iteration(self, iteration_id)

    def _write_prepared_trace(self, output, metadata, options):
        return write_archive(
            output.with_suffix(".sqlite3"), self.events, metadata, options
        )


def _append_events(connection, events):
    """Rebuild edited or partial histories using one bounded chunk at a time."""
    events, count = iter(events), 0
    while chunk := tuple(islice(events, 1024)):
        with connection:
            chunk_id = connection.execute(
                "INSERT INTO chunks(payload) VALUES (?)", (_encode(chunk),)
            ).lastrowid
            connection.executemany(
                "INSERT INTO positions VALUES (?, ?, ?, ?)",
                (
                    (count + ordinal, event.iteration_id, chunk_id, ordinal)
                    for ordinal, event in enumerate(chunk)
                ),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO iterations VALUES (?, ?)",
                ((event.iteration_id, count + i) for i, event in enumerate(chunk)),
            )
        count += len(chunk)
    return count


def write_archive(output, events, metadata, options):
    """Snapshot finalized archives directly; rebuild partial or edited histories."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
        connection = sqlite3.connect(temporary)
        try:
            store = getattr(events, "store", None)
            if (
                store is not None
                and not getattr(events, "invalidated", True)
                and not events.retained_count
            ):
                store._open().backup(connection)
                count = len(events)
            else:
                with closing(TraceArchiveStore()) as empty:
                    empty._open().backup(connection)
                connection.execute("PRAGMA cache_size=-2048")
                connection.execute("PRAGMA temp_store=FILE")
                count = _append_events(connection, events)
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
                        count,
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
                records[ordinal], TraceEvent
            ):
                raise ValueError("invalid trace chunk event")
            count += 1
            yield records[ordinal]
        if count != expected:
            raise ValueError("trace event count does not match its export record")
    finally:
        cursor.close()


def convert(source, destination):
    """Read a versioned indexed snapshot with fixed record constructors."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT format, metadata, options, event_count FROM trace_export"
        ).fetchall()
        if len(row) != 1 or row[0][0] not in (FORMAT, LEGACY_FORMAT):
            raise ValueError("unsupported indexed trace snapshot")
        _, metadata, options, count = row[0]
        if type(count) is not int or count < 0:
            raise ValueError("invalid trace event count")
        write_chrome_trace(
            destination,
            _events(connection, count),
            json.loads(metadata),
            trace_config=TraceConfig.from_mapping(json.loads(options)),
        )
        return count
    finally:
        connection.close()
