"""Diagnostic interval index for contiguous positions inside archive chunks."""

from contextlib import contextmanager

from TokenSim.af.trace import archive
from TokenSim.af.trace.archive_store import TraceArchiveStore, _encode


class SpanArchiveStore(TraceArchiveStore):
    def _open(self):
        fresh = self._connection is None
        connection = super()._open()
        if fresh:
            connection.executescript(
                "DROP TABLE positions;"
                "CREATE TABLE spans (start INTEGER PRIMARY KEY, stop INTEGER,"
                " iteration_id INTEGER, chunk_id INTEGER, ordinal INTEGER);"
                "CREATE INDEX by_iteration ON spans(iteration_id, start);"
                "CREATE VIEW positions AS WITH RECURSIVE expanded"
                "(position, iteration_id, chunk_id, ordinal, stop) AS ("
                "SELECT start, iteration_id, chunk_id, ordinal, stop FROM spans "
                "UNION ALL SELECT position+1, iteration_id, chunk_id, ordinal+1, stop "
                "FROM expanded WHERE position+1<stop) "
                "SELECT position, iteration_id, chunk_id, ordinal FROM expanded;"
            )
        return connection

    def append_iteration(self, iteration_id, positions, events):
        connection = self._open()
        with connection:
            connection.execute(
                "INSERT INTO iterations VALUES (?, ?)", (iteration_id, positions[0])
            )
            for start in range(0, len(positions), self.chunk_size):
                indices = positions[start : start + self.chunk_size]
                chunk_id = connection.execute(
                    "INSERT INTO chunks(payload) VALUES (?)",
                    (_encode(tuple(events[index] for index in indices)),),
                ).lastrowid
                connection.executemany(
                    "INSERT INTO spans VALUES (?, ?, ?, ?, ?)",
                    self._spans(indices, iteration_id, chunk_id),
                )

    @staticmethod
    def _spans(indices, iteration_id, chunk_id):
        if indices[-1] - indices[0] == len(indices) - 1:
            yield indices[0], indices[-1] + 1, iteration_id, chunk_id, 0
            return
        start, ordinal = indices[0], 0
        for offset in range(1, len(indices)):
            if indices[offset] != indices[offset - 1] + 1:
                yield start, indices[offset - 1] + 1, iteration_id, chunk_id, ordinal
                start, ordinal = indices[offset], offset
        yield start, indices[-1] + 1, iteration_id, chunk_id, ordinal

    def _location(self, position):
        row = (
            self._connection.execute(
                "SELECT start, stop, chunk_id, ordinal, iteration_id FROM spans "
                "WHERE start<=? ORDER BY start DESC LIMIT 1",
                (position,),
            ).fetchone()
            if self._connection is not None
            else None
        )
        if row is None or position >= row[1]:
            raise IndexError(position)
        return row[2], row[3] + position - row[0], row[4]

    def event(self, position):
        chunk_id, ordinal, _ = self._location(position)
        return self._chunk(chunk_id)[ordinal]

    def replace_event(self, position, event):
        chunk_id, ordinal, iteration_id = self._location(position)
        if event.iteration_id != iteration_id:
            raise ValueError("trace replacement must preserve iteration identity")
        events = list(self._chunk(chunk_id))
        events[ordinal] = event
        with self._connection:
            self._connection.execute(
                "UPDATE chunks SET payload=? WHERE id=?",
                (_encode(tuple(events)), chunk_id),
            )
        self._cache.pop(chunk_id, None)

    def items(self, iteration_id=None, *, start=None, end=None):
        if self._connection is None:
            return
        conditions, parameters = [], []
        for column, operator, value in (
            ("iteration_id", "=", iteration_id),
            ("stop", ">", start),
            ("start", "<", end),
        ):
            if value is not None:
                conditions.append(column + operator + "?")
                parameters.append(value)
        condition = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._connection.execute(
            "SELECT start, stop, chunk_id, ordinal FROM spans"
            + condition
            + " ORDER BY start",
            parameters,
        )
        try:
            for first, stop, chunk_id, ordinal in cursor:
                chunk = self._chunk(chunk_id)
                for position in range(
                    first if start is None else max(first, start),
                    stop if end is None else min(stop, end),
                ):
                    yield position, chunk[ordinal + position - first]
        finally:
            cursor.close()

    def positions_for(self, iteration_id):
        if self._connection is None:
            return []
        return [
            position
            for start, stop in self._connection.execute(
                "SELECT start, stop FROM spans WHERE iteration_id=? ORDER BY start",
                (iteration_id,),
            )
            for position in range(start, stop)
        ]


@contextmanager
def installed(output=None):
    original = archive.TraceArchiveStore
    archive.TraceArchiveStore = SpanArchiveStore
    try:
        yield
    finally:
        archive.TraceArchiveStore = original
