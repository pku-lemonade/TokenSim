"""Private temporary trace chunks and disk indexes with bounded read caching."""

from collections import OrderedDict
from pathlib import Path
import pickle
import sqlite3
from tempfile import TemporaryDirectory
import weakref


def _encode(events):
    return pickle.dumps(events, protocol=pickle.HIGHEST_PROTOCOL)


def _cleanup(connection, directory):
    try:
        connection.close()
    finally:
        directory.cleanup()


class TraceArchiveStore:
    """Archive trusted in-process records; never open user-supplied pickles.

    Chunk and position indexes live in SQLite, including iteration membership.
    Synchronous transactions provide backpressure without an unbounded queue.
    """

    def __init__(self, *, chunk_size=256, cache_chunks=2):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("trace archive chunk size must be positive")
        if type(cache_chunks) is not int or cache_chunks < 1:
            raise ValueError("trace archive cache size must be positive")
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        self._cache = OrderedDict()
        self._connection = None
        self._finalizer = None
        self.path = None

    def _open(self):
        if self._connection is not None:
            return self._connection
        directory = TemporaryDirectory(prefix="tokensim-af-trace-")
        path = Path(directory.name) / "trace.sqlite3"
        connection = None
        try:
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.executescript(
                "CREATE TABLE chunks (id INTEGER PRIMARY KEY, payload BLOB);"
                "CREATE TABLE positions (position INTEGER PRIMARY KEY,"
                " iteration_id INTEGER, chunk_id INTEGER, ordinal INTEGER);"
                "CREATE INDEX by_iteration ON positions(iteration_id, position);"
                "CREATE TABLE iterations (iteration_id INTEGER PRIMARY KEY,"
                " first_position INTEGER);"
                "CREATE INDEX by_first_position ON iterations(first_position);"
            )
        except BaseException:
            if connection is not None:
                connection.close()
            directory.cleanup()
            raise
        self._connection = connection
        self.path = path
        self._finalizer = weakref.finalize(self, _cleanup, connection, directory)
        return connection

    def append_iteration(self, iteration_id, positions, events):
        connection = self._open()
        with connection:
            connection.execute(
                "INSERT INTO iterations VALUES (?, ?)",
                (iteration_id, positions[0]),
            )
            for start in range(0, len(positions), self.chunk_size):
                indices = positions[start : start + self.chunk_size]
                payload = _encode(tuple(events[index] for index in indices))
                chunk_id = connection.execute(
                    "INSERT INTO chunks(payload) VALUES (?)", (payload,)
                ).lastrowid
                connection.executemany(
                    "INSERT INTO positions VALUES (?, ?, ?, ?)",
                    (
                        (index, iteration_id, chunk_id, ordinal)
                        for ordinal, index in enumerate(indices)
                    ),
                )

    def _chunk(self, chunk_id):
        if chunk_id in self._cache:
            self._cache.move_to_end(chunk_id)
            return self._cache[chunk_id]
        payload = self._connection.execute(
            "SELECT payload FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone()[0]
        # Only this object's private, internally generated database is read.
        events = pickle.loads(payload)
        self._cache[chunk_id] = events
        if len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return events

    def event(self, position):
        row = (
            self._connection.execute(
                "SELECT chunk_id, ordinal FROM positions WHERE position=?",
                (position,),
            ).fetchone()
            if self._connection is not None
            else None
        )
        if row is None:
            raise IndexError(position)
        return self._chunk(row[0])[row[1]]

    def replace_event(self, position, event):
        chunk_id, ordinal, iteration_id = self._connection.execute(
            "SELECT chunk_id, ordinal, iteration_id FROM positions WHERE position=?",
            (position,),
        ).fetchone()
        if event.iteration_id != iteration_id:
            raise ValueError("trace replacement must preserve iteration identity")
        events = list(self._chunk(chunk_id))
        events[ordinal] = event
        payload = _encode(tuple(events))
        with self._connection:
            self._connection.execute(
                "UPDATE chunks SET payload=? WHERE id=?", (payload, chunk_id)
            )
        self._cache.pop(chunk_id, None)

    def items(self, iteration_id=None, *, start=None, end=None):
        if self._connection is None:
            return
        conditions, parameters = [], []
        for column, operator, value in (
            ("iteration_id", "=", iteration_id),
            ("position", ">=", start),
            ("position", "<", end),
        ):
            if value is not None:
                conditions.append(column + operator + "?")
                parameters.append(value)
        condition = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._connection.execute(
            "SELECT position, chunk_id, ordinal FROM positions"
            + condition
            + " ORDER BY position",
            parameters,
        )
        try:
            for position, chunk_id, ordinal in cursor:
                yield position, self._chunk(chunk_id)[ordinal]
        finally:
            cursor.close()

    def positions_for(self, iteration_id):
        if self._connection is None:
            return []
        return [
            row[0]
            for row in self._connection.execute(
                "SELECT position FROM positions WHERE iteration_id=? ORDER BY position",
                (iteration_id,),
            )
        ]

    def iterations(self):
        if self._connection is not None:
            yield from self._connection.execute(
                "SELECT iteration_id, first_position FROM iterations ORDER BY first_position"
            )

    def contains_iteration(self, iteration_id):
        return (
            self._connection is not None
            and self._connection.execute(
                "SELECT 1 FROM iterations WHERE iteration_id=?", (iteration_id,)
            ).fetchone()
            is not None
        )

    @property
    def iteration_count(self):
        if self._connection is None:
            return 0
        return self._connection.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]

    @property
    def cached_event_count(self):
        return sum(len(chunk) for chunk in self._cache.values())

    def close(self):
        self._cache.clear()
        if self._finalizer is not None:
            self._finalizer()
        self._connection = self._finalizer = self.path = None
