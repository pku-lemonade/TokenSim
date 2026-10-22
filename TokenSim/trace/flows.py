"""Bounded disk indexing of exported dependency endpoints and track metadata."""

import sqlite3
from tempfile import TemporaryDirectory


class FlowIndex:
    def __init__(self):
        self.directory = TemporaryDirectory(prefix="tokensim-trace-flows-")
        self.connection = sqlite3.connect(self.directory.name + "/flows.sqlite3")
        self.connection.execute("PRAGMA cache_size=-256")
        self.connection.executescript(
            "CREATE TABLE nodes(id TEXT PRIMARY KEY, track INTEGER, start REAL, end REAL);"
            "CREATE TABLE edges(parent TEXT, child TEXT);"
        )
        self.tracks = {}

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.connection.close()
        self.directory.cleanup()

    def add(self, event):
        name = event.resource_id
        metadata = None
        if name not in self.tracks:
            self.tracks[name] = len(self.tracks) + 1
            metadata = dict(
                ph="M",
                name="thread_name",
                pid=1,
                tid=self.tracks[name],
                args={"name": name},
            )
        track = self.tracks[name]
        if event.node_id:
            self.connection.execute(
                "INSERT INTO nodes VALUES (?, ?, ?, ?)",
                (event.node_id, track, event.start_at, event.end_at),
            )
            self.connection.executemany(
                "INSERT INTO edges VALUES (?, ?)",
                ((key, event.node_id) for key in event.predecessor_ids),
            )
        return track, metadata

    def flows(self):
        rows = self.connection.execute(
            "SELECT p.track, p.end, c.track, c.start FROM edges e "
            "JOIN nodes p ON p.id=e.parent JOIN nodes c ON c.id=e.child "
            "ORDER BY e.rowid"
        )
        for index, (source, end, target, start) in enumerate(rows):
            yield dict(
                name="dependency",
                cat="dependency",
                ph="s",
                id=index,
                pid=1,
                tid=source,
                ts=end * 1e6,
            )
            yield dict(
                name="dependency",
                cat="dependency",
                ph="f",
                id=index,
                pid=1,
                tid=target,
                ts=start * 1e6,
                bp="e",
            )
