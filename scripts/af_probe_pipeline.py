"""Opt-in finalized-prefix JSON export with bounded optional I/O overlap."""

from contextlib import closing, contextmanager
from heapq import merge
import json
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread

from TokenSim.af import factory
from TokenSim.af.trace.archive import ArchivedEventHistory, ArchivedTraceRecorder
from TokenSim.af.trace.export import _record_dict


class PrefixWriter:
    def __init__(self, directory, threaded=False):
        stream = NamedTemporaryFile(
            mode="wb",
            buffering=1024 * 1024,
            dir=directory,
            prefix=".trace-prefix.",
            suffix=".tmp",
            delete=False,
        )
        self.stream, self.path = stream, Path(stream.name)
        self.encoder = json.JSONEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
            default=_record_dict,
            check_circular=False,
        )
        self.count = 0
        self.error = None
        self.queue = Queue(maxsize=4) if threaded else None
        self.thread = Thread(target=self.consume) if threaded else None
        if self.thread:
            self.thread.start()
        self.write(b'{"displayTimeUnit":"us","traceEvents":[')

    def consume(self):
        while True:
            data = self.queue.get()
            if data is None:
                return
            if self.error is None:
                try:
                    self.stream.write(data)
                except BaseException as exc:
                    self.error = exc

    def write(self, data):
        if self.error:
            raise self.error
        if self.queue is None:
            self.stream.write(data)
        else:
            self.queue.put(data)

    def append(self, events):
        buffer = bytearray()
        for event in events:
            row = _record_dict(event)
            item = dict(
                name=row["stage"] or row["operation"] or row["event_type"],
                cat=row["event_type"],
                ph="X",
                pid=1,
                tid=row["resource_id"],
                ts=row["start_at"] * 1e6,
                dur=row["service_duration"] * 1e6,
                args=row,
            )
            try:
                data = self.encoder.encode(item)
            except RecursionError:
                data = json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_record_dict,
                )
            if self.count:
                buffer.extend(b",")
            buffer.extend(data.encode("utf-8"))
            self.count += 1
            if len(buffer) >= 1024 * 1024:
                self.write(bytes(buffer))
                buffer.clear()
        if buffer:
            self.write(bytes(buffer))

    def drain(self):
        if self.thread:
            self.queue.put(None)
            self.thread.join()
            self.thread = None
        try:
            self.stream.close()
        finally:
            if self.error:
                raise self.error

    def finish(self, path, metadata):
        self.write(b'],"metadata":')
        self.write(
            json.dumps(
                dict(metadata) if metadata is not None else {},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.write(b"}")
        self.drain()
        self.path.replace(path)
        return Path(path)

    def close(self):
        try:
            self.drain()
        finally:
            self.path.unlink(missing_ok=True)


class ObservedHistory(ArchivedEventHistory):
    exported_count = 0
    invalidated = False

    def __setitem__(self, index, event):
        index = self._position(index)
        super().__setitem__(index, event)
        if index < self.exported_count:
            self.invalidated = True


class PipelinedTraceRecorder(ArchivedTraceRecorder):
    def __init__(
        self, directory, threaded=False, enabled=True, resource_capacities=None
    ):
        super().__init__(enabled, resource_capacities)
        self.events = ObservedHistory(self.archive)
        self.directory, self.threaded = directory, threaded
        self.writer = None
        self.published = False

    def _writer(self):
        if self.writer is None:
            self.writer = PrefixWriter(self.directory, self.threaded)
        return self.writer

    def _prefix(self, start, end):
        active = (
            item for item in self.events._events.items() if start <= item[0] < end
        )
        connection = self.archive._connection
        cursor = (
            None
            if connection is None
            else connection.execute(
                "SELECT position, chunk_id, ordinal FROM positions "
                "WHERE position>=? AND position<? ORDER BY position",
                (start, end),
            )
        )
        archived = (
            ()
            if cursor is None
            else (
                (position, self.archive._chunk(chunk)[ordinal])
                for position, chunk, ordinal in cursor
            )
        )
        try:
            yield from (
                event for _, event in merge(archived, active, key=lambda item: item[0])
            )
        finally:
            if cursor is not None:
                cursor.close()

    def complete_iteration(self, iteration_id):
        remaining = [
            positions[0]
            for key, positions in self.iteration_event_indices.active.items()
            if key != iteration_id and positions
        ]
        end = min(remaining, default=len(self.events))
        if (
            not self.published
            and not self.events.invalidated
            and end > self.events.exported_count
        ):
            start = self.events.exported_count
            # The finalized prefix follows global append positions, including
            # interleaved iterations already archived behind an active gap.
            with closing(self._prefix(start, end)) as finalized:
                self._writer().append(finalized)
            self.events.exported_count = end
        super().complete_iteration(iteration_id)

    def finish_profile(self, path, metadata):
        from TokenSim.af.trace.export import write_chrome_trace

        if self.events.invalidated or self.events.retained_count or self.published:
            self.events.invalidated = True
            if self.writer is not None:
                self.writer.close()
            return write_chrome_trace(path, self.events, metadata)
        result = self._writer().finish(path, metadata)
        self.published = True
        return result

    def clear(self):
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            super().clear()
            self.writer = None
            self.published = False
            self.events.exported_count = 0
            self.events.invalidated = False


@contextmanager
def installed(output, threaded=False):
    from scripts import deepseek_v3_runtime, run_deepseek_mla

    original_recorder = factory.ArchivedTraceRecorder
    original_export = deepseek_v3_runtime.write_profile
    original_cli_export = run_deepseek_mla.write_profile

    def recorder(enabled=True, resource_capacities=None):
        return PipelinedTraceRecorder(output, threaded, enabled, resource_capacities)

    def export(path, executor, metadata=None):
        if isinstance(executor.trace, PipelinedTraceRecorder):
            return executor.trace.finish_profile(
                path, metadata or {"timing": "simulated operator-table service time"}
            )
        return original_export(path, executor, metadata)

    factory.ArchivedTraceRecorder = recorder
    deepseek_v3_runtime.write_profile = run_deepseek_mla.write_profile = export
    try:
        yield
    finally:
        factory.ArchivedTraceRecorder = original_recorder
        deepseek_v3_runtime.write_profile = original_export
        run_deepseek_mla.write_profile = original_cli_export
