"""Encode finalized trace prefixes while their events are still in memory."""

from contextlib import closing
from heapq import merge
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import weakref

from TokenSim.af.trace.archive import ArchivedEventHistory, ArchivedTraceRecorder
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import _encode_event, _event_encoder
from TokenSim.errors import ConfigurationError


def _cleanup(stream, path):
    try:
        stream.close()
    finally:
        path.unlink(missing_ok=True)


class _PrefixWriter:
    """Bounded synchronous output with atomic publication and abandonment cleanup."""

    def __init__(self, directory, options):
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = NamedTemporaryFile(
            mode="wb",
            buffering=1024 * 1024,
            dir=directory,
            prefix=".trace-prefix.",
            suffix=".tmp",
            delete=False,
        )
        self.path = Path(self.stream.name)
        self._finalizer = weakref.finalize(self, _cleanup, self.stream, self.path)
        self.options, self.encoder, self.count = options, _event_encoder(), 0
        self.stream.write(b'{"displayTimeUnit":"us","traceEvents":[')

    def append(self, events):
        buffer = bytearray()
        options, encoder = self.options, self.encoder
        includes = options.includes if options.filtered else None
        for event in events:
            if includes is not None and not includes(event):
                continue
            data = _encode_event(event, options, encoder).encode("utf-8")
            if self.count:
                buffer.extend(b",")
            buffer.extend(data)
            self.count += 1
            if len(buffer) >= 1024 * 1024:
                self.stream.write(buffer)
                buffer.clear()
        if buffer:
            self.stream.write(buffer)

    def finish(self, output, metadata):
        try:
            self.stream.write(b'],"metadata":')
            self.stream.write(
                json.dumps(
                    dict(metadata) if metadata is not None else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            self.stream.write(b"}")
            self.stream.close()
            self.path.replace(output)
        finally:
            self.close()
        return output

    def close(self):
        self._finalizer()


class _PreparedHistory(ArchivedEventHistory):
    """Invalidate prepared bytes on replacements or exposed archived records."""

    def __init__(self, store, recorder):
        super().__init__(store)
        self._recorder = weakref.ref(recorder)
        self.encoded_count = 0
        self.invalidated = False

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        position = self._position(index)
        # Nested details can be mutable even though the event is frozen.
        # A caller receiving archived objects may change their cached payload.
        if position not in self._events:
            self.invalidated = True
            return self.store.event(position)
        return self._events[position]

    def __iter__(self):
        self.invalidated = True
        return super().__iter__()

    def __setitem__(self, index, event):
        position = self._position(index)
        super().__setitem__(position, event)
        if position < self.encoded_count:
            self.invalidated = True

    def _write_prepared_trace(self, output, metadata, options):
        recorder = self._recorder()
        return (
            None
            if recorder is None
            else recorder._write_prepared_trace(output, metadata, options)
        )


class PipelinedTraceRecorder(ArchivedTraceRecorder):
    """Keep complete readback and prepare ordered JSON at iteration completion.

    Only a contiguous prefix with no active iteration is safe to encode.
    Partial exports, changed options and subsequent edits use archive readback.
    This recorder needs an intended output directory to prepare files there.
    """

    def __init__(self, directory, *, trace_config=None, resource_capacities=None):
        if trace_config is not None and not isinstance(trace_config, AFTraceConfig):
            raise ConfigurationError("trace_config must be an AFTraceConfig")
        if trace_config is not None and trace_config.mode != "full":
            raise ConfigurationError("trace preparation requires full mode")
        super().__init__(True, resource_capacities)
        self.options = trace_config or AFTraceConfig()
        self.directory = Path(directory).resolve()
        self.archive.chunk_size = 1024
        self.events = _PreparedHistory(self.archive, self)
        self.writer = None
        self.published = False

    def _writer(self):
        if self.writer is None:
            self.writer = _PrefixWriter(self.directory, self.options)
        return self.writer

    def _prefix(self, start, end):
        active = (
            item for item in self.events._events.items() if start <= item[0] < end
        )
        with closing(self.archive.items(start=start, end=end)) as archived:
            yield from (
                event for _, event in merge(archived, active, key=lambda item: item[0])
            )

    def complete_iteration(self, iteration_id):
        remaining = (
            positions[0]
            for key, positions in self.iteration_event_indices.active.items()
            if key != iteration_id and positions
        )
        end = min(remaining, default=len(self.events))
        if (
            not self.published
            and not self.events.invalidated
            and end > self.events.encoded_count
        ):
            try:
                with closing(self._prefix(self.events.encoded_count, end)) as events:
                    self._writer().append(events)
                self.events.encoded_count = end
            except BaseException:
                self._discard_prefix()
                raise
        try:
            super().complete_iteration(iteration_id)
        except BaseException:
            self._discard_prefix()
            raise

    def _discard_prefix(self):
        self.events.invalidated = True
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _write_prepared_trace(self, output, metadata, options):
        if (
            self.events.invalidated
            or self.events.retained_count
            or self.published
            or self.options != options
            or output.parent.resolve() != self.directory
        ):
            self._discard_prefix()
            return None
        try:
            return self._writer().finish(output, metadata)
        finally:
            self.published = True

    def events_for(self, iteration_id):
        if iteration_id not in self.iteration_event_indices.active:
            self.events.invalidated = True
        return super().events_for(iteration_id)

    def clear(self):
        try:
            self._discard_prefix()
        finally:
            super().clear()
            self.published = False
            self.events.invalidated = False
            self.events.encoded_count = 0
