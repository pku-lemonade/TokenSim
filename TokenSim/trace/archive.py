"""Finalize AF iterations to disk while keeping full ordered trace readback."""

from collections.abc import Mapping, Sequence
from heapq import merge
from operator import index as integer_index

from TokenSim.trace.archive_store import TraceArchiveStore
from TokenSim.trace.history import IndexedEventHistory
from TokenSim.trace.recorder import TraceRecorder
from TokenSim.errors import SimulationStateError


class ArchivedEventHistory(IndexedEventHistory, Sequence):
    """Mutable positions with disk readback for completed event snapshots."""

    def __init__(self, store):
        super().__init__()
        self.store = store

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = self._position(index)
        if index in self._events:
            return self._events[index]
        return self.store.event(index)

    def _position(self, index):
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index

    def __setitem__(self, index, event):
        index = self._position(index)
        if index in self._events:
            super().__setitem__(index, event)
        else:
            self.store.replace_event(index, event)

    def __iter__(self):
        items = merge(
            self.store.items(), self._events.items(), key=lambda item: item[0]
        )
        return (event for _, event in items)

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))


class ArchivedIterationIndex(Mapping):
    """Only in-flight position lists live in Python memory."""

    def __init__(self, store):
        self.active = {}
        self.store = store

    def __getitem__(self, iteration_id):
        if iteration_id in self.active:
            return self.active[iteration_id]
        positions = self.store.positions_for(iteration_id)
        if not positions:
            raise KeyError(iteration_id)
        return positions

    def __iter__(self):
        active = sorted(
            ((key, indices[0]) for key, indices in self.active.items()),
            key=lambda item: item[1],
        )
        return (
            key
            for key, _ in merge(
                self.store.iterations(), active, key=lambda item: item[1]
            )
        )

    def __len__(self):
        return len(self.active) + self.store.iteration_count

    def setdefault(self, iteration_id, default):
        return self.active.setdefault(iteration_id, default)

    def clear(self):
        self.active.clear()


class ArchivedTraceRecorder(TraceRecorder):
    """Retain active iterations and a readback cache bounded by concurrency.

    Completion is called after dependency patching and invariant validation.
    Completed records are snapshots; explicit full-history materialization can
    still consume memory proportional to the requested history.
    """

    def __init__(self, enabled=True, resource_capacities=None):
        super().__init__(enabled, resource_capacities)
        self.archive = TraceArchiveStore()
        self.events = ArchivedEventHistory(self.archive)
        self.iteration_event_indices = ArchivedIterationIndex(self.archive)

    def record(self, event):
        if (
            self.enabled
            and event.iteration_id not in self.iteration_event_indices.active
            and self.archive.contains_iteration(event.iteration_id)
        ):
            raise SimulationStateError("cannot append to a completed trace iteration")
        super().record(event)
        # One hot chunk per concurrent iteration prevents interleaved exports
        # from repeatedly decoding the same chunks. This bound tracks peak
        # concurrency, never the number of completed iterations.
        self.archive.cache_chunks = max(
            self.archive.cache_chunks, len(self.iteration_event_indices.active)
        )

    def complete_iteration(self, iteration_id):
        positions = self.iteration_event_indices.active.get(iteration_id)
        if not positions:
            return
        try:
            self.archive.append_iteration(iteration_id, positions, self.events)
        except Exception as exc:
            raise SimulationStateError(
                f"could not archive AF trace iteration {iteration_id}: {exc}"
            ) from exc
        self.events.discard(positions)
        del self.iteration_event_indices.active[iteration_id]

    def events_for(self, iteration_id):
        if iteration_id in self.iteration_event_indices.active:
            return super().events_for(iteration_id)
        return tuple(event for _, event in self.archive.items(iteration_id))

    def clear(self):
        self.archive.close()
        self.archive.cache_chunks = 2
        super().clear()
