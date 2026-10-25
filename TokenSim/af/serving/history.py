"""Stable event positions while completed iterations release their trace storage."""

from TokenSim.trace.recorder import TraceRecorder
from TokenSim.trace.history import IndexedEventHistory


class StreamingTraceRecorder(TraceRecorder):
    def __init__(self, enabled, sink):
        super().__init__(enabled)
        self.events = IndexedEventHistory()
        self.sink = sink

    def record(self, event):
        super().record(event)
        if self.enabled:
            self.sink(event)

    def release_iteration(self, iteration_id):
        self.events.discard(self.iteration_event_indices.pop(iteration_id))
