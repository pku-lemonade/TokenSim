"""Stable event positions while completed iterations release their trace storage."""

from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.af.trace.history import IndexedEventHistory


class StreamingTraceRecorder(AFTraceRecorder):
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
