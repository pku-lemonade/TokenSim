"""Retain active events for validation without archiving completed iterations."""

from TokenSim.trace.history import IndexedEventHistory
from TokenSim.trace.recorder import TraceRecorder


class ValidationTraceRecorder(TraceRecorder):
    """Completion runs after final dependency patches and full validation."""

    def __init__(self, resource_capacities=None):
        super().__init__(True, resource_capacities)
        self.events = IndexedEventHistory()

    def complete_iteration(self, iteration_id):
        self.events.discard(self.iteration_event_indices.pop(iteration_id, ()))
