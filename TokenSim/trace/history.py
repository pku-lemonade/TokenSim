"""Stable append positions with releasable storage for active trace events."""


class IndexedEventHistory:
    """Keep append positions stable while retaining only active iterations.

    len() is the append position used by AFTraceRecorder, not the live count.
    Iteration yields live events, and retained_count reports their count.
    """

    def __init__(self):
        self._events = {}
        self._next = 0
        self.peak_retained = 0

    def __len__(self):
        return self._next

    def __iter__(self):
        return iter(self._events.values())

    def __getitem__(self, index):
        return self._events[index]

    def __setitem__(self, index, event):
        if index not in self._events:
            raise KeyError(f"cannot replace retired trace position {index}")
        self._events[index] = event

    def append(self, event):
        self._events[self._next] = event
        self._next += 1
        self.peak_retained = max(self.peak_retained, len(self._events))

    def discard(self, indices):
        for index in indices:
            del self._events[index]

    @property
    def retained_count(self):
        return len(self._events)

    def clear(self):
        self._events.clear()
        self._next = 0
        self.peak_retained = 0
