"""Opt-in SimPy 4.1.2 dispatch-loop experiment with stock fallbacks.

Adapted from SimPy 4.1.2; see third_party/licenses/SimPy-MIT.txt.
"""

from contextlib import contextmanager
from heapq import heappop

import simpy
from simpy.core import StopSimulation


@contextmanager
def installed(output=None):
    if simpy.__version__ != "4.1.2":
        raise RuntimeError("dispatch probe requires SimPy 4.1.2")
    original_run = simpy.Environment.run
    original_step = simpy.Environment.step

    def run(self, until=None):
        if (
            until is not None
            or type(self) is not simpy.Environment
            or getattr(self.step, "__func__", None) is not original_step
            or original_run.__module__ != "simpy.core"
            or original_step.__module__ != "simpy.core"
        ):
            return original_run(self, until)
        queue = self._queue
        try:
            while queue:
                self._now, _, _, event = heappop(queue)
                callbacks, event.callbacks = event.callbacks, None
                try:
                    for callback in callbacks:
                        callback(event)
                except StopSimulation:
                    event.callbacks = callbacks[callbacks.index(callback) + 1 :]
                    self.schedule(event, -1)
                    raise
                if not event._ok and not hasattr(event, "_defused"):
                    exc = type(event._value)(*event._value.args)
                    exc.__cause__ = event._value
                    raise exc
                del event, callbacks
            return None
        except StopSimulation as exc:
            return exc.args[0]

    simpy.Environment.run = run
    try:
        yield
    finally:
        simpy.Environment.run = original_run
