"""Opt-in SimPy condition callback allocation experiment.

Adapted from SimPy 4.1.2; see third_party/licenses/SimPy-MIT.txt.
Event creation, scheduling, callback order and cleanup remain unchanged.
"""

from contextlib import contextmanager

import simpy
from simpy.events import AllOf, AnyOf, Condition, ConditionValue, Event


@contextmanager
def installed(output=None):
    if simpy.__version__ != "4.1.2":
        raise RuntimeError("condition probe requires SimPy 4.1.2")
    original_init = Condition.__init__
    original_cleanup = Condition._remove_check_callbacks
    original_check = Condition._check

    def initialize(self, env, evaluate, events):
        if (
            type(self) not in (Condition, AllOf, AnyOf)
            or getattr(self._check, "__func__", None) is not original_check
        ):
            return original_init(self, env, evaluate, events)
        Event.__init__(self, env)
        self._evaluate = evaluate
        self._events = tuple(events)
        self._count = 0
        if not self._events:
            self.succeed(ConditionValue())
            return
        for event in self._events:
            if self.env != event.env:
                raise ValueError(
                    "It is not allowed to mix events from different environments"
                )
        check = self._check
        for event in self._events:
            if event.callbacks is None:
                check(event)
            else:
                event.callbacks.append(check)
        self.callbacks.append(self._build_value)

    def remove_checks(self):
        if (
            type(self) not in (Condition, AllOf, AnyOf)
            or getattr(self._check, "__func__", None) is not original_check
        ):
            return original_cleanup(self)
        check = self._check
        for event in self._events:
            if event.callbacks and check in event.callbacks:
                event.callbacks.remove(check)
            if isinstance(event, Condition):
                event._remove_check_callbacks()

    Condition.__init__ = initialize
    Condition._remove_check_callbacks = remove_checks
    try:
        yield
    finally:
        Condition.__init__ = original_init
        Condition._remove_check_callbacks = original_cleanup
