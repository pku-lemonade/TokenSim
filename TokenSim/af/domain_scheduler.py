"""Atomic FIFO reservations. Waiting requests never hold a subset of resources."""

from dataclasses import dataclass
import math

import simpy

from TokenSim.errors import ConfigurationError


@dataclass(eq=False)
class DomainRequest:
    keys: tuple[str, ...]
    priority: tuple
    event: simpy.Event
    acquired: bool = False
    cancelled: bool = False


class DomainScheduler:
    def __init__(self, env, resource_ids):
        self.env = env
        self.resource_ids = frozenset(resource_ids)
        self.owners = {}
        self.waiters = []
        self._scheduled = False

    def request(self, keys, identity):
        keys = tuple(sorted(set(keys)))
        if not keys or not set(keys) <= self.resource_ids:
            raise ConfigurationError(f"invalid communication domain request: {keys!r}")
        if any(type(part) not in (int, str) for part in identity):
            raise ConfigurationError(
                "domain request identity must contain stable integer/string components"
            )
        stable_identity = tuple((type(part).__name__, part) for part in identity)
        request = DomainRequest(keys, (self.env.now, stable_identity), self.env.event())
        if any(item.priority == request.priority for item in self.waiters):
            raise ConfigurationError(
                "concurrent domain requests need unique stable logical identities"
            )
        self.waiters.append(request)
        self._wake()
        return request

    def _wake(self):
        if not self._scheduled:
            self._scheduled = True
            # A later priority than normal SimPy events gathers every release
            # and readiness event at this timestamp before tie arbitration.
            event = simpy.Event(self.env)
            event._ok, event._value = True, None
            event.callbacks.append(lambda _: self._arbitrate())
            self.env.schedule(event, priority=2, delay=0)

    def _arbitrate(self):
        self._scheduled = False
        blocked = set()
        remaining = []
        for request in sorted(self.waiters, key=lambda item: item.priority):
            if request.cancelled:
                continue
            if blocked.intersection(request.keys) or any(
                key in self.owners for key in request.keys
            ):
                blocked.update(request.keys)
                remaining.append(request)
                continue
            request.acquired = True
            for key in request.keys:
                self.owners[key] = request
            request.event.succeed()
        self.waiters = remaining

    def release(self, request):
        if request.acquired:
            for key in request.keys:
                if self.owners.get(key) is request:
                    del self.owners[key]
            request.acquired = False
        else:
            request.cancelled = True
            self.waiters = [item for item in self.waiters if item is not request]
        self._wake()

    def run(self, keys, duration, identity):
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ConfigurationError("domain service must be finite and nonnegative")
        ready = self.env.now
        if duration == 0:
            return ready, ready, ready
        request = self.request(keys, identity)
        try:
            yield request.event
            start = self.env.now
            yield self.env.timeout(duration)
            return ready, start, self.env.now
        finally:
            self.release(request)
