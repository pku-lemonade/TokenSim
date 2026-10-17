"""Diagnostic adapter retaining DomainScheduler's event timing and public records."""

from TokenSim.af.communication.resources import DomainRequest, DomainScheduler
from TokenSim.errors import ConfigurationError


def scheduler_type(native):
    class RustDomainScheduler(DomainScheduler):
        def __init__(self, env, resource_ids):
            self.env = env
            self.resource_ids = frozenset(resource_ids)
            self._scheduled = False
            self._native = native.DomainQueues()

        def request(self, keys, identity):
            keys = tuple(sorted(set(keys)))
            if not keys or not set(keys) <= self.resource_ids:
                raise ConfigurationError(
                    f"invalid communication domain request: {keys!r}"
                )
            if any(type(part) not in (int, str) for part in identity):
                raise ConfigurationError(
                    "domain request identity must contain stable integer/string components"
                )
            identity = tuple(identity)
            stable = tuple((type(part).__name__, part) for part in identity)
            request = DomainRequest(keys, (self.env.now, stable), self.env.event())
            try:
                request._rust_serial = self._native.submit(
                    request, keys, self.env.now, identity
                )
            except (ValueError, OverflowError, UnicodeError) as exc:
                raise ConfigurationError(str(exc)) from exc
            self._wake()
            return request

        def _arbitrate(self):
            self._scheduled = False
            for request in self._native.arbitrate():
                request.acquired = True
                request.event.succeed()

        def _front(self, key):
            return self._native.peek(key)

        @property
        def owners(self):
            return self._native.owners

        @property
        def waiters(self):
            return self._native.waiters

        def release(self, request):
            self._native.release(request._rust_serial)
            if request.acquired:
                request.acquired = False
            else:
                request.cancelled = True
            self._wake()

    return RustDomainScheduler
