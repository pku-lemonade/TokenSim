"""Group-aware external storage and queued DMA primitives for native KV."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import simpy

from TokenSim.errors import SimulationStateError
from TokenSim.config.validation import integer, number


@dataclass
class StoredPage:
    key: tuple
    bytes: int
    references: int = 0
    ready: bool = False


class NativePageStore:
    """Bounded LRU whose writes/reads pin pages until DMA completion.

    SimpleCPU supplies its packed backing stride as allocation_unit. Mooncake
    supplies actual per-group object bytes and uses one shared node budget.
    """

    def __init__(self, capacity_bytes, *, allocation_unit=None, scope="shared"):
        self.capacity_bytes = integer(capacity_bytes, "capacity_bytes", minimum=1)
        if allocation_unit is not None:
            integer(allocation_unit, "allocation_unit", minimum=1)
        self.allocation_unit = allocation_unit
        self.scope = scope
        self.pages = {}
        self.lru = OrderedDict()
        self.used_bytes = self.evictions = self.peak_used_bytes = 0

    def allocation_bytes(self, payload_bytes):
        return self.allocation_unit or payload_bytes

    def lookup(self, key):
        page = self.pages.get(key)
        if page is None or not page.ready:
            return None
        self.lru.move_to_end(key)
        return page

    def reserve_write(self, key, payload_bytes):
        integer(payload_bytes, "payload_bytes", minimum=1)
        existing = self.pages.get(key)
        if existing is not None:
            return None
        size = self.allocation_bytes(payload_bytes)
        if size > self.capacity_bytes:
            return None
        while self.used_bytes + size > self.capacity_bytes:
            candidate = next(
                (k for k in self.lru if self.pages[k].references == 0), None
            )
            if candidate is None:
                return None
            old = self.pages.pop(candidate)
            self.lru.pop(candidate)
            self.used_bytes -= old.bytes
            self.evictions += 1
        page = StoredPage(key, size, references=1)
        self.pages[key] = page
        self.lru[key] = None
        self.used_bytes += size
        self.peak_used_bytes = max(self.peak_used_bytes, self.used_bytes)
        return page

    def finish_write(self, page):
        if self.pages.get(page.key) is not page:
            raise SimulationStateError(
                "a pinned external page was evicted during write"
            )
        page.ready = True
        self.release(page)

    def acquire(self, page):
        if self.pages.get(page.key) is not page or not page.ready:
            raise SimulationStateError("external read needs a ready resident page")
        page.references += 1
        self.lru.move_to_end(page.key)

    def release(self, page):
        if page.references <= 0:
            raise SimulationStateError("invalid external page release")
        page.references -= 1

    def validate(self):
        if (
            self.used_bytes != sum(p.bytes for p in self.pages.values())
            or self.used_bytes > self.capacity_bytes
        ):
            raise SimulationStateError("external storage byte conservation failed")
        if set(self.pages) != set(self.lru) or any(
            p.references < 0 for p in self.pages.values()
        ):
            raise SimulationStateError(
                "external storage reference/LRU conservation failed"
            )
        return True

    def describe(self):
        return {
            "scope": self.scope,
            "capacity_bytes": self.capacity_bytes,
            "allocation_unit_bytes": self.allocation_unit,
            "used_bytes": self.used_bytes,
            "peak_used_bytes": self.peak_used_bytes,
            "resident_pages": len(self.pages),
            "pinned_pages": sum(p.references > 0 for p in self.pages.values()),
            "evictions": self.evictions,
        }


class QueuedKVLink:
    """One shared directional link; independent callers contend in one queue."""

    def __init__(self, env, bandwidth_bytes_per_s, *, utilization=1.0, latency_us=0):
        number(bandwidth_bytes_per_s, "bandwidth_bytes_per_s", positive=True)
        number(utilization, "utilization", positive=True, maximum=1)
        number(latency_us, "latency_us")
        self.env = env
        self.bandwidth_bytes_per_s = bandwidth_bytes_per_s
        self.utilization = utilization
        self.latency_seconds = latency_us / 1e6
        self.queue = simpy.Resource(env, capacity=1)
        self.bytes = self.jobs = 0
        self.queue_wait_seconds = self.service_seconds = 0.0

    def transfer(self, bytes_):
        integer(bytes_, "transfer bytes")
        issued = self.env.now
        with self.queue.request() as slot:
            yield slot
            self.queue_wait_seconds += self.env.now - issued
            duration = self.latency_seconds + bytes_ / (
                self.bandwidth_bytes_per_s * self.utilization
            )
            self.jobs += 1
            self.bytes += bytes_
            self.service_seconds += duration
            yield self.env.timeout(duration)

    def describe(self):
        return {
            "theoretical_bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "utilization_assumption": self.utilization,
            "bytes": self.bytes,
            "jobs": self.jobs,
            "queue_wait_seconds": self.queue_wait_seconds,
            "service_seconds": self.service_seconds,
        }
