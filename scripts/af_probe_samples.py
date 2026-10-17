"""Single-thread wall-clock stack sampling for the diagnostic AF probe."""

from collections import defaultdict
import gc
import signal
import sys
from time import perf_counter


class StackSampler:
    """Sample Python stacks with a POSIX timer; measure collections separately.

    Signal delivery can be deferred inside a C extension, so category weights
    are approximate wall-clock measurements rather than exact native stacks.
    Existing timers or signal handlers are rejected to avoid replacing them.
    """

    def __init__(self, category, interval=0.001):
        self.category, self.interval = category, interval
        self.totals = defaultdict(float)
        self.functions = defaultdict(lambda: [0, 0.0])
        self.collections = []
        self.active_gc = None
        self.pending_gc = 0.0
        self.samples = 0

    def _collection(self, phase, info):
        now = perf_counter()
        if phase == "start":
            self.active_gc = (now, now)
        elif self.active_gc is not None:
            started, accounted = self.active_gc
            self.pending_gc += now - accounted
            self.collections.append(
                dict(
                    generation=info["generation"],
                    seconds=now - started,
                    collected=info["collected"],
                )
            )
            self.active_gc = None

    def _sample(self, _, frame):
        now = perf_counter()
        elapsed, self.previous = now - self.previous, now
        if self.active_gc is not None:
            started, accounted = self.active_gc
            self.pending_gc += now - accounted
            self.active_gc = (started, now)
        gc_elapsed = min(elapsed, self.pending_gc)
        self.pending_gc -= gc_elapsed
        self.totals["garbage_collection"] += gc_elapsed
        elapsed -= gc_elapsed
        group = None
        while frame is not None:
            code = frame.f_code
            key = (code.co_filename, code.co_firstlineno, code.co_name)
            group = self.category(key)
            if group is not None:
                break
            frame = frame.f_back
        group = group or "python_runtime"
        self.totals[group] += elapsed
        self.functions[key][0] += 1
        self.functions[key][1] += elapsed
        self.samples += 1

    def __enter__(self):
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        if self.previous_handler != signal.SIG_DFL or any(
            signal.getitimer(signal.ITIMER_REAL)
        ):
            raise RuntimeError("wall sampling requires an unused SIGALRM timer")
        self.started = self.previous = perf_counter()
        gc.callbacks.append(self._collection)
        signal.signal(signal.SIGALRM, self._sample)
        signal.setitimer(signal.ITIMER_REAL, self.interval, self.interval)
        return self

    def __exit__(self, *exception):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self._sample(None, sys._getframe(1))
        signal.signal(signal.SIGALRM, self.previous_handler)
        gc.callbacks.remove(self._collection)

    def summarize(self):
        total = sum(self.totals.values())
        return dict(
            sampled_seconds=total,
            samples=self.samples,
            interval_seconds=self.interval,
            categories={
                group: dict(seconds=seconds, percent=100 * seconds / total)
                for group, seconds in sorted(
                    self.totals.items(), key=lambda item: item[1], reverse=True
                )
            },
            functions=[
                dict(
                    file=key[0],
                    line=key[1],
                    function=key[2],
                    samples=values[0],
                    sampled_seconds=values[1],
                )
                for key, values in sorted(
                    self.functions.items(), key=lambda item: item[1][1], reverse=True
                )
            ],
            garbage_collections=self.collections,
            gc_seconds=sum(item["seconds"] for item in self.collections),
        )
