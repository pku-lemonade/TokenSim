"""Host garbage collection policy for allocation-heavy exact simulations."""

from contextlib import contextmanager
import gc
from threading import RLock


# Automatic collection stays enabled; only its allocation trigger is increased.
_ALLOCATION_THRESHOLD = 250_000
_lock = RLock()
_active_runs = 0
_saved_thresholds = None


@contextmanager
def _simulation_gc():
    global _active_runs, _saved_thresholds
    with _lock:
        if _active_runs == 0:
            previous = gc.get_threshold()
            _saved_thresholds = None
            if gc.isenabled() and 0 < previous[0] < _ALLOCATION_THRESHOLD:
                applied = (_ALLOCATION_THRESHOLD, *previous[1:])
                gc.set_threshold(*applied)
                _saved_thresholds = previous, applied
        _active_runs += 1
    try:
        yield
    finally:
        with _lock:
            _active_runs -= 1
            if _active_runs == 0 and _saved_thresholds is not None:
                previous, applied = _saved_thresholds
                _saved_thresholds = None
                # Respect explicit policy changes made by the caller during a run.
                if gc.get_threshold() == applied:
                    gc.set_threshold(*previous)


def run_environment(environment, until=None):
    """Run an unchanged event loop with less frequent automatic collection.

    The process-wide threshold is shared by overlapping calls and restored when
    the last call exits, including errors and bounded stop/resume runs.
    Disabled collection and higher caller thresholds are preserved.
    """
    with _simulation_gc():
        if until is None:
            return environment.run()
        return environment.run(until=until)
