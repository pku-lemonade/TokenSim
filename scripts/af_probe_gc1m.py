"""Diagnostic higher allocation threshold with automatic collection enabled."""

from contextlib import contextmanager
import gc


@contextmanager
def installed(output=None):
    previous = gc.get_threshold()
    applied = (max(previous[0], 1_000_000), *previous[1:])
    gc.set_threshold(*applied)
    try:
        yield
    finally:
        if gc.get_threshold() == applied:
            gc.set_threshold(*previous)
