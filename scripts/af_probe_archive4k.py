"""Opt-in 4,096-event archive blocks."""

from contextlib import contextmanager

from scripts import af_probe_archive


@contextmanager
def installed(output=None):
    with af_probe_archive.installed(output, chunk_size=4096):
        yield
