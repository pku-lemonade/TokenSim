"""Single-process pipeline variant with a bounded byte-writing thread."""

from contextlib import contextmanager

from scripts import af_probe_pipeline


@contextmanager
def installed(output):
    with af_probe_pipeline.installed(output, threaded=True):
        yield
