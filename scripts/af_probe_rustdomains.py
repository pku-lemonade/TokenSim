"""Install selected native kernels in the complete-run measurement harness."""

from contextlib import contextmanager
import os

from scripts.af_rust import backend


@contextmanager
def installed(output=None):
    kernels = os.environ.get("TOKENSIM_RUST_PROBE_KERNELS", "domains").split(",")
    with backend.installed(kernels):
        yield
