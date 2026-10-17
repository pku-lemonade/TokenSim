"""Opt-in timing-only experiment; complete event traces are deliberately empty."""

from contextlib import contextmanager


@contextmanager
def installed(output=None):
    from scripts import run_deepseek_mla

    original = run_deepseek_mla.simulate

    def simulate(*args, **kwargs):
        kwargs["trace"] = False
        return original(*args, **kwargs)

    run_deepseek_mla.simulate = simulate
    try:
        yield
    finally:
        run_deepseek_mla.simulate = original
