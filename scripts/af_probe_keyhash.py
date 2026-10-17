"""Diagnostic hash reuse on memoized, deeply immutable structural tuples."""

from contextlib import contextmanager
import inspect
import textwrap

from TokenSim.af.caching import estimates


class HashedTuple(tuple):
    def __new__(cls, values):
        result = super().__new__(cls, values)
        result.digest = tuple.__hash__(result)
        return result

    def __hash__(self):
        return self.digest


@contextmanager
def installed(output=None):
    original = estimates.FingerprintMemo._visit
    source = textwrap.dedent(inspect.getsource(original))
    source = source.replace("if safe:\n", "if safe:\n        key = HashedTuple(key)\n")
    namespace = vars(estimates) | {"HashedTuple": HashedTuple}
    exec(compile(source, __file__, "exec"), namespace)
    estimates.FingerprintMemo._visit = namespace["_visit"]
    try:
        yield
    finally:
        estimates.FingerprintMemo._visit = original
