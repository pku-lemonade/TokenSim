"""Diagnostic scalar tuple classification using C iterator operations."""

from contextlib import contextmanager
import inspect
import textwrap

from TokenSim.af.caching import estimates


@contextmanager
def installed(output=None):
    original = estimates.FingerprintMemo._visit
    source = textwrap.dedent(inspect.getsource(original)).replace(
        "all(type(item) in _SCALARS for item in value)",
        "_SCALAR_TYPES.issuperset(map(type, value))",
    )
    namespace = vars(estimates) | {"_SCALAR_TYPES": frozenset(estimates._SCALARS)}
    exec(compile(source, __file__, "exec"), namespace)
    estimates.FingerprintMemo._visit = namespace["_visit"]
    try:
        yield
    finally:
        estimates.FingerprintMemo._visit = original
